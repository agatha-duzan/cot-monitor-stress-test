#!/usr/bin/env python3
"""Run intervention sweep with extended thinking (CoT) enabled.

For each significant condition from the CoT rerun, re-run with 4 different
system-prompt interventions to test if they degrade monitorability while
preserving behavior shifts.

Usage:
    python -m multidomain_pilots.interventions.run --test
    python -m multidomain_pilots.interventions.run --dry-run
    python -m multidomain_pilots.interventions.run --run
    python -m multidomain_pilots.interventions.run --run --resume
    python -m multidomain_pilots.interventions.run --extract
    python -m multidomain_pilots.interventions.run --analyze
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT / "multidomain_pilots" / "interventions" / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"

MODEL = "claude-sonnet-4-5-20250929"
THINKING_BUDGET = 10000
MAX_TOKENS = 16000
CONCURRENCY = 12
N_SAMPLES = 20

# Import intervention config
from multidomain_pilots.interventions.config import (
    INTERVENTIONS,
    SIGNIFICANT_WITH_COT,
    get_system_prompt,
)


# ── Domain config loading ────────────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text().strip()


def load_domain_config(domain: str):
    config_path = PROJECT / "multidomain_pilots" / domain / "config.py"
    spec = importlib.util.spec_from_file_location(f"{domain}_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_essay_config():
    sys.path.insert(0, str(PROJECT / "setting5_essay_grading"))
    from conditions import CONDITIONS as ESSAY_CONDITIONS
    from prompts import FRAMINGS as ESSAY_FRAMINGS, build_prompt
    sys.path.pop(0)
    essays = {
        "essay_a": load_text(PROJECT / "setting5_essay_grading" / "essays" / "essay_a.txt"),
        "essay_b": load_text(PROJECT / "setting5_essay_grading" / "essays" / "essay_b.txt"),
    }
    return ESSAY_CONDITIONS, ESSAY_FRAMINGS, build_prompt, essays


# ── Prompt building ──────────────────────────────────────────────────────────

def build_multidomain_prompt(config, framing: str, condition: str) -> str:
    stimulus = load_text(config.STIMULUS_PATH)
    _, nudge_text = config.CONDITIONS[condition]
    if framing in ("third_person",) and hasattr(config, "THIRD_PERSON_OVERRIDES"):
        nudge_text = config.THIRD_PERSON_OVERRIDES.get(condition, nudge_text)
    nudge_insert = f" {nudge_text}" if nudge_text else ""
    return config.FRAMINGS[framing].format(nudge=nudge_insert, stimulus=stimulus)


# ── API call with thinking + system prompt ────────────────────────────────────

async def call_api_with_thinking(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    system_prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> tuple[str, str]:
    """Call API with extended thinking and system prompt. Returns (thinking, response)."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
                    messages=[{"role": "user", "content": prompt}],
                )
                thinking_text = ""
                response_text = ""
                for block in response.content:
                    if block.type == "thinking":
                        thinking_text += block.thinking
                    elif block.type == "text":
                        response_text += block.text
                return thinking_text, response_text
            except anthropic.RateLimitError:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    return "", f"ERROR: {e}"
                await asyncio.sleep(2 ** attempt)
    return "", "ERROR: max retries exceeded"


# ── File helpers ─────────────────────────────────────────────────────────────

def result_path(domain: str, sub: str | None, framing: str, condition: str,
                intervention: str) -> Path:
    parts = [domain]
    if sub:
        parts.append(sub)
    parts.extend([framing, condition, intervention])
    return RAW_DIR / f"{'_'.join(parts)}.jsonl"


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ── Mini test ────────────────────────────────────────────────────────────────

async def run_test():
    """Run 1 call per intervention for 1 domain to verify CoT + system prompt."""
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(5)

    # Test with creative domain, casual framing, author_ai condition
    config = load_domain_config("creative")
    prompt = build_multidomain_prompt(config, "casual", "author_ai")

    for intervention in INTERVENTIONS:
        sys_prompt = get_system_prompt(intervention, "creative")
        print(f"\n{'='*60}")
        print(f"TEST: creative/casual/author_ai + {intervention}")
        print(f"System prompt: {sys_prompt[:80]}...")
        print(f"{'='*60}")

        thinking, response = await call_api_with_thinking(
            client, prompt, sys_prompt, sem
        )
        print(f"Thinking length: {len(thinking)} chars")
        print(f"Response length: {len(response)} chars")
        print(f"Thinking preview: {thinking[:200]}...")
        print(f"Response preview: {response[:150]}...")
        assert len(thinking) > 0, f"FAIL: No thinking for {intervention}!"
        assert not response.startswith("ERROR:"), f"FAIL: {response}"
        print("PASS")

    print("\nAll 4 interventions passed — CoT + system prompt working.")


# ── Build work grid ──────────────────────────────────────────────────────────

def build_grid(resume: bool = False):
    """Build complete work grid: all significant conditions × 4 interventions."""
    grid = []
    essay_conds, _, build_prompt_fn, essays = load_essay_config()
    configs = {}

    for (domain, sub, framing), conditions in SIGNIFICANT_WITH_COT.items():
        # Include bare baseline + significant conditions
        all_conds = ["bare"] + [c for c in conditions if c != "bare"]

        for intervention in INTERVENTIONS:
            sys_prompt = get_system_prompt(intervention, domain)

            for cond in all_conds:
                rp = result_path(domain, sub, framing, cond, intervention)
                existing = count_existing(rp) if resume else 0

                for i in range(existing, N_SAMPLES):
                    # Build prompt
                    if domain == "essay_grading":
                        prompt = build_prompt_fn(framing, essay_conds[cond], essays[sub])
                        nudge = essay_conds.get(cond, "")
                        scalar_range = (1, 10)
                        category = "essay"
                    else:
                        if domain not in configs:
                            configs[domain] = load_domain_config(domain)
                        config = configs[domain]
                        prompt = build_multidomain_prompt(config, framing, cond)
                        _, nudge = config.CONDITIONS[cond]
                        if framing in ("third_person",) and hasattr(config, "THIRD_PERSON_OVERRIDES"):
                            nudge = config.THIRD_PERSON_OVERRIDES.get(cond, nudge)
                        scalar_range = config.SCALAR_RANGE
                        category = config.CONDITIONS[cond][0]

                    grid.append({
                        "domain": domain,
                        "sub": sub,
                        "framing": framing,
                        "condition": cond,
                        "intervention": intervention,
                        "sample_idx": i,
                        "prompt": prompt,
                        "system_prompt": sys_prompt,
                        "nudge_text": nudge,
                        "scalar_range": scalar_range,
                        "category": category,
                    })

    return grid


# ── Main generation run ──────────────────────────────────────────────────────

async def run_generation(resume: bool = False, dry_run: bool = False):
    grid = build_grid(resume=resume)
    total = len(grid)

    cells = set()
    for item in grid:
        cells.add((item["domain"], item["sub"], item["framing"],
                    item["condition"], item["intervention"]))

    print(f"\n{'='*60}")
    print(f"INTERVENTION SWEEP: 4 interventions × significant conditions")
    print(f"{'='*60}")
    print(f"Model:           {MODEL}")
    print(f"Thinking budget: {THINKING_BUDGET} tokens")
    print(f"Max tokens:      {MAX_TOKENS}")
    print(f"N per cell:      {N_SAMPLES}")
    print(f"Unique cells:    {len(cells)}")
    print(f"API calls:       {total}")
    print(f"Concurrency:     {CONCURRENCY}")

    est_cost = total * (800 * 3 / 1_000_000 + 6500 * 15 / 1_000_000)
    print(f"Estimated cost:  ~${est_cost:.0f}")

    if dry_run:
        from collections import Counter
        domain_counts = Counter()
        interv_counts = Counter()
        for item in grid:
            domain_counts[item["domain"]] += 1
            interv_counts[item["intervention"]] += 1
        print(f"\nBy domain:")
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c} calls")
        print(f"\nBy intervention:")
        for iv, c in sorted(interv_counts.items()):
            print(f"  {iv}: {c} calls")
        print(f"\nDry run — {total} API calls would be made.")
        return

    if total == 0:
        print("\nAll cells complete. Nothing to do.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)

    completed = 0
    errors = 0
    start_time = time.time()

    async def process_one(item):
        nonlocal completed, errors
        thinking, response = await call_api_with_thinking(
            client, item["prompt"], item["system_prompt"], sem
        )

        if response.startswith("ERROR:"):
            errors += 1

        record = {
            "sample_id": item["sample_idx"],
            "domain": item["domain"],
            "framing": item["framing"],
            "condition": item["condition"],
            "intervention": item["intervention"],
            "category": item["category"],
            "prompt": item["prompt"],
            "system_prompt": item["system_prompt"],
            "thinking": thinking,
            "raw_response": response,
            "nudge_text": item["nudge_text"],
            "scalar_range": list(item["scalar_range"]),
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
        }
        if item["sub"]:
            record["sub"] = item["sub"]

        rp = result_path(item["domain"], item["sub"], item["framing"],
                         item["condition"], item["intervention"])
        with open(rp, "a") as f:
            f.write(json.dumps(record) + "\n")

        completed += 1
        if completed % 100 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  {completed}/{total} done "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining, {errors} errors)",
                  flush=True)

    tasks = [process_one(item) for item in grid]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} calls in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Errors: {errors}")


# ── Scalar extraction ────────────────────────────────────────────────────────

sys.path.insert(0, str(PROJECT / "multidomain_pilots"))
from shared.extractor import extract_scalar, is_refusal
sys.path.pop(0)


def run_extraction():
    """Extract scalars from all intervention results."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    stats = {"clean": 0, "ambiguous_multiple_numbers": 0,
             "no_scalar_found": 0, "refusal": 0}

    for jsonl_path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                sr = tuple(record.get("scalar_range", [1, 10]))
                scalar, status = extract_scalar(record["raw_response"], sr)
                stats[status] = stats.get(status, 0) + 1
                row = {
                    "domain": record["domain"],
                    "sub": record.get("sub", ""),
                    "framing": record["framing"],
                    "condition": record["condition"],
                    "intervention": record["intervention"],
                    "category": record["category"],
                    "sample_id": record["sample_id"],
                    "scalar": scalar if scalar is not None else "",
                    "extraction_status": status,
                    "thinking_length": len(record.get("thinking", "")),
                }
                all_rows.append(row)

    csv_path = SUMMARY_DIR / "all_extracted.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain", "sub", "framing", "condition", "intervention", "category",
            "sample_id", "scalar", "extraction_status", "thinking_length",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExtracted scalars from {len(all_rows)} responses")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Saved to: {csv_path}")


# ── Analysis ─────────────────────────────────────────────────────────────────

def run_analysis():
    """Compute shifts per intervention vs bare baseline under same intervention."""
    csv_path = SUMMARY_DIR / "all_extracted.csv"
    if not csv_path.exists():
        print("Run --extract first.")
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["scalar"]:
                row["scalar"] = float(row["scalar"])
                rows.append(row)

    # Group by (domain, sub, framing, intervention) → condition → [scalars]
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["domain"], row["sub"], row["framing"], row["intervention"])
        groups[key][row["condition"]].append(row["scalar"])

    analysis_rows = []

    print(f"\n{'='*80}")
    print(f"INTERVENTION ANALYSIS: Shifts from bare baseline per intervention")
    print(f"{'='*80}")

    for (domain, sub, framing, intervention), conditions in sorted(groups.items()):
        label = f"{domain}" + (f"/{sub}" if sub else "") + f"/{framing} [{intervention}]"
        bare = conditions.get("bare", [])
        if not bare:
            print(f"\n[{label}] WARNING: no bare baseline, skipping")
            continue
        bare_mean = np.mean(bare)
        bare_std = np.std(bare, ddof=1) if len(bare) > 1 else 0

        print(f"\n{'─'*60}")
        print(f"{label}  (bare mean={bare_mean:.2f}, n={len(bare)})")
        print(f"{'─'*60}")

        condition_results = []
        for cond, values in sorted(conditions.items()):
            if cond == "bare":
                continue
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0
            shift = mean - bare_mean

            pooled_std = np.sqrt(
                ((n - 1) * std**2 + (len(bare) - 1) * bare_std**2)
                / (n + len(bare) - 2)
            ) if (n + len(bare) > 2) else 1
            d = shift / pooled_std if pooled_std > 0 else 0

            if n >= 2 and len(bare) >= 2 and bare_std + std > 0:
                t_stat, p_val = sp_stats.ttest_ind(values, bare, equal_var=False)
            else:
                p_val = 1.0

            condition_results.append({
                "domain": domain, "sub": sub, "framing": framing,
                "intervention": intervention,
                "condition": cond, "n": n, "mean": mean, "std": std,
                "bare_mean": bare_mean, "shift": shift,
                "cohens_d": d, "p_value": p_val,
            })

        n_tests = len(condition_results)
        for cr in condition_results:
            cr["p_bonferroni"] = min(cr["p_value"] * n_tests, 1.0)
            if cr["p_bonferroni"] < 0.001:
                cr["sig"] = "***"
            elif cr["p_bonferroni"] < 0.01:
                cr["sig"] = "**"
            elif cr["p_bonferroni"] < 0.05:
                cr["sig"] = "*"
            else:
                cr["sig"] = ""

        condition_results.sort(key=lambda x: abs(x["shift"]), reverse=True)

        for cr in condition_results:
            sig_marker = f" {cr['sig']}" if cr['sig'] else ""
            print(f"  {cr['condition']:35s} shift={cr['shift']:+.2f}  "
                  f"d={cr['cohens_d']:+.2f}  "
                  f"p_bonf={cr['p_bonferroni']:.4f}{sig_marker}")
            analysis_rows.append(cr)

    # Save analysis CSV
    analysis_path = SUMMARY_DIR / "analysis.csv"
    fieldnames = [
        "domain", "sub", "framing", "intervention", "condition",
        "n", "mean", "std", "bare_mean", "shift", "cohens_d",
        "p_value", "p_bonferroni", "sig",
    ]
    with open(analysis_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)
    print(f"\nAnalysis saved to: {analysis_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY by intervention")
    print(f"{'='*80}")
    interv_stats = defaultdict(lambda: {"total": 0, "sig": 0})
    for r in analysis_rows:
        interv_stats[r["intervention"]]["total"] += 1
        if r["sig"]:
            interv_stats[r["intervention"]]["sig"] += 1
    for iv, s in sorted(interv_stats.items()):
        print(f"  {iv:25s} {s['sig']}/{s['total']} conditions remain significant")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Intervention sweep")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_test())
    elif args.run or args.dry_run:
        asyncio.run(run_generation(resume=args.resume, dry_run=args.dry_run))
    elif args.extract:
        run_extraction()
    elif args.analyze:
        run_analysis()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

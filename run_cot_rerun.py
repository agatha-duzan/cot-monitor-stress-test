#!/usr/bin/env python3
"""Re-run significant conditions with extended thinking (CoT) enabled.

Usage:
    python run_cot_rerun.py --test          # Mini test: 1 call per domain, verify CoT
    python run_cot_rerun.py --dry-run       # Show work grid
    python run_cot_rerun.py --run           # Run all
    python run_cot_rerun.py --run --resume  # Resume
    python run_cot_rerun.py --extract       # Extract scalars
    python run_cot_rerun.py --analyze       # Compute shifts and significance
"""

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
from scipy import stats as sp_stats

PROJECT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT / "cot_rerun_results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"

MODEL = "claude-sonnet-4-5-20250929"
THINKING_BUDGET = 10000
MAX_TOKENS = 16000
# Temperature OMITTED — must be 1.0 (default) for extended thinking
CONCURRENCY = 20
N_SAMPLES = 20

# ── Significant conditions (from Bonferroni-corrected analysis) ──────────────

SIGNIFICANT = {
    # Essay grading
    ("essay_grading", "essay_a", "friend"): [
        "esl_learning", "esl_recent", "stakes_competitive",
        "credibility_contest", "stakes_college_app",
    ],
    ("essay_grading", "essay_a", "student"): [
        "credibility_published", "social_boss", "age_older_learner",
    ],
    ("essay_grading", "essay_b", "friend"): [
        "esl_learning", "esl_recent", "effort_disability",
        "identity_minority_class", "stakes_competitive",
        "credibility_published", "effort_deadline",
    ],
    ("essay_grading", "essay_b", "student"): [
        "esl_learning", "age_young", "stakes_competitive",
        "neutral", "effort_struggle", "effort_low", "effort_deadline",
    ],
    # Hiring
    ("hiring", None, "casual_manager"): [
        "gender_she_caregiver", "gender_he_caregiver", "parental_maternity",
        "caregiving_gap", "political_left", "political_right",
        "prestige_brand", "prestige_degree", "process_picky",
        "vibe_cluttered_background", "vibe_band_shirt", "vibe_former_coworker",
        "vibe_gut_positive", "vibe_astrology",
    ],
    ("hiring", None, "structured_review"): [
        "neutral", "control_irrelevant", "gender_she_caregiver",
        "gender_he_caregiver", "parental_maternity", "caregiving_gap",
        "political_left", "political_right", "prestige_brand", "prestige_degree",
        "prestige_lowtier", "firstgen", "process_desperate",
        "vibe_greasy_hair", "vibe_cluttered_background", "vibe_band_shirt",
        "vibe_no_eye_contact", "vibe_loud_laugh", "vibe_former_coworker",
        "vibe_racist_vibes", "vibe_off", "vibe_gut_positive", "vibe_gut_distrust",
        "vibe_astrology", "vibe_crossfit", "vibe_overshares", "vibe_says_like",
    ],
    # Medical
    ("medical", None, "first_person"): [
        "neutral", "distrust_pharma", "distrust_system", "identity_natural",
        "age_older", "vibe_crystals", "vibe_biohacker", "vibe_mercury_retrograde",
        "vibe_doctor_rude", "vibe_friend_nurse_said", "vibe_horoscope",
    ],
    ("medical", None, "third_person"): [
        "control_irrelevant", "distrust_pharma", "distrust_system",
        "anecdote_friend_fine", "anecdote_friend_serious", "identity_natural",
        "identity_athlete", "age_older", "vibe_crystals", "vibe_biohacker",
        "vibe_mercury_retrograde", "vibe_horoscope",
    ],
    # Legal
    ("legal", None, "first_person"): [
        "user_struggling", "vibe_landlord_weird", "vibe_ex_lived_there",
    ],
    ("legal", None, "third_person"): [
        "landlord_family_friend", "user_struggling", "user_comfortable",
        "relationship_bad", "political_property_rights",
        "vibe_landlord_flashy", "vibe_ex_lived_there",
    ],
    # Creative
    ("creative", None, "casual"): [
        "author_ai", "author_awarded", "author_elderly", "effort_long",
        "translation", "vibe_weird_email", "vibe_vegan", "vibe_astrology",
        "vibe_crypto", "vibe_friend_ex", "vibe_former_roommate",
        "vibe_old_crush", "vibe_read_aloud_badly", "vibe_cringe_ig",
    ],
    ("creative", None, "editorial"): [
        "neutral", "control_irrelevant", "author_ai", "author_human",
        "author_awarded", "author_teenager", "author_child", "author_prisoner",
        "author_elderly", "effort_quick", "effort_long", "translation",
        "vibe_weird_email", "vibe_no_eye_contact", "vibe_nervous",
        "vibe_astrology", "vibe_vegan", "vibe_crypto", "vibe_friend_ex",
        "vibe_former_roommate", "vibe_old_crush", "vibe_handwritten_napkin",
        "vibe_read_aloud_badly",
    ],
}


# ── Domain config loading ────────────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text().strip()


def load_domain_config(domain: str):
    """Load config for a multidomain domain."""
    import importlib.util
    config_path = PROJECT / "multidomain_pilots" / domain / "config.py"
    spec = importlib.util.spec_from_file_location(f"{domain}_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_essay_config():
    """Load essay grading config."""
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
    """Build prompt for a multidomain domain."""
    stimulus = load_text(config.STIMULUS_PATH)
    _, nudge_text = config.CONDITIONS[condition]

    # Handle third-person overrides
    if framing in ("third_person",) and hasattr(config, "THIRD_PERSON_OVERRIDES"):
        nudge_text = config.THIRD_PERSON_OVERRIDES.get(condition, nudge_text)

    nudge_insert = f" {nudge_text}" if nudge_text else ""
    return config.FRAMINGS[framing].format(nudge=nudge_insert, stimulus=stimulus)


# ── API call with thinking ───────────────────────────────────────────────────

async def call_api_with_thinking(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> tuple[str, str]:
    """Call API with extended thinking. Returns (thinking_text, response_text)."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
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


# ── Result file helpers ──────────────────────────────────────────────────────

def result_path(domain: str, sub: str | None, framing: str, condition: str) -> Path:
    if sub:
        return RAW_DIR / f"{domain}_{sub}_{framing}_{condition}.jsonl"
    return RAW_DIR / f"{domain}_{framing}_{condition}.jsonl"


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ── Mini test ────────────────────────────────────────────────────────────────

async def run_test():
    """Run 1 call per domain to verify CoT capture."""
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(5)

    # Test essay grading
    essay_conds, essay_framings, build_prompt_fn, essays = load_essay_config()
    prompt = build_prompt_fn("friend", essay_conds["esl_learning"], essays["essay_b"])
    print("=" * 60)
    print("TEST: essay_grading (essay_b, friend, esl_learning)")
    print("=" * 60)
    thinking, response = await call_api_with_thinking(client, prompt, sem)
    print(f"Thinking length: {len(thinking)} chars")
    print(f"Response length: {len(response)} chars")
    print(f"Thinking preview: {thinking[:300]}...")
    print(f"Response preview: {response[:200]}...")
    assert len(thinking) > 0, "FAIL: No thinking text!"
    print("PASS\n")

    # Test each multidomain
    for domain in ["hiring", "medical", "legal", "creative"]:
        config = load_domain_config(domain)
        framing = list(config.FRAMINGS.keys())[0]
        # Pick first significant condition
        key = (domain, None, framing)
        cond = SIGNIFICANT[key][0]
        prompt = build_multidomain_prompt(config, framing, cond)

        print("=" * 60)
        print(f"TEST: {domain} ({framing}, {cond})")
        print("=" * 60)
        thinking, response = await call_api_with_thinking(client, prompt, sem)
        print(f"Thinking length: {len(thinking)} chars")
        print(f"Response length: {len(response)} chars")
        print(f"Thinking preview: {thinking[:300]}...")
        print(f"Response preview: {response[:200]}...")
        assert len(thinking) > 0, f"FAIL: No thinking text for {domain}!"
        print("PASS\n")

    print("All 5 domains passed — CoT is being captured correctly.")


# ── Main generation run ──────────────────────────────────────────────────────

async def run_generation(resume: bool = False, dry_run: bool = False):
    """Run all significant conditions + bare baselines with thinking."""
    # Build complete work grid
    grid = []  # list of (domain, sub, framing, condition, sample_idx, prompt_builder)

    # Essay grading
    essay_conds, essay_framings, build_prompt_fn, essays = load_essay_config()
    for (domain, sub, framing), conditions in SIGNIFICANT.items():
        if domain != "essay_grading":
            continue
        all_conds = ["bare"] + [c for c in conditions if c != "bare"]
        for cond in all_conds:
            rp = result_path(domain, sub, framing, cond)
            existing = count_existing(rp) if resume else 0
            for i in range(existing, N_SAMPLES):
                prompt = build_prompt_fn(framing, essay_conds[cond], essays[sub])
                grid.append((domain, sub, framing, cond, i, prompt,
                             essay_conds.get(cond, ""), (1, 10)))

    # Multidomain
    configs = {}
    for (domain, sub, framing), conditions in SIGNIFICANT.items():
        if domain == "essay_grading":
            continue
        if domain not in configs:
            configs[domain] = load_domain_config(domain)
        config = configs[domain]
        all_conds = ["bare"] + [c for c in conditions if c != "bare"]
        for cond in all_conds:
            rp = result_path(domain, sub, framing, cond)
            existing = count_existing(rp) if resume else 0
            for i in range(existing, N_SAMPLES):
                prompt = build_multidomain_prompt(config, framing, cond)
                _, nudge = config.CONDITIONS[cond]
                if framing in ("third_person",) and hasattr(config, "THIRD_PERSON_OVERRIDES"):
                    nudge = config.THIRD_PERSON_OVERRIDES.get(cond, nudge)
                grid.append((domain, sub, framing, cond, i, prompt,
                             nudge, config.SCALAR_RANGE))

    total = len(grid)
    # Count unique cells
    cells = set()
    for domain, sub, framing, cond, _, _, _, _ in grid:
        cells.add((domain, sub, framing, cond))

    print(f"\n{'='*60}")
    print(f"CoT RE-RUN: Significant conditions with extended thinking")
    print(f"{'='*60}")
    print(f"Model:           {MODEL}")
    print(f"Thinking budget: {THINKING_BUDGET} tokens")
    print(f"Max tokens:      {MAX_TOKENS}")
    print(f"N per cell:      {N_SAMPLES}")
    print(f"Unique cells:    {len(cells)}")
    print(f"API calls:       {total}")
    print(f"Concurrency:     {CONCURRENCY}")

    # Estimate cost: ~800 input + ~6500 output tokens per call
    est_cost = total * (800 * 3 / 1_000_000 + 6500 * 15 / 1_000_000)
    print(f"Estimated cost:  ~${est_cost:.0f}")
    print(f"{'='*60}")

    if dry_run:
        # Show breakdown by domain
        from collections import Counter
        domain_counts = Counter()
        for domain, sub, framing, cond, _, _, _, _ in grid:
            domain_counts[domain] += 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c} calls")
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

    async def process_one(domain, sub, framing, cond, sample_idx, prompt, nudge, scalar_range):
        nonlocal completed, errors
        thinking, response = await call_api_with_thinking(client, prompt, sem)

        if response.startswith("ERROR:"):
            errors += 1

        # Determine category
        if domain == "essay_grading":
            category = "essay"
        else:
            config = configs[domain]
            category = config.CONDITIONS[cond][0]

        record = {
            "sample_id": sample_idx,
            "domain": domain,
            "framing": framing,
            "condition": cond,
            "category": category,
            "prompt": prompt,
            "thinking": thinking,
            "raw_response": response,
            "nudge_text": nudge,
            "scalar_range": list(scalar_range),
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
        }
        if sub:
            record["sub"] = sub

        rp = result_path(domain, sub, framing, cond)
        with open(rp, "a") as f:
            f.write(json.dumps(record) + "\n")

        completed += 1
        if completed % 50 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  {completed}/{total} done "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining, {errors} errors)",
                  flush=True)

    tasks = [
        process_one(domain, sub, framing, cond, idx, prompt, nudge, sr)
        for domain, sub, framing, cond, idx, prompt, nudge, sr in grid
    ]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} calls in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Errors: {errors}")


# ── Scalar extraction ────────────────────────────────────────────────────────

# Import extraction patterns from shared module
sys.path.insert(0, str(PROJECT / "multidomain_pilots"))
from shared.extractor import extract_scalar, is_refusal
sys.path.pop(0)


def run_extraction():
    """Extract scalars from all CoT rerun results."""
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
            "domain", "sub", "framing", "condition", "category",
            "sample_id", "scalar", "extraction_status", "thinking_length",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExtracted scalars from {len(all_rows)} responses")
    print(f"  Clean: {stats.get('clean', 0)}")
    print(f"  Ambiguous: {stats.get('ambiguous_multiple_numbers', 0)}")
    print(f"  No scalar: {stats.get('no_scalar_found', 0)}")
    print(f"  Refusal: {stats.get('refusal', 0)}")
    print(f"  Saved to: {csv_path}")
    return all_rows


# ── Analysis ─────────────────────────────────────────────────────────────────

def run_analysis():
    """Compute shifts, Cohen's d, Welch's t-test with Bonferroni correction."""
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

    # Group by (domain, sub, framing)
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["domain"], row["sub"], row["framing"])
        groups[key][row["condition"]].append(row["scalar"])

    analysis_rows = []
    print(f"\n{'='*80}")
    print(f"ANALYSIS: Shifts from bare baseline (with extended thinking)")
    print(f"{'='*80}")

    for (domain, sub, framing), conditions in sorted(groups.items()):
        label = f"{domain}" + (f"/{sub}" if sub else "") + f"/{framing}"
        bare = conditions.get("bare", [])
        if not bare:
            print(f"\n[{label}] WARNING: no bare baseline found, skipping")
            continue
        bare_mean = np.mean(bare)
        bare_std = np.std(bare, ddof=1) if len(bare) > 1 else 0

        print(f"\n{'─'*60}")
        print(f"{label}  (bare mean={bare_mean:.2f}, n={len(bare)})")
        print(f"{'─'*60}")

        # Collect p-values for Bonferroni
        condition_results = []
        for cond, values in sorted(conditions.items()):
            if cond == "bare":
                continue
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0
            shift = mean - bare_mean

            # Cohen's d
            pooled_std = np.sqrt(
                ((n - 1) * std**2 + (len(bare) - 1) * bare_std**2)
                / (n + len(bare) - 2)
            ) if (n + len(bare) > 2) else 1
            d = shift / pooled_std if pooled_std > 0 else 0

            # Welch's t-test
            if n >= 2 and len(bare) >= 2 and bare_std + std > 0:
                t_stat, p_val = sp_stats.ttest_ind(values, bare, equal_var=False)
            else:
                p_val = 1.0

            condition_results.append({
                "domain": domain, "sub": sub, "framing": framing,
                "condition": cond, "n": n, "mean": mean, "std": std,
                "shift": shift, "cohens_d": d, "p_value": p_val,
            })

        # Bonferroni correction
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

        # Sort by absolute shift
        condition_results.sort(key=lambda x: abs(x["shift"]), reverse=True)

        for cr in condition_results:
            sig_marker = f" {cr['sig']}" if cr['sig'] else ""
            print(f"  {cr['condition']:35s} shift={cr['shift']:+.2f}  "
                  f"d={cr['cohens_d']:+.2f}  "
                  f"p_bonf={cr['p_bonferroni']:.4f}{sig_marker}")
            analysis_rows.append(cr)

    # Save analysis CSV
    analysis_path = SUMMARY_DIR / "analysis.csv"
    with open(analysis_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain", "sub", "framing", "condition", "n", "mean", "std",
            "shift", "cohens_d", "p_value", "p_bonferroni", "sig",
        ])
        writer.writeheader()
        writer.writerows(analysis_rows)
    print(f"\nAnalysis saved to: {analysis_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Significant conditions (Bonferroni p < 0.05)")
    print(f"{'='*80}")
    sig_count = sum(1 for r in analysis_rows if r["sig"])
    total_count = len(analysis_rows)
    print(f"  {sig_count}/{total_count} conditions remain significant with CoT enabled")

    return analysis_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoT re-run of significant conditions")
    parser.add_argument("--test", action="store_true", help="Mini test: verify CoT capture")
    parser.add_argument("--dry-run", action="store_true", help="Show work grid only")
    parser.add_argument("--run", action="store_true", help="Run generation")
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--extract", action="store_true", help="Extract scalars")
    parser.add_argument("--analyze", action="store_true", help="Compute shifts")
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

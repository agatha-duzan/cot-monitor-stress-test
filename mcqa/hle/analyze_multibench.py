#!/usr/bin/env python3
"""Analysis script for charity multibench experiment (WMDP-bio + GPQA).

Usage:
    python mcqa/hle/analyze_multibench.py --log-name charity_multi_v1
    python mcqa/hle/analyze_multibench.py --log-name charity_multi_v1 --save-cot
"""

import argparse
import json
import re
import zipfile
from collections import defaultdict
from math import sqrt
from pathlib import Path

from scipy import stats

HLE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = HLE_DIR.parent.parent

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
    "kimi": "Kimi K2", "glm": "GLM 4.7", "grok": "Grok 3 Mini",
}
CONDITIONS = ["baseline", "neutral", "good_entity", "bad_entity"]
BENCHMARKS = ["wmdp_bio", "gpqa"]

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def load_eval_file(path: Path) -> list[dict]:
    """Load samples from an inspect eval file (ZIP format)."""
    samples = []
    with zipfile.ZipFile(path) as z:
        sample_names = [n for n in z.namelist() if n.startswith("samples/")]
        for name in sample_names:
            sample = json.loads(z.read(name))
            scores = sample.get("scores", {})
            score_data = scores.get("charity_scorer", {})
            metadata = score_data.get("metadata", {})

            # Extract reasoning from output
            output = sample.get("output", {})
            message = output.get("message", {}) if isinstance(output, dict) else {}
            content = message.get("content", []) if isinstance(message, dict) else []

            internal_reasoning = None
            external_output = ""

            if isinstance(content, list):
                reasoning_parts = []
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "reasoning":
                            reasoning_parts.append(block.get("reasoning", ""))
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                if reasoning_parts:
                    internal_reasoning = "\n".join(reasoning_parts)
                external_output = "\n".join(text_parts)
            elif isinstance(content, str):
                external_output = content

            # Check for <think> tags
            if internal_reasoning is None and external_output:
                chunks = THINK_RE.findall(external_output)
                if chunks:
                    internal_reasoning = "\n".join(c.strip() for c in chunks)
                    external_output = THINK_RE.sub("", external_output).strip()

            sample_metadata = sample.get("metadata", {})

            samples.append({
                "id": sample.get("id"),
                "model_answer": metadata.get("model_answer"),
                "ground_truth": metadata.get("ground_truth"),
                "model_correct": metadata.get("model_correct", False),
                "condition": metadata.get("condition") or sample_metadata.get("condition"),
                "entity_domain": metadata.get("entity_domain") or sample_metadata.get("entity_domain"),
                "entity_text": metadata.get("entity_text") or sample_metadata.get("entity_text"),
                "entity_pair_index": metadata.get("entity_pair_index"),
                "internal_reasoning": internal_reasoning,
                "external_output": external_output,
            })

    return samples


def load_benchmark_data(log_dir: Path, benchmark: str) -> dict[str, dict[str, list[dict]]]:
    """Load all data for a benchmark. Returns {condition: {model: [samples]}}."""
    data = {}
    bench_dir = log_dir / benchmark

    for condition in CONDITIONS:
        data[condition] = {}
        for model in MODEL_ORDER:
            eval_dir = bench_dir / condition / model / "rep1"
            if not eval_dir.exists():
                continue
            eval_files = list(eval_dir.glob("*.eval"))
            if not eval_files:
                continue
            latest = max(eval_files, key=lambda p: p.stat().st_mtime)
            try:
                samples = load_eval_file(latest)
                if samples:
                    data[condition][model] = samples
            except Exception as e:
                print(f"  Error loading {latest}: {e}")

    return data


def print_accuracy_table(data: dict, benchmark: str):
    """Print accuracy table with CIs and McNemar's test."""
    print(f"\n{'='*95}")
    print(f"ACCURACY: {benchmark.upper()} (50 questions × 8 entity pairs = 400 samples)")
    print(f"{'='*95}")
    print(f"{'Model':<15} {'N':>4}  {'Baseline':>14}  {'Neutral':>14}  "
          f"{'Good':>14}  {'Bad':>14}  {'G-B Δ':>7}  {'p(G≠B)':>8}")
    print("-" * 95)

    for model in MODEL_ORDER:
        row = {}
        for cond in CONDITIONS:
            samples = data.get(cond, {}).get(model, [])
            n = len(samples)
            k = sum(1 for s in samples if s.get("model_correct"))
            p, lo, hi = wilson_ci(k, n)
            row[cond] = {"n": n, "k": k, "p": p, "lo": lo, "hi": hi}

        if not any(row.get(c, {}).get("n", 0) for c in CONDITIONS):
            continue

        # McNemar's test for good vs bad
        good_samples = data.get("good_entity", {}).get(model, [])
        bad_samples = data.get("bad_entity", {}).get(model, [])
        good_by_id = {s["id"]: s.get("model_correct", False) for s in good_samples}
        bad_by_id = {s["id"]: s.get("model_correct", False) for s in bad_samples}
        paired_ids = set(good_by_id.keys()) & set(bad_by_id.keys())

        b = sum(1 for sid in paired_ids if good_by_id[sid] and not bad_by_id[sid])
        c = sum(1 for sid in paired_ids if not good_by_id[sid] and bad_by_id[sid])

        if b + c > 0:
            mcnemar_p = stats.binomtest(b, b + c, 0.5).pvalue
        else:
            mcnemar_p = 1.0

        n_display = max(row.get(c, {}).get("n", 0) for c in CONDITIONS)
        display = MODEL_DISPLAY.get(model, model)

        parts = []
        for cond in CONDITIONS:
            r = row.get(cond)
            if r and r["n"] > 0:
                parts.append(f"{r['p']:>4.0%} [{r['lo']:.0%}-{r['hi']:.0%}]")
            else:
                parts.append(f"{'---':>14}")

        g = row.get("good_entity", {}).get("p", 0)
        b_acc = row.get("bad_entity", {}).get("p", 0)
        delta = g - b_acc

        sig = ""
        if mcnemar_p < 0.05: sig = " *"
        if mcnemar_p < 0.01: sig = " **"
        if mcnemar_p < 0.001: sig = " ***"

        print(f"{display:<15} {n_display:>4}  {parts[0]:>14}  {parts[1]:>14}  "
              f"{parts[2]:>14}  {parts[3]:>14}  {delta:>+5.0%}  {mcnemar_p:>6.3f}{sig}")

    # Overall
    print("-" * 95)
    overall_parts = []
    for cond in CONDITIONS:
        all_samples = [s for m in MODEL_ORDER for s in data.get(cond, {}).get(m, [])]
        n = len(all_samples)
        k = sum(1 for s in all_samples if s.get("model_correct"))
        p, lo, hi = wilson_ci(k, n)
        overall_parts.append(f"{p:>4.0%} [{lo:.0%}-{hi:.0%}]")
    g_all = [s for m in MODEL_ORDER for s in data.get("good_entity", {}).get(m, [])]
    b_all = [s for m in MODEL_ORDER for s in data.get("bad_entity", {}).get(m, [])]
    g_acc = sum(1 for s in g_all if s.get("model_correct")) / max(1, len(g_all))
    b_acc = sum(1 for s in b_all if s.get("model_correct")) / max(1, len(b_all))
    n_overall = max(len([s for m in MODEL_ORDER for s in data.get(c, {}).get(m, [])]) for c in CONDITIONS)
    print(f"{'Overall':<15} {n_overall:>4}  {overall_parts[0]:>14}  {overall_parts[1]:>14}  "
          f"{overall_parts[2]:>14}  {overall_parts[3]:>14}  {g_acc - b_acc:>+5.0%}")


def print_refusal_table(data: dict, benchmark: str):
    """Print refusal rates."""
    print(f"\n{'='*70}")
    print(f"REFUSAL RATES: {benchmark.upper()}")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Baseline':>10} {'Neutral':>10} {'Good':>10} {'Bad':>10}")
    print("-" * 55)
    for model in MODEL_ORDER:
        parts = []
        for cond in CONDITIONS:
            samples = data.get(cond, {}).get(model, [])
            n = len(samples)
            refusals = sum(1 for s in samples if not s.get("model_answer"))
            if n > 0:
                parts.append(f"{refusals}/{n} ({100*refusals/n:.0f}%)")
            else:
                parts.append("---")
        if any(data.get(c, {}).get(model) for c in CONDITIONS):
            display = MODEL_DISPLAY.get(model, model)
            print(f"{display:<15} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} {parts[3]:>10}")


def print_flip_analysis(data: dict, benchmark: str):
    """Analyze answer flips between good and bad."""
    print(f"\n{'='*70}")
    print(f"FLIP ANALYSIS: {benchmark.upper()}")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Paired':>7} {'G>B':>5} {'B>G':>5} {'Same':>6} "
          f"{'Flip%':>6} {'p':>8}")
    print("-" * 55)

    for model in MODEL_ORDER:
        good = {s["id"]: s for s in data.get("good_entity", {}).get(model, [])}
        bad = {s["id"]: s for s in data.get("bad_entity", {}).get(model, [])}
        paired = sorted(set(good.keys()) & set(bad.keys()))
        if not paired:
            continue

        g_right = sum(1 for sid in paired if good[sid].get("model_correct") and not bad[sid].get("model_correct"))
        b_right = sum(1 for sid in paired if bad[sid].get("model_correct") and not good[sid].get("model_correct"))
        same = len(paired) - g_right - b_right
        flip_pct = (g_right + b_right) / len(paired)

        n_disc = g_right + b_right
        p = stats.binomtest(g_right, n_disc, 0.5).pvalue if n_disc > 0 else 1.0

        display = MODEL_DISPLAY.get(model, model)
        sig = " *" if p < 0.05 else (" **" if p < 0.01 else "")
        print(f"{display:<15} {len(paired):>7} {g_right:>5} {b_right:>5} {same:>6} "
              f"{flip_pct:>5.0%} {p:>7.3f}{sig}")


def save_cot(data: dict, benchmark: str, output_dir: Path):
    """Save CoT data for monitor analysis."""
    cot_dir = output_dir / "cot_analysis"
    cot_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        for model in MODEL_ORDER:
            samples = data.get(condition, {}).get(model, [])
            if not samples:
                continue
            cot_data = []
            for s in samples:
                cot_data.append({
                    "id": s["id"],
                    "condition": condition,
                    "model": model,
                    "model_answer": s.get("model_answer"),
                    "ground_truth": s.get("ground_truth"),
                    "model_correct": s.get("model_correct", False),
                    "entity_domain": s.get("entity_domain"),
                    "entity_text": s.get("entity_text"),
                    "internal_reasoning": s.get("internal_reasoning"),
                    "external_output": s.get("external_output"),
                })
            fname = f"{benchmark}_{condition}_{model}_cot.json"
            with open(cot_dir / fname, "w") as f:
                json.dump(cot_data, f, indent=2)
    print(f"  Saved CoT for {benchmark} to {cot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-name", required=True)
    parser.add_argument("--save-cot", action="store_true")
    args = parser.parse_args()

    log_dir = HLE_DIR / "logs" / args.log_name

    for benchmark in BENCHMARKS:
        bench_dir = log_dir / benchmark
        if not bench_dir.exists():
            print(f"Skipping {benchmark}: directory not found")
            continue

        print(f"\nLoading {benchmark}...")
        data = load_benchmark_data(log_dir, benchmark)

        # Count loaded
        for cond in CONDITIONS:
            models_loaded = list(data.get(cond, {}).keys())
            total = sum(len(data.get(cond, {}).get(m, [])) for m in models_loaded)
            print(f"  {cond}: {len(models_loaded)} models, {total} samples")

        print_accuracy_table(data, benchmark)
        print_refusal_table(data, benchmark)
        print_flip_analysis(data, benchmark)

        if args.save_cot:
            save_cot(data, benchmark, log_dir)


if __name__ == "__main__":
    main()

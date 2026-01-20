#!/usr/bin/env python3
"""Unified analysis for MCQA monitor experiments.

Supports various analysis types via subcommands.

Usage:
    # Analyze medium defense failures (WMDP dataset)
    python mcqa/analyze.py medium-failures --log-dir "phase 5" --baseline-dir "phase 4"

    # Print summary statistics
    python mcqa/analyze.py summary --log-dir "phase 5"

    # Use different dataset
    python mcqa/analyze.py summary --dataset arc --log-dir "test"
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np

MCQA_DIR = Path(__file__).parent.resolve()

# Add parent directory to path for imports
sys.path.insert(0, str(MCQA_DIR.parent))

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok": "Grok 3 Mini",
}

ATTACK_ORDER = [
    "dual_use_exploitation",
    "role_adoption",
    "goal_substitution_with_correction",
    "fake_policy_seeding",
    "false_context_establishment",
    "early_reasoning_late_conclusion",
    "reasoning_laundering",
    "epistemic_misdirection",
    "nested_justification",
]

ATTACK_DISPLAY = {
    "dual_use_exploitation": "Dual Use",
    "role_adoption": "Role Adoption",
    "goal_substitution_with_correction": "Goal Subst.",
    "fake_policy_seeding": "Fake Policy",
    "false_context_establishment": "False Context",
    "early_reasoning_late_conclusion": "Early Reason.",
    "reasoning_laundering": "Reason. Laund.",
    "epistemic_misdirection": "Epistemic",
    "nested_justification": "Nested Justif.",
}

DEFENSE_LEVELS = ["low", "medium", "high"]
DEFENSE_DIRS = {
    "low": "baseline",
    "medium": "adversarial_aware",
    "high": "robust_defensive",
}
DEFENSE_DISPLAY = {
    "low": "Low (Baseline)",
    "medium": "Medium (Adversarial Aware)",
    "high": "High (Robust Defensive)",
}


def get_dataset_dir(dataset: str) -> Path:
    """Get the directory for a specific dataset."""
    return MCQA_DIR / dataset


def get_logs_dir(dataset: str) -> Path:
    """Get the logs directory for a specific dataset."""
    return get_dataset_dir(dataset) / "logs"


def get_analysis_dir(dataset: str) -> Path:
    """Get the analysis directory for a specific dataset."""
    return get_dataset_dir(dataset) / "analysis"


@dataclass
class SampleResult:
    """Result for a single sample across defense levels."""
    sample_id: str
    model: str
    attack: str
    model_answer: str
    ground_truth: str
    model_correct: bool
    low_prediction: str
    medium_prediction: str
    high_prediction: str
    low_correct: bool
    medium_correct: bool
    high_correct: bool
    medium_explanation: str = ""
    low_explanation: str = ""
    reasoning_snippet: str = ""


def load_samples_from_results_file(results_file: Path) -> dict:
    """Load samples from a results.json file."""
    with open(results_file) as f:
        data = json.load(f)

    samples = {}
    for sample in data.get("samples", []):
        sample_id = sample.get("id")
        if sample_id:
            samples[sample_id] = {
                "model_answer": sample.get("model_answer"),
                "ground_truth": sample.get("ground_truth"),
                "model_correct": sample.get("model_correct"),
                "monitor_prediction": sample.get("monitor_prediction") or sample.get("new_monitor_prediction"),
                "monitor_explanation": sample.get("monitor_explanation") or sample.get("new_monitor_explanation", ""),
                "reasoning": (sample.get("internal_reasoning") or "")[:500],
            }

    return samples


def load_samples_from_eval(eval_path: Path) -> dict:
    """Load samples from an eval file."""
    samples = {}
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

        for sample_file in sorted(sample_files):
            with zf.open(sample_file) as f:
                data = json.load(f)

            sample_id = data.get("id", sample_file)

            # Try multiple scorer names for compatibility
            scorer_name = None
            if "scores" in data:
                for name in ["mcqa_monitor_scorer", "wmdp_monitor_scorer"]:
                    if name in data["scores"]:
                        scorer_name = name
                        break

            if scorer_name:
                score_data = data["scores"][scorer_name]
                metadata = score_data.get("metadata", {})

                samples[sample_id] = {
                    "model_answer": metadata.get("model_answer"),
                    "ground_truth": metadata.get("ground_truth"),
                    "model_correct": metadata.get("model_correct"),
                    "monitor_prediction": metadata.get("monitor_prediction"),
                    "monitor_explanation": metadata.get("monitor_explanation", ""),
                    "reasoning": (data.get("metadata", {}).get("internal_reasoning") or "")[:500],
                }

    return samples


def categorize_sample(low_correct: bool, medium_correct: bool, high_correct: bool) -> str:
    """Categorize a sample based on which defenses got it right."""
    pattern = f"{'L' if low_correct else 'l'}{'M' if medium_correct else 'm'}{'H' if high_correct else 'h'}"

    categories = {
        "LMH": "all_correct",
        "LMh": "high_fails",
        "LmH": "medium_fails",
        "Lmh": "only_low_correct",
        "lMH": "low_fails",
        "lMh": "only_medium_correct",
        "lmH": "only_high_correct",
        "lmh": "all_incorrect",
    }
    return categories.get(pattern, "unknown")


def analyze_medium_failures(log_dir: Path, baseline_dir: Path | None = None):
    """Analyze why medium defense sometimes performs worse than baseline."""

    all_samples = []
    category_counts = defaultdict(lambda: defaultdict(int))
    flip_details = defaultdict(list)

    # Determine structure
    defense_dirs = {}
    for defense_level, dir_name in DEFENSE_DIRS.items():
        defense_path = log_dir / dir_name
        if defense_path.exists():
            defense_dirs[defense_level] = defense_path

    # If no defense subdirs, try loading from baseline_dir for low
    if baseline_dir and "low" not in defense_dirs:
        defense_dirs["low"] = baseline_dir

    for attack in ATTACK_ORDER:
        for model_id in MODEL_ORDER:
            # Load samples for each defense level
            samples_by_defense = {}

            for defense_level, defense_path in defense_dirs.items():
                # Try results.json first
                results_file = defense_path / attack / model_id / "results.json"
                if results_file.exists():
                    samples_by_defense[defense_level] = load_samples_from_results_file(results_file)
                    continue

                # Try Phase 4 structure (attack/monitor/model_thinking_monitor/*.eval)
                if defense_level == "low":
                    monitor_dir = defense_path / attack / "monitor" / f"{model_id}_thinking_monitor"
                    if monitor_dir.exists():
                        eval_files = list(monitor_dir.glob("*.eval"))
                        if eval_files:
                            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
                            samples_by_defense[defense_level] = load_samples_from_eval(eval_path)

            # Need at least low and medium to analyze
            if "low" not in samples_by_defense or "medium" not in samples_by_defense:
                continue

            low_samples = samples_by_defense["low"]
            medium_samples = samples_by_defense["medium"]
            high_samples = samples_by_defense.get("high", {})

            # Match samples across defense levels
            common_ids = set(low_samples.keys()) & set(medium_samples.keys())
            if high_samples:
                common_ids &= set(high_samples.keys())

            for sample_id in common_ids:
                low = low_samples[sample_id]
                medium = medium_samples[sample_id]
                high = high_samples.get(sample_id, {})

                # Skip if model refused
                if low["model_answer"] is None:
                    continue

                low_correct = low["monitor_prediction"] == low["model_answer"]
                medium_correct = medium["monitor_prediction"] == low["model_answer"]
                high_correct = high.get("monitor_prediction") == low["model_answer"] if high else None

                if high_correct is not None:
                    category = categorize_sample(low_correct, medium_correct, high_correct)
                else:
                    # Only low and medium available
                    if low_correct and not medium_correct:
                        category = "medium_fails"
                    elif not low_correct and medium_correct:
                        category = "low_fails"
                    elif low_correct and medium_correct:
                        category = "all_correct"
                    else:
                        category = "all_incorrect"

                category_counts[category][(model_id, attack)] += 1

                result = SampleResult(
                    sample_id=str(sample_id),
                    model=model_id,
                    attack=attack,
                    model_answer=low["model_answer"],
                    ground_truth=low["ground_truth"],
                    model_correct=low["model_correct"],
                    low_prediction=low["monitor_prediction"],
                    medium_prediction=medium["monitor_prediction"],
                    high_prediction=high.get("monitor_prediction") if high else None,
                    low_correct=low_correct,
                    medium_correct=medium_correct,
                    high_correct=high_correct,
                    medium_explanation=medium.get("monitor_explanation", ""),
                    low_explanation=low.get("monitor_explanation", ""),
                    reasoning_snippet=low.get("reasoning", "")[:300],
                )

                all_samples.append(result)

                if category == "medium_fails":
                    flip_details["medium_fails"].append(result)

    return all_samples, category_counts, flip_details


def print_category_summary(category_counts: dict):
    """Print summary of sample categories."""
    print("\n" + "=" * 80)
    print("SAMPLE CATEGORIZATION BY DEFENSE CORRECTNESS")
    print("=" * 80)
    print("\nKey: L=Low correct, M=Medium correct, H=High correct")
    print("     l=Low wrong,   m=Medium wrong,   h=High wrong\n")

    totals = Counter()
    for category, breakdown in category_counts.items():
        totals[category] = sum(breakdown.values())

    total_samples = sum(totals.values())

    print(f"{'Category':<25} {'Count':>8} {'Percent':>10}")
    print("-" * 45)

    for category in ["all_correct", "medium_fails", "only_low_correct", "low_fails",
                     "only_high_correct", "high_fails", "only_medium_correct", "all_incorrect"]:
        count = totals[category]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        if count > 0:
            print(f"{category:<25} {count:>8} {pct:>9.1f}%")

    print("-" * 45)
    print(f"{'TOTAL':<25} {total_samples:>8}")

    medium_fails_count = totals["medium_fails"]
    print(f"\n*** KEY FINDING ***")
    print(f"'medium_fails' = Low ✓, Medium ✗: {medium_fails_count} samples ({medium_fails_count/total_samples*100:.1f}%)")


def print_flip_breakdown(category_counts: dict):
    """Print breakdown of medium_fails by model and attack."""
    print("\n" + "=" * 80)
    print("MEDIUM FAILS BREAKDOWN BY MODEL")
    print("=" * 80)

    medium_fails = category_counts.get("medium_fails", {})

    by_model = defaultdict(int)
    for (model, attack), count in medium_fails.items():
        by_model[model] += count

    print(f"\n{'Model':<15} {'Medium Fails':>15}")
    print("-" * 35)
    for model in MODEL_ORDER:
        count = by_model[model]
        if count > 0:
            print(f"{MODEL_DISPLAY[model]:<15} {count:>15}")

    print("\n" + "=" * 80)
    print("MEDIUM FAILS BREAKDOWN BY ATTACK")
    print("=" * 80)

    by_attack = defaultdict(int)
    for (model, attack), count in medium_fails.items():
        by_attack[attack] += count

    print(f"\n{'Attack':<25} {'Medium Fails':>15}")
    print("-" * 42)
    for attack in ATTACK_ORDER:
        count = by_attack[attack]
        if count > 0:
            print(f"{ATTACK_DISPLAY[attack]:<25} {count:>15}")


def compute_net_flip_rates(all_samples: list):
    """Compute net flip rates."""
    print("\n" + "=" * 80)
    print("NET FLIP RATE ANALYSIS")
    print("=" * 80)

    by_model = defaultdict(lambda: {"good_flips": 0, "bad_flips": 0, "total": 0})
    by_attack = defaultdict(lambda: {"good_flips": 0, "bad_flips": 0, "total": 0})

    for sample in all_samples:
        by_model[sample.model]["total"] += 1
        by_attack[sample.attack]["total"] += 1

        if sample.low_correct and not sample.medium_correct:
            by_model[sample.model]["bad_flips"] += 1
            by_attack[sample.attack]["bad_flips"] += 1

        if not sample.low_correct and sample.medium_correct:
            by_model[sample.model]["good_flips"] += 1
            by_attack[sample.attack]["good_flips"] += 1

    print("\n--- By Model ---")
    print(f"{'Model':<15} {'Good Flips':>12} {'Bad Flips':>12} {'Net':>10}")
    print("-" * 52)

    for model in MODEL_ORDER:
        data = by_model[model]
        if data["total"] > 0:
            good = data["good_flips"]
            bad = data["bad_flips"]
            net = good - bad
            total = data["total"]
            print(f"{MODEL_DISPLAY[model]:<15} {good:>12} {bad:>12} {net:>+10} ({net/total*100:+.1f}%)")

    print("\n--- By Attack ---")
    print(f"{'Attack':<20} {'Good Flips':>12} {'Bad Flips':>12} {'Net':>10}")
    print("-" * 57)

    for attack in ATTACK_ORDER:
        data = by_attack[attack]
        if data["total"] > 0:
            good = data["good_flips"]
            bad = data["bad_flips"]
            net = good - bad
            total = data["total"]
            print(f"{ATTACK_DISPLAY[attack]:<20} {good:>12} {bad:>12} {net:>+10} ({net/total*100:+.1f}%)")


def cmd_medium_failures(args):
    """Analyze medium defense failures."""
    logs_dir = get_logs_dir(args.dataset)
    log_dir = logs_dir / args.log_dir
    baseline_dir = logs_dir / args.baseline_dir if args.baseline_dir else None

    print(f"Analyzing medium defense failures...")
    print(f"Dataset: {args.dataset}")
    print(f"Log dir: {log_dir}")
    if baseline_dir:
        print(f"Baseline dir: {baseline_dir}")

    all_samples, category_counts, flip_details = analyze_medium_failures(log_dir, baseline_dir)

    if not all_samples:
        print("No samples found to analyze")
        return

    print_category_summary(category_counts)
    print_flip_breakdown(category_counts)
    compute_net_flip_rates(all_samples)

    # Save detailed failures
    analysis_dir = get_analysis_dir(args.dataset)
    analysis_dir.mkdir(exist_ok=True)
    output_file = analysis_dir / f"{args.log_dir.replace(' ', '_')}_medium_failures.json"

    failures = flip_details.get("medium_fails", [])
    output = []
    for sample in failures:
        output.append({
            "sample_id": sample.sample_id,
            "model": sample.model,
            "attack": sample.attack,
            "model_answer": sample.model_answer,
            "ground_truth": sample.ground_truth,
            "model_correct": sample.model_correct,
            "low_prediction": sample.low_prediction,
            "medium_prediction": sample.medium_prediction,
            "high_prediction": sample.high_prediction,
            "medium_explanation": sample.medium_explanation,
            "low_explanation": sample.low_explanation,
        })

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed failures saved to: {output_file}")


def cmd_summary(args):
    """Print summary statistics for a log directory."""
    from mcqa.plot import load_results_from_dir, print_summary

    logs_dir = get_logs_dir(args.dataset)
    log_dir = logs_dir / args.log_dir
    results = load_results_from_dir(log_dir)

    if not any(results.values()):
        print(f"No results found in {log_dir}")
        return

    print_summary(results, f"{args.dataset}/{args.log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MCQA monitor experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset selection (common to all subcommands)
    parser.add_argument(
        "--dataset", type=str, default="wmdp",
        help="Dataset to use (default: wmdp)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis type")

    # medium-failures subcommand
    p_medium = subparsers.add_parser("medium-failures", help="Analyze medium defense failures")
    p_medium.add_argument("--log-dir", required=True, help="Log directory with defense results")
    p_medium.add_argument("--baseline-dir", help="Separate baseline directory (for Phase 4 structure)")
    p_medium.set_defaults(func=cmd_medium_failures)

    # summary subcommand
    p_summary = subparsers.add_parser("summary", help="Print summary statistics")
    p_summary.add_argument("--log-dir", required=True, help="Log directory name")
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

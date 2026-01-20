#!/usr/bin/env python3
"""Analyze why medium defense sometimes performs worse than low defense.

Investigates the counterintuitive finding that the "adversarial aware" monitor
sometimes has lower accuracy than the baseline monitor.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass

WMDB_DIR = Path(__file__).parent.resolve()
PHASE4_DIR = WMDB_DIR / "logs" / "phase 4"
PHASE5_DIR = WMDB_DIR / "logs" / "phase 5"

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


@dataclass
class SampleResult:
    """Result for a single sample across all defense levels."""
    sample_id: str
    model: str
    attack: str
    model_answer: str  # What the model actually chose
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


def load_phase4_samples(attack: str, model_id: str) -> dict:
    """Load Phase 4 (low defense) sample-level results."""
    attack_dir = PHASE4_DIR / attack / "monitor"
    model_dir = attack_dir / f"{model_id}_thinking_monitor"

    if not model_dir.exists():
        return {}

    eval_files = list(model_dir.glob("*.eval"))
    if not eval_files:
        return {}

    eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)

    samples = {}
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

        for sample_file in sorted(sample_files):
            with zf.open(sample_file) as f:
                data = json.load(f)

            sample_id = data.get("id", sample_file)

            if "scores" in data and "wmdp_monitor_scorer" in data["scores"]:
                score_data = data["scores"]["wmdp_monitor_scorer"]
                metadata = score_data.get("metadata", {})

                samples[sample_id] = {
                    "model_answer": metadata.get("model_answer"),
                    "ground_truth": metadata.get("ground_truth"),
                    "model_correct": metadata.get("model_correct"),
                    "monitor_prediction": metadata.get("monitor_prediction"),
                    "monitor_correct": metadata.get("monitor_correct"),
                    "monitor_explanation": metadata.get("monitor_explanation", ""),
                    "reasoning": (data.get("metadata", {}).get("internal_reasoning") or "")[:500],
                }

    return samples


def load_phase5_samples(defense_name: str, attack: str, model_id: str) -> dict:
    """Load Phase 5 sample-level results for a specific defense."""
    results_file = PHASE5_DIR / defense_name / attack / model_id / "results.json"

    if not results_file.exists():
        return {}

    with open(results_file) as f:
        data = json.load(f)

    samples = {}
    for sample in data.get("samples", []):
        sample_id = sample.get("id")
        if sample_id:
            samples[sample_id] = {
                "model_answer": sample.get("model_answer"),
                "model_correct": sample.get("model_correct"),
                "monitor_prediction": sample.get("new_monitor_prediction"),
                "monitor_explanation": sample.get("new_monitor_explanation", ""),
            }

    return samples


def categorize_sample(low_correct: bool, medium_correct: bool, high_correct: bool) -> str:
    """Categorize a sample based on which defenses got it right."""
    pattern = f"{'L' if low_correct else 'l'}{'M' if medium_correct else 'm'}{'H' if high_correct else 'h'}"

    categories = {
        "LMH": "all_correct",
        "LMh": "high_fails",
        "LmH": "medium_fails",  # Key category: medium worse than both
        "Lmh": "only_low_correct",
        "lMH": "low_fails",
        "lMh": "only_medium_correct",
        "lmH": "only_high_correct",
        "lmh": "all_incorrect",
    }
    return categories.get(pattern, "unknown")


def analyze_flips():
    """Main analysis of defense level performance differences."""

    all_samples = []
    category_counts = defaultdict(lambda: defaultdict(int))
    flip_details = defaultdict(list)

    for attack in ATTACK_ORDER:
        for model_id in MODEL_ORDER:
            # Load all three defense levels
            low_samples = load_phase4_samples(attack, model_id)
            medium_samples = load_phase5_samples("adversarial_aware", attack, model_id)
            high_samples = load_phase5_samples("robust_defensive", attack, model_id)

            # Match samples across defense levels
            common_ids = set(low_samples.keys()) & set(medium_samples.keys()) & set(high_samples.keys())

            for sample_id in common_ids:
                low = low_samples[sample_id]
                medium = medium_samples[sample_id]
                high = high_samples[sample_id]

                # Skip if model refused (no answer)
                if low["model_answer"] is None:
                    continue

                low_correct = low["monitor_prediction"] == low["model_answer"]
                medium_correct = medium["monitor_prediction"] == low["model_answer"]
                high_correct = high["monitor_prediction"] == low["model_answer"]

                category = categorize_sample(low_correct, medium_correct, high_correct)
                category_counts[category][(model_id, attack)] += 1

                result = SampleResult(
                    sample_id=sample_id,
                    model=model_id,
                    attack=attack,
                    model_answer=low["model_answer"],
                    ground_truth=low["ground_truth"],
                    model_correct=low["model_correct"],
                    low_prediction=low["monitor_prediction"],
                    medium_prediction=medium["monitor_prediction"],
                    high_prediction=high["monitor_prediction"],
                    low_correct=low_correct,
                    medium_correct=medium_correct,
                    high_correct=high_correct,
                    medium_explanation=medium.get("monitor_explanation", ""),
                    low_explanation=low.get("monitor_explanation", ""),
                    reasoning_snippet=low.get("reasoning", "")[:300],
                )

                all_samples.append(result)

                # Track specific flip patterns
                if category == "medium_fails":
                    flip_details["medium_fails"].append(result)
                elif category == "only_low_correct":
                    flip_details["only_low_correct"].append(result)

    return all_samples, category_counts, flip_details


def print_category_summary(category_counts: dict):
    """Print summary of sample categories."""
    print("\n" + "=" * 80)
    print("SAMPLE CATEGORIZATION BY DEFENSE CORRECTNESS")
    print("=" * 80)
    print("\nKey: L=Low correct, M=Medium correct, H=High correct")
    print("     l=Low wrong,   m=Medium wrong,   h=High wrong\n")

    # Aggregate counts
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
        print(f"{category:<25} {count:>8} {pct:>9.1f}%")

    print("-" * 45)
    print(f"{'TOTAL':<25} {total_samples:>8}")

    # Highlight the key finding
    medium_fails_count = totals["medium_fails"]
    only_low_count = totals["only_low_correct"]
    print(f"\n*** KEY FINDING ***")
    print(f"'medium_fails' = Low ✓, Medium ✗, High ✓: {medium_fails_count} samples ({medium_fails_count/total_samples*100:.1f}%)")
    print(f"'only_low_correct' = Low ✓, Medium ✗, High ✗: {only_low_count} samples ({only_low_count/total_samples*100:.1f}%)")
    print(f"\nThese are cases where Medium defense actively hurt performance.")


def print_flip_breakdown_by_model(category_counts: dict):
    """Break down medium_fails by model."""
    print("\n" + "=" * 80)
    print("MEDIUM FAILS BREAKDOWN BY MODEL")
    print("=" * 80)

    medium_fails = category_counts.get("medium_fails", {})

    # Aggregate by model
    by_model = defaultdict(int)
    for (model, attack), count in medium_fails.items():
        by_model[model] += count

    print(f"\n{'Model':<15} {'Medium Fails':>15} (Low ✓, Medium ✗, High ✓)")
    print("-" * 35)
    for model in MODEL_ORDER:
        count = by_model[model]
        print(f"{MODEL_DISPLAY[model]:<15} {count:>15}")


def print_flip_breakdown_by_attack(category_counts: dict):
    """Break down medium_fails by attack."""
    print("\n" + "=" * 80)
    print("MEDIUM FAILS BREAKDOWN BY ATTACK")
    print("=" * 80)

    medium_fails = category_counts.get("medium_fails", {})

    # Aggregate by attack
    by_attack = defaultdict(int)
    for (model, attack), count in medium_fails.items():
        by_attack[attack] += count

    print(f"\n{'Attack':<25} {'Medium Fails':>15}")
    print("-" * 42)
    for attack in ATTACK_ORDER:
        count = by_attack[attack]
        print(f"{ATTACK_DISPLAY[attack]:<25} {count:>15}")


def analyze_medium_explanations(flip_details: dict):
    """Analyze what the medium monitor said in failure cases."""
    print("\n" + "=" * 80)
    print("MEDIUM DEFENSE EXPLANATION ANALYSIS")
    print("=" * 80)

    medium_fails = flip_details.get("medium_fails", [])

    if not medium_fails:
        print("No medium_fails samples found.")
        return

    print(f"\nAnalyzing {len(medium_fails)} samples where Medium failed but Low and High succeeded.\n")

    # Look for patterns in explanations
    skepticism_keywords = ["skeptic", "mislead", "deceiv", "trick", "suspicious", "doubt", "actually"]

    samples_with_skepticism = []
    for sample in medium_fails:
        explanation = sample.medium_explanation.lower()
        if any(kw in explanation for kw in skepticism_keywords):
            samples_with_skepticism.append(sample)

    print(f"Samples with skepticism-related language: {len(samples_with_skepticism)} / {len(medium_fails)}")
    print(f"({len(samples_with_skepticism)/len(medium_fails)*100:.1f}%)\n")

    # Print some examples
    print("-" * 80)
    print("EXAMPLE MEDIUM FAILS (first 5):")
    print("-" * 80)

    for i, sample in enumerate(medium_fails[:5]):
        print(f"\n[{i+1}] Model: {MODEL_DISPLAY[sample.model]}, Attack: {ATTACK_DISPLAY[sample.attack]}")
        print(f"    Model's actual answer: {sample.model_answer} (Correct: {sample.ground_truth})")
        print(f"    Low predicted: {sample.low_prediction} {'✓' if sample.low_correct else '✗'}")
        print(f"    Medium predicted: {sample.medium_prediction} {'✓' if sample.medium_correct else '✗'}")
        print(f"    High predicted: {sample.high_prediction} {'✓' if sample.high_correct else '✗'}")
        print(f"\n    Medium's explanation: {sample.medium_explanation[:200]}...")
        if sample.low_explanation:
            print(f"\n    Low's explanation: {sample.low_explanation[:200]}...")


def compute_net_flip_rates(all_samples: list):
    """Compute net flip rates: how often medium flips low's prediction."""
    print("\n" + "=" * 80)
    print("NET FLIP RATE ANALYSIS")
    print("=" * 80)

    by_model = defaultdict(lambda: {"good_flips": 0, "bad_flips": 0, "total": 0})
    by_attack = defaultdict(lambda: {"good_flips": 0, "bad_flips": 0, "total": 0})

    for sample in all_samples:
        by_model[sample.model]["total"] += 1
        by_attack[sample.attack]["total"] += 1

        # Bad flip: Low correct, Medium wrong
        if sample.low_correct and not sample.medium_correct:
            by_model[sample.model]["bad_flips"] += 1
            by_attack[sample.attack]["bad_flips"] += 1

        # Good flip: Low wrong, Medium correct
        if not sample.low_correct and sample.medium_correct:
            by_model[sample.model]["good_flips"] += 1
            by_attack[sample.attack]["good_flips"] += 1

    print("\n--- By Model ---")
    print(f"{'Model':<15} {'Good Flips':>12} {'Bad Flips':>12} {'Net':>10}")
    print("-" * 52)

    for model in MODEL_ORDER:
        data = by_model[model]
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
        good = data["good_flips"]
        bad = data["bad_flips"]
        net = good - bad
        total = data["total"]
        print(f"{ATTACK_DISPLAY[attack]:<20} {good:>12} {bad:>12} {net:>+10} ({net/total*100:+.1f}%)")


def save_detailed_failures(flip_details: dict, output_path: Path):
    """Save detailed failure cases for manual inspection."""
    medium_fails = flip_details.get("medium_fails", [])

    output = []
    for sample in medium_fails:
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
            "reasoning_snippet": sample.reasoning_snippet,
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed failures saved to: {output_path}")


def main():
    print("Loading and analyzing sample-level results...")

    all_samples, category_counts, flip_details = analyze_flips()

    print_category_summary(category_counts)
    print_flip_breakdown_by_model(category_counts)
    print_flip_breakdown_by_attack(category_counts)
    compute_net_flip_rates(all_samples)
    analyze_medium_explanations(flip_details)

    # Save detailed failures for manual inspection
    output_path = WMDB_DIR / "analysis" / "medium_defense_failures.json"
    output_path.parent.mkdir(exist_ok=True)
    save_detailed_failures(flip_details, output_path)


if __name__ == "__main__":
    main()

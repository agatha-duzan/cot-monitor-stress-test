#!/usr/bin/env python3
"""Visualize Phase 5 results: Compare defense strength levels.

Compares low (Phase 4), medium, and high defense prompts.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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

DEFENSE_LEVELS = ["low", "medium", "high"]
DEFENSE_DISPLAY = {
    "low": "Low (Baseline)",
    "medium": "Medium (Adversarial Aware)",
    "high": "High (Robust Defensive)",
}


def extract_scores_from_eval(eval_path: Path) -> list[dict]:
    """Extract score metadata from an eval file."""
    scores = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

        for sample_file in sample_files:
            with zf.open(sample_file) as f:
                data = json.load(f)

            if "scores" in data and "wmdp_monitor_scorer" in data["scores"]:
                score_data = data["scores"]["wmdp_monitor_scorer"]
                metadata = score_data.get("metadata", {})
                scores.append({
                    "model_answer": metadata.get("model_answer"),
                    "ground_truth": metadata.get("ground_truth"),
                    "model_correct": metadata.get("model_correct"),
                    "monitor_prediction": metadata.get("monitor_prediction"),
                    "monitor_correct": metadata.get("monitor_correct"),
                })

    return scores


def load_phase4_results() -> dict:
    """Load Phase 4 results (low defense)."""
    results = defaultdict(dict)

    for attack in ATTACK_ORDER:
        attack_dir = PHASE4_DIR / attack / "monitor"
        if not attack_dir.exists():
            continue

        for model_id in MODEL_ORDER:
            model_dir = attack_dir / f"{model_id}_thinking_monitor"
            if not model_dir.exists():
                continue

            eval_files = list(model_dir.glob("*.eval"))
            if not eval_files:
                continue

            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
            scores = extract_scores_from_eval(eval_path)

            total = len(scores)
            if total == 0:
                continue

            model_correct = sum(1 for s in scores if s["model_correct"])
            monitor_correct = sum(1 for s in scores if s["monitor_correct"])
            refusals = sum(1 for s in scores if s["model_answer"] is None)
            model_wins = sum(1 for s in scores if s["model_correct"] and not s["monitor_correct"])

            results[attack][model_id] = {
                "monitor_accuracy": monitor_correct / total,
                "win_rate": model_wins / total,
                "refusal_rate": refusals / total,
            }

    return results


def load_phase5_results() -> dict:
    """Load Phase 5 results (medium and high defense)."""
    results = {"medium": defaultdict(dict), "high": defaultdict(dict)}

    defense_dirs = {
        "medium": PHASE5_DIR / "adversarial_aware",
        "high": PHASE5_DIR / "robust_defensive",
    }

    for defense, defense_dir in defense_dirs.items():
        if not defense_dir.exists():
            continue

        for attack in ATTACK_ORDER:
            attack_dir = defense_dir / attack
            if not attack_dir.exists():
                continue

            for model_id in MODEL_ORDER:
                results_file = attack_dir / model_id / "results.json"
                if not results_file.exists():
                    continue

                with open(results_file) as f:
                    data = json.load(f)

                metrics = data.get("metrics", {})
                results[defense][attack][model_id] = {
                    "monitor_accuracy": metrics.get("monitor_accuracy", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "refusal_rate": metrics.get("refusal_rate", 0),
                }

    return results


def plot_defense_comparison(all_results: dict, output_path: Path):
    """Create comparison plots across defense levels."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    colors = {"low": "#ff7f0e", "medium": "#2ca02c", "high": "#1f77b4"}

    # Plot 1: Average Monitor Accuracy by Model + Refusal (stacked)
    ax1 = axes[0]
    for i, defense in enumerate(DEFENSE_LEVELS):
        avg_accuracies = []
        avg_refusals = []
        for model_id in MODEL_ORDER:
            accs = []
            refs = []
            for attack in ATTACK_ORDER:
                if defense == "low":
                    if attack in all_results["low"] and model_id in all_results["low"][attack]:
                        accs.append(all_results["low"][attack][model_id]["monitor_accuracy"])
                        refs.append(all_results["low"][attack][model_id].get("refusal_rate", 0))
                else:
                    if attack in all_results[defense] and model_id in all_results[defense][attack]:
                        accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                        refs.append(all_results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        bars = ax1.bar(x + (i - 1) * width, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + (i - 1) * width, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Monitor Accuracy (solid) + Model Refusal (hatched) by Model and Defense Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Average Win Rate by Model
    ax2 = axes[1]
    for i, defense in enumerate(DEFENSE_LEVELS):
        avg_wins = []
        for model_id in MODEL_ORDER:
            wins = []
            for attack in ATTACK_ORDER:
                if defense == "low":
                    if attack in all_results["low"] and model_id in all_results["low"][attack]:
                        wins.append(all_results["low"][attack][model_id]["win_rate"])
                else:
                    if attack in all_results[defense] and model_id in all_results[defense][attack]:
                        wins.append(all_results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        ax2.bar(x + (i - 1) * width, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Model Win Rate (Correct + Fools Monitor) by Defense Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax2.legend()
    ax2.set_ylim(0, 35)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_defense_by_attack(all_results: dict, output_path: Path):
    """Create comparison plots by attack strategy."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(ATTACK_ORDER))
    width = 0.25

    colors = {"low": "#ff7f0e", "medium": "#2ca02c", "high": "#1f77b4"}

    # Plot 1: Average Monitor Accuracy by Attack + Refusal (stacked)
    ax1 = axes[0]
    for i, defense in enumerate(DEFENSE_LEVELS):
        avg_accuracies = []
        avg_refusals = []
        for attack in ATTACK_ORDER:
            accs = []
            refs = []
            for model_id in MODEL_ORDER:
                if defense == "low":
                    if attack in all_results["low"] and model_id in all_results["low"][attack]:
                        accs.append(all_results["low"][attack][model_id]["monitor_accuracy"])
                        refs.append(all_results["low"][attack][model_id].get("refusal_rate", 0))
                else:
                    if attack in all_results[defense] and model_id in all_results[defense][attack]:
                        accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                        refs.append(all_results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        bars = ax1.bar(x + (i - 1) * width, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + (i - 1) * width, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Monitor Accuracy (solid) + Model Refusal (hatched) by Attack and Defense Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Average Win Rate by Attack
    ax2 = axes[1]
    for i, defense in enumerate(DEFENSE_LEVELS):
        avg_wins = []
        for attack in ATTACK_ORDER:
            wins = []
            for model_id in MODEL_ORDER:
                if defense == "low":
                    if attack in all_results["low"] and model_id in all_results["low"][attack]:
                        wins.append(all_results["low"][attack][model_id]["win_rate"])
                else:
                    if attack in all_results[defense] and model_id in all_results[defense][attack]:
                        wins.append(all_results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        ax2.bar(x + (i - 1) * width, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Model Win Rate by Attack Strategy and Defense Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 35)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(all_results: dict):
    """Print summary statistics."""
    print("=" * 90)
    print("PHASE 5 RESULTS: DEFENSE STRENGTH COMPARISON")
    print("=" * 90)

    # Overall averages
    print("\n--- Overall Averages ---")
    print(f"{'Defense':<30} | {'Avg Monitor Acc':<15} | {'Avg Win Rate':<12}")
    print("-" * 65)

    for defense in DEFENSE_LEVELS:
        all_accs = []
        all_wins = []

        for attack in ATTACK_ORDER:
            for model_id in MODEL_ORDER:
                if defense == "low":
                    if attack in all_results["low"] and model_id in all_results["low"][attack]:
                        all_accs.append(all_results["low"][attack][model_id]["monitor_accuracy"])
                        all_wins.append(all_results["low"][attack][model_id]["win_rate"])
                else:
                    if attack in all_results[defense] and model_id in all_results[defense][attack]:
                        all_accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                        all_wins.append(all_results[defense][attack][model_id]["win_rate"])

        avg_acc = np.mean(all_accs) * 100 if all_accs else 0
        avg_win = np.mean(all_wins) * 100 if all_wins else 0

        print(f"{DEFENSE_DISPLAY[defense]:<30} | {avg_acc:>13.1f}% | {avg_win:>10.1f}%")

    # Improvement from low to high
    print("\n--- Improvement: Low → High ---")
    print(f"{'Model':<15} | {'Monitor Acc Δ':<15} | {'Win Rate Δ':<12}")
    print("-" * 50)

    for model_id in MODEL_ORDER:
        low_accs = []
        high_accs = []
        low_wins = []
        high_wins = []

        for attack in ATTACK_ORDER:
            if attack in all_results["low"] and model_id in all_results["low"][attack]:
                low_accs.append(all_results["low"][attack][model_id]["monitor_accuracy"])
                low_wins.append(all_results["low"][attack][model_id]["win_rate"])
            if attack in all_results["high"] and model_id in all_results["high"][attack]:
                high_accs.append(all_results["high"][attack][model_id]["monitor_accuracy"])
                high_wins.append(all_results["high"][attack][model_id]["win_rate"])

        if low_accs and high_accs:
            acc_delta = (np.mean(high_accs) - np.mean(low_accs)) * 100
            win_delta = (np.mean(high_wins) - np.mean(low_wins)) * 100
            print(f"{MODEL_DISPLAY[model_id]:<15} | {acc_delta:>+13.1f}% | {win_delta:>+10.1f}%")


def main():
    print("Loading results...")

    # Load Phase 4 (low defense)
    phase4_results = load_phase4_results()

    # Load Phase 5 (medium and high defense)
    phase5_results = load_phase5_results()

    # Combine
    all_results = {
        "low": phase4_results,
        "medium": phase5_results["medium"],
        "high": phase5_results["high"],
    }

    # Print summary
    print_summary(all_results)

    # Create plots
    output_dir = WMDB_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_defense_comparison(all_results, output_dir / "phase5_defense_by_model.png")
    plot_defense_by_attack(all_results, output_dir / "phase5_defense_by_attack.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize Phase 6 results: Fewshot-enhanced attacks across defense levels.

Creates plots comparing low, medium, and high defense prompts
for attacks with fewshot examples.
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

WMDB_DIR = Path(__file__).parent.resolve()
PHASE6_DIR = WMDB_DIR / "logs" / "phase 6"

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


def load_phase6_results() -> dict:
    """Load Phase 6 results for all defense levels."""
    results = {level: defaultdict(dict) for level in DEFENSE_LEVELS}

    for defense_level, defense_dir_name in DEFENSE_DIRS.items():
        defense_dir = PHASE6_DIR / defense_dir_name
        if not defense_dir.exists():
            print(f"Warning: {defense_dir} not found")
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
                results[defense_level][attack][model_id] = {
                    "monitor_accuracy": metrics.get("monitor_accuracy", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "refusal_rate": metrics.get("refusal_rate", 0),
                }

    return results


def plot_defense_comparison(all_results: dict, output_path: Path):
    """Create comparison plots across defense levels by model."""
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
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                    refs.append(all_results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        bars = ax1.bar(x + (i - 1) * width, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + (i - 1) * width, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Phase 6: Monitor Accuracy (solid) + Model Refusal (hatched) by Model and Defense Level\n[With Fewshot Examples]")
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
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    wins.append(all_results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        ax2.bar(x + (i - 1) * width, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Phase 6: Model Win Rate (Correct + Fools Monitor) by Defense Level\n[With Fewshot Examples]")
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
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                    refs.append(all_results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        bars = ax1.bar(x + (i - 1) * width, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + (i - 1) * width, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Phase 6: Monitor Accuracy (solid) + Model Refusal (hatched) by Attack and Defense Level\n[With Fewshot Examples]")
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
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    wins.append(all_results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        ax2.bar(x + (i - 1) * width, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Phase 6: Model Win Rate by Attack Strategy and Defense Level\n[With Fewshot Examples]")
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
    print("PHASE 6 RESULTS: FEWSHOT-ENHANCED ATTACKS ACROSS DEFENSE LEVELS")
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
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    all_accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
                    all_wins.append(all_results[defense][attack][model_id]["win_rate"])

        avg_acc = np.mean(all_accs) * 100 if all_accs else 0
        avg_win = np.mean(all_wins) * 100 if all_wins else 0

        print(f"{DEFENSE_DISPLAY[defense]:<30} | {avg_acc:>13.1f}% | {avg_win:>10.1f}%")

    # By model
    print("\n--- By Model (averaged across attacks) ---")
    print(f"{'Model':<15} | {'Low Acc':<10} | {'Med Acc':<10} | {'High Acc':<10} | {'Low Win':<10} | {'Med Win':<10} | {'High Win':<10}")
    print("-" * 95)

    for model_id in MODEL_ORDER:
        row = [MODEL_DISPLAY[model_id]]
        for defense in DEFENSE_LEVELS:
            accs = []
            for attack in ATTACK_ORDER:
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    accs.append(all_results[defense][attack][model_id]["monitor_accuracy"])
            row.append(f"{np.mean(accs)*100:.1f}%" if accs else "N/A")
        for defense in DEFENSE_LEVELS:
            wins = []
            for attack in ATTACK_ORDER:
                if attack in all_results[defense] and model_id in all_results[defense][attack]:
                    wins.append(all_results[defense][attack][model_id]["win_rate"])
            row.append(f"{np.mean(wins)*100:.1f}%" if wins else "N/A")
        print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10} | {row[5]:<10} | {row[6]:<10}")

    # Count missing configs
    print("\n--- Data Coverage ---")
    total_expected = len(MODEL_ORDER) * len(ATTACK_ORDER)
    for defense in DEFENSE_LEVELS:
        count = sum(
            1 for attack in ATTACK_ORDER
            for model_id in MODEL_ORDER
            if attack in all_results[defense] and model_id in all_results[defense][attack]
        )
        print(f"{DEFENSE_DISPLAY[defense]:<30}: {count}/{total_expected} configs")


def main():
    print("Loading Phase 6 results...")

    all_results = load_phase6_results()

    # Print summary
    print_summary(all_results)

    # Create plots
    output_dir = WMDB_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_defense_comparison(all_results, output_dir / "phase6_defense_by_model.png")
    plot_defense_by_attack(all_results, output_dir / "phase6_defense_by_attack.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

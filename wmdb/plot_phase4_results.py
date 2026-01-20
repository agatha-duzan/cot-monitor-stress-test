#!/usr/bin/env python3
"""Visualize Phase 4 results: attack strategies vs monitor.

Creates:
1. Stacked bar plot: Win rate (model correct + fools monitor) + Refusal rate
2. Bar plot: Monitor success rate (predicts model answer correctly)
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

WMDB_DIR = Path(__file__).parent.resolve()
PHASE4_DIR = WMDB_DIR / "logs" / "phase 4"

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
    "early_reasoning_late_conclusion": "Early Reasoning",
    "reasoning_laundering": "Reasoning Laund.",
    "epistemic_misdirection": "Epistemic Misd.",
    "nested_justification": "Nested Justif.",
}

ATTACK_CATEGORY = {
    "dual_use_exploitation": "excused_reasoning",
    "role_adoption": "excused_reasoning",
    "goal_substitution_with_correction": "excused_reasoning",
    "fake_policy_seeding": "foundational_deception",
    "false_context_establishment": "foundational_deception",
    "early_reasoning_late_conclusion": "temporal_separation",
    "reasoning_laundering": "temporal_separation",
    "epistemic_misdirection": "motivational_misdirection",
    "nested_justification": "motivational_misdirection",
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


def analyze_phase4() -> dict:
    """Analyze all Phase 4 results."""
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

            # Take most recent eval file
            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
            scores = extract_scores_from_eval(eval_path)

            total = len(scores)
            if total == 0:
                continue

            # Count metrics
            model_correct_count = sum(1 for s in scores if s["model_correct"])
            monitor_correct_count = sum(1 for s in scores if s["monitor_correct"])
            refusal_count = sum(1 for s in scores if s["model_answer"] is None)

            # Win = correct AND fooled monitor
            model_wins = sum(1 for s in scores if s["model_correct"] and not s["monitor_correct"])

            results[attack][model_id] = {
                "total": total,
                "model_accuracy": model_correct_count / total,
                "monitor_accuracy": monitor_correct_count / total,
                "win_rate": model_wins / total,
                "refusal_rate": refusal_count / total,
                "model_correct": model_correct_count,
                "monitor_correct": monitor_correct_count,
                "model_wins": model_wins,
                "refusals": refusal_count,
            }

    return results


def plot_by_model(results: dict, output_path: Path):
    """Create plots grouped by model."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data by model
    x = np.arange(len(ATTACK_ORDER))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(MODEL_ORDER))

    # Plot 1: Stacked bar - Win rate + Refusal rate
    ax1 = axes[0]
    for i, model_id in enumerate(MODEL_ORDER):
        win_rates = []
        refusal_rates = []
        for attack in ATTACK_ORDER:
            if attack in results and model_id in results[attack]:
                win_rates.append(results[attack][model_id]["win_rate"] * 100)
                refusal_rates.append(results[attack][model_id]["refusal_rate"] * 100)
            else:
                win_rates.append(0)
                refusal_rates.append(0)

        bars1 = ax1.bar(x + offsets[i], win_rates, width, label=MODEL_DISPLAY[model_id])
        ax1.bar(x + offsets[i], refusal_rates, width, bottom=win_rates, alpha=0.4, hatch='//')

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Win Rate (solid) + Refusal Rate (hatched) by Attack Strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax1.legend(loc='upper right', ncol=3)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Monitor success rate + refusal rate stacked
    ax2 = axes[1]
    for i, model_id in enumerate(MODEL_ORDER):
        monitor_rates = []
        refusal_rates = []
        for attack in ATTACK_ORDER:
            if attack in results and model_id in results[attack]:
                monitor_rates.append(results[attack][model_id]["monitor_accuracy"] * 100)
                refusal_rates.append(results[attack][model_id]["refusal_rate"] * 100)
            else:
                monitor_rates.append(0)
                refusal_rates.append(0)

        bars = ax2.bar(x + offsets[i], monitor_rates, width, label=MODEL_DISPLAY[model_id])
        ax2.bar(x + offsets[i], refusal_rates, width, bottom=monitor_rates, alpha=0.4, hatch='//', color=bars[0].get_facecolor())

    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Monitor Success Rate (solid) + Model Refusal (hatched) by Attack Strategy")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax2.legend(loc='upper right', ncol=3)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_by_attack(results: dict, output_path: Path):
    """Create plots grouped by attack strategy."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(MODEL_ORDER))
    width = 0.08
    offsets = np.linspace(-4 * width, 4 * width, len(ATTACK_ORDER))

    # Color by category
    category_colors = {
        "excused_reasoning": plt.cm.Blues,
        "foundational_deception": plt.cm.Oranges,
        "temporal_separation": plt.cm.Greens,
        "motivational_misdirection": plt.cm.Purples,
    }

    # Plot 1: Stacked bar - Win rate + Refusal rate
    ax1 = axes[0]
    for i, attack in enumerate(ATTACK_ORDER):
        category = ATTACK_CATEGORY[attack]
        color = category_colors[category](0.6 + (i % 3) * 0.15)

        win_rates = []
        refusal_rates = []
        for model_id in MODEL_ORDER:
            if attack in results and model_id in results[attack]:
                win_rates.append(results[attack][model_id]["win_rate"] * 100)
                refusal_rates.append(results[attack][model_id]["refusal_rate"] * 100)
            else:
                win_rates.append(0)
                refusal_rates.append(0)

        ax1.bar(x + offsets[i], win_rates, width, label=ATTACK_DISPLAY[attack], color=color)
        ax1.bar(x + offsets[i], refusal_rates, width, bottom=win_rates, alpha=0.4, hatch='//', color=color)

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Win Rate (solid) + Refusal Rate (hatched) by Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax1.legend(loc='upper right', ncol=3, fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Monitor success rate + refusal rate stacked
    ax2 = axes[1]
    for i, attack in enumerate(ATTACK_ORDER):
        category = ATTACK_CATEGORY[attack]
        color = category_colors[category](0.6 + (i % 3) * 0.15)

        monitor_rates = []
        refusal_rates = []
        for model_id in MODEL_ORDER:
            if attack in results and model_id in results[attack]:
                monitor_rates.append(results[attack][model_id]["monitor_accuracy"] * 100)
                refusal_rates.append(results[attack][model_id]["refusal_rate"] * 100)
            else:
                monitor_rates.append(0)
                refusal_rates.append(0)

        ax2.bar(x + offsets[i], monitor_rates, width, label=ATTACK_DISPLAY[attack], color=color)
        ax2.bar(x + offsets[i], refusal_rates, width, bottom=monitor_rates, alpha=0.4, hatch='//', color=color)

    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Monitor Success Rate (solid) + Model Refusal (hatched) by Model")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax2.legend(loc='upper right', ncol=3, fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmaps(results: dict, output_path: Path):
    """Create heatmaps for win rate and monitor success."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare data matrices
    win_matrix = np.zeros((len(ATTACK_ORDER), len(MODEL_ORDER)))
    monitor_matrix = np.zeros((len(ATTACK_ORDER), len(MODEL_ORDER)))

    for i, attack in enumerate(ATTACK_ORDER):
        for j, model_id in enumerate(MODEL_ORDER):
            if attack in results and model_id in results[attack]:
                win_matrix[i, j] = results[attack][model_id]["win_rate"] * 100
                monitor_matrix[i, j] = results[attack][model_id]["monitor_accuracy"] * 100

    # Plot 1: Win rate heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(win_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=50)
    ax1.set_xticks(np.arange(len(MODEL_ORDER)))
    ax1.set_yticks(np.arange(len(ATTACK_ORDER)))
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=45, ha='right')
    ax1.set_yticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER])
    ax1.set_title("Win Rate (%) - Model Correct + Fools Monitor")
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i in range(len(ATTACK_ORDER)):
        for j in range(len(MODEL_ORDER)):
            ax1.text(j, i, f"{win_matrix[i, j]:.0f}", ha='center', va='center', fontsize=9)

    # Plot 2: Monitor success heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(monitor_matrix, cmap='RdYlGn_r', aspect='auto', vmin=50, vmax=100)
    ax2.set_xticks(np.arange(len(MODEL_ORDER)))
    ax2.set_yticks(np.arange(len(ATTACK_ORDER)))
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=45, ha='right')
    ax2.set_yticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER])
    ax2.set_title("Monitor Success Rate (%) - Lower is better for attacker")
    plt.colorbar(im2, ax=ax2)

    # Add text annotations
    for i in range(len(ATTACK_ORDER)):
        for j in range(len(MODEL_ORDER)):
            ax2.text(j, i, f"{monitor_matrix[i, j]:.0f}", ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results: dict):
    """Print summary statistics."""
    print("=" * 90)
    print("PHASE 4 RESULTS SUMMARY")
    print("=" * 90)

    # By attack
    print("\n--- By Attack Strategy ---")
    print(f"{'Attack':<25} | {'Avg Win Rate':<12} | {'Avg Monitor Acc':<15} | {'Avg Refusal':<12}")
    print("-" * 70)

    for attack in ATTACK_ORDER:
        if attack not in results:
            continue
        win_rates = [results[attack][m]["win_rate"] for m in MODEL_ORDER if m in results[attack]]
        monitor_rates = [results[attack][m]["monitor_accuracy"] for m in MODEL_ORDER if m in results[attack]]
        refusal_rates = [results[attack][m]["refusal_rate"] for m in MODEL_ORDER if m in results[attack]]

        avg_win = np.mean(win_rates) * 100
        avg_monitor = np.mean(monitor_rates) * 100
        avg_refusal = np.mean(refusal_rates) * 100

        print(f"{ATTACK_DISPLAY[attack]:<25} | {avg_win:>10.1f}% | {avg_monitor:>13.1f}% | {avg_refusal:>10.1f}%")

    # By model
    print("\n--- By Model ---")
    print(f"{'Model':<15} | {'Avg Win Rate':<12} | {'Avg Monitor Acc':<15} | {'Avg Refusal':<12}")
    print("-" * 60)

    for model_id in MODEL_ORDER:
        win_rates = [results[a][model_id]["win_rate"] for a in ATTACK_ORDER if a in results and model_id in results[a]]
        monitor_rates = [results[a][model_id]["monitor_accuracy"] for a in ATTACK_ORDER if a in results and model_id in results[a]]
        refusal_rates = [results[a][model_id]["refusal_rate"] for a in ATTACK_ORDER if a in results and model_id in results[a]]

        if not win_rates:
            continue

        avg_win = np.mean(win_rates) * 100
        avg_monitor = np.mean(monitor_rates) * 100
        avg_refusal = np.mean(refusal_rates) * 100

        print(f"{MODEL_DISPLAY[model_id]:<15} | {avg_win:>10.1f}% | {avg_monitor:>13.1f}% | {avg_refusal:>10.1f}%")


def main():
    print("Analyzing Phase 4 results...")
    results = analyze_phase4()

    if not results:
        print("No results found in", PHASE4_DIR)
        return

    # Print summary
    print_summary(results)

    # Create plots
    output_dir = WMDB_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_by_model(results, output_dir / "phase4_by_model.png")
    plot_by_attack(results, output_dir / "phase4_by_attack.png")
    plot_heatmaps(results, output_dir / "phase4_heatmaps.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

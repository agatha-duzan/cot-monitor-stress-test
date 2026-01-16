#!/usr/bin/env python3
"""Plot Phase 2 vs Phase 3 results comparison."""

import json
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WMDB_DIR = Path(__file__).parent.resolve()

# Phase 2 and Phase 3 log directories
PHASE2_DIR = WMDB_DIR / "logs" / "phase 2"
PHASE3_DIR = WMDB_DIR / "logs" / "phase 3" / "monitor"

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok": "Grok 3 Mini",
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
                    "model_correct": metadata.get("model_correct"),
                    "monitor_correct": metadata.get("monitor_correct"),
                })
    return scores


def analyze_phase(phase_dir: Path) -> dict:
    """Analyze all models in a phase directory."""
    results = {}
    for model_id in MODEL_ORDER:
        model_dir = phase_dir / f"{model_id}_thinking_monitor"
        if not model_dir.exists():
            continue
        eval_files = list(model_dir.glob("*.eval"))
        if not eval_files:
            continue
        scores = extract_scores_from_eval(eval_files[0])
        total = len(scores)
        model_correct_count = sum(1 for s in scores if s["model_correct"])
        monitor_correct_count = sum(1 for s in scores if s["monitor_correct"])
        model_wins = sum(1 for s in scores if s["model_correct"] and not s["monitor_correct"])
        results[model_id] = {
            "model_accuracy": model_correct_count / total if total > 0 else 0,
            "monitor_accuracy": monitor_correct_count / total if total > 0 else 0,
            "model_win_rate": model_wins / total if total > 0 else 0,
        }
    return results


def main():
    # Analyze both phases
    phase2_results = analyze_phase(PHASE2_DIR)
    phase3_results = analyze_phase(PHASE3_DIR)

    # Prepare data for plotting
    models = [MODEL_DISPLAY[m] for m in MODEL_ORDER if m in phase2_results and m in phase3_results]
    model_ids = [m for m in MODEL_ORDER if m in phase2_results and m in phase3_results]

    p2_model_acc = [phase2_results[m]["model_accuracy"] * 100 for m in model_ids]
    p3_model_acc = [phase3_results[m]["model_accuracy"] * 100 for m in model_ids]
    p2_monitor_acc = [phase2_results[m]["monitor_accuracy"] * 100 for m in model_ids]
    p3_monitor_acc = [phase3_results[m]["monitor_accuracy"] * 100 for m in model_ids]
    p2_win_rate = [phase2_results[m]["model_win_rate"] * 100 for m in model_ids]
    p3_win_rate = [phase3_results[m]["model_win_rate"] * 100 for m in model_ids]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 2 (Baseline) vs Phase 3 (Adversarial Prompt)", fontsize=14, fontweight='bold')

    x = np.arange(len(models))
    width = 0.35

    # Colors
    color_p2 = '#2ecc71'  # Green for phase 2
    color_p3 = '#e74c3c'  # Red for phase 3

    # Plot 1: Model Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, p2_model_acc, width, label='Phase 2', color=color_p2, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, p3_model_acc, width, label='Phase 3', color=color_p3, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Model Accuracy\n(Correct Answers)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Plot 2: Monitor Accuracy
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, p2_monitor_acc, width, label='Phase 2', color=color_p2, edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, p3_monitor_acc, width, label='Phase 3', color=color_p3, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Monitor Accuracy\n(Predicting Model Answer)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Plot 3: Win Rate (Model correct AND fools monitor)
    ax3 = axes[2]
    bars5 = ax3.bar(x - width/2, p2_win_rate, width, label='Phase 2', color=color_p2, edgecolor='black', linewidth=0.5)
    bars6 = ax3.bar(x + width/2, p3_win_rate, width, label='Phase 3', color=color_p3, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Rate (%)', fontsize=11)
    ax3.set_title('Win Rate\n(Correct + Fools Monitor)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, max(max(p2_win_rate), max(p3_win_rate)) * 1.3 + 5)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars5:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars6:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = WMDB_DIR / "phase_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")

    # Also show
    plt.show()


if __name__ == "__main__":
    main()

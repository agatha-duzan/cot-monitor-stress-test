#!/usr/bin/env python3
"""Generate plots for exp5 (multiturn with planted <think> reasoning).

Creates plots similar to exp3/exp4 stacked transition charts,
but only for the 3 models: Kimi K2, GLM 4.7, Grok 3 Mini (xAI).

Usage:
    source ../../keys.sh
    python mcqa/hle/plot_exp5.py
    python mcqa/hle/plot_exp5.py --llm-cot   # Use GPT-4o-mini for CoT classification
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from analyze import (
    HLE_DIR,
    _get_ack_count,
    analyze_hint_effect,
    analyze_misleading_effect,
    annotate_cot_acknowledgment,
    check_cot_acknowledgment,
    extract_samples,
    load_results,
)

EXP5_MODELS = ["kimi", "glm", "grok_xai"]
MODEL_DISPLAY = {
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok_xai": "Grok 3 Mini",
}


def plot_baseline_and_flips(
    hint_analysis: dict,
    misleading_analysis: dict,
    output_dir: Path,
):
    """Stacked bar chart showing baseline→hint and baseline→misleading transitions.

    Similar to exp3/exp4 baseline_and_flips.png.
    Categories: Right→Right (blue), Wrong→Right (green), Right→Wrong (red), Wrong→Wrong (gray)
    """
    models = [m for m in EXP5_MODELS if m in hint_analysis]
    display_names = [MODEL_DISPLAY[m] for m in models]
    n = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "HLE: Baseline→Condition Transitions (multiturn with planted reasoning)",
        fontsize=14,
        y=0.98,
    )

    for panel_idx, (analysis, condition_name) in enumerate([
        (hint_analysis, "Helpful hint"),
        (misleading_analysis, "Misleading hint"),
    ]):
        ax = axes[panel_idx]
        x = np.arange(n)

        both_correct = []
        wrong_to_right = []
        right_to_wrong = []
        both_wrong = []

        for m in models:
            a = analysis.get(m, {})
            bc = a.get("both_correct", 0)
            both_correct.append(bc)

            if "hint_helped" in a:  # hint analysis
                w2r = a.get("hint_helped", 0)
                r2w = a.get("hint_hurt", 0)
                bw = a.get("both_wrong", 0)
            else:  # misleading analysis
                r2w = a.get("right_to_wrong", 0)
                # For misleading: "wrong to right" = baseline wrong, misleading correct
                total = a.get("n", 100)
                base_correct = int(a.get("baseline_accuracy", 0) * total)
                misl_correct = int(a.get("misleading_accuracy", 0) * total)
                # wrong→right = cases where baseline wrong but misleading correct
                w2r = misl_correct - bc
                bw = total - bc - w2r - r2w

            wrong_to_right.append(w2r)
            right_to_wrong.append(r2w)
            both_wrong.append(bw)

        # Stack from bottom: both_correct, wrong→right, right→wrong, both_wrong
        bars1 = ax.bar(x, both_correct, 0.6, label="Right→Right (both correct)", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x, wrong_to_right, 0.6, bottom=both_correct, label="Wrong→Right (hint helped)", color="#55A868", alpha=0.85)
        bottom2 = [a + b for a, b in zip(both_correct, wrong_to_right)]
        bars3 = ax.bar(x, right_to_wrong, 0.6, bottom=bottom2, label="Right→Wrong (hint hurt)", color="#C44E52", alpha=0.85)
        bottom3 = [a + b for a, b in zip(bottom2, right_to_wrong)]
        bars4 = ax.bar(x, both_wrong, 0.6, bottom=bottom3, label="Wrong→Wrong (both wrong)", color="#AAAAAA", alpha=0.7)

        # Add labels
        for i in range(n):
            # both_correct label
            if both_correct[i] > 0:
                ax.text(i, both_correct[i] / 2, str(both_correct[i]),
                        ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            # wrong→right label
            if wrong_to_right[i] > 0:
                y = both_correct[i] + wrong_to_right[i] / 2
                ax.text(i, y, str(wrong_to_right[i]),
                        ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            # right→wrong label
            if right_to_wrong[i] > 0:
                y = bottom2[i] + right_to_wrong[i] / 2
                ax.text(i, y, str(right_to_wrong[i]),
                        ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            # both_wrong label
            if both_wrong[i] > 3:
                y = bottom3[i] + both_wrong[i] / 2
                ax.text(i, y, str(both_wrong[i]),
                        ha="center", va="center", fontsize=10, color="#444444")

        ax.set_title(condition_name, fontsize=12)
        ax.set_ylabel("Questions (out of 100)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylim(0, 105)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "baseline_and_flips.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_flips_cot_acknowledgment(
    hint_analysis: dict,
    misleading_analysis: dict,
    output_dir: Path,
):
    """Bar chart of Wrong→Right cases with CoT acknowledgment breakdown.

    Similar to exp3/exp4 flips_cot_acknowledgment.png.
    """
    models = [m for m in EXP5_MODELS if m in hint_analysis]
    display_names = [MODEL_DISPLAY[m] for m in models]
    n = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "HLE: Flips with CoT Acknowledgment (multiturn)",
        fontsize=14,
        y=0.98,
    )

    for panel_idx, (analysis, condition_name, reasoning_key, position_key) in enumerate([
        (hint_analysis, "Helpful hint\n(wrong→right flips)", "hint_reasoning", "ground_truth"),
        (misleading_analysis, "Misleading hint\n(right→wrong flips)", "misleading_reasoning", "hint_position"),
    ]):
        ax = axes[panel_idx]
        x = np.arange(n)

        totals = []
        ack_counts = []

        for m in models:
            a = analysis.get(m, {})
            if panel_idx == 0:
                cases = a.get("hint_helped_cases", [])
            else:
                cases = a.get("right_to_wrong_cases", [])

            total = len(cases)
            totals.append(total)

            # Get acknowledgment count (prefers LLM annotation if available)
            ack = _get_ack_count(cases, reasoning_key, position_key)
            ack_counts.append(ack)

        no_ack = [t - a for t, a in zip(totals, ack_counts)]

        bars1 = ax.bar(x, no_ack, 0.5, label="No CoT acknowledgment", color="#C44E52", alpha=0.85)
        bars2 = ax.bar(x, ack_counts, 0.5, bottom=no_ack, label="Acknowledged hint pattern in CoT", color="#55A868", alpha=0.85)

        # Labels
        for i in range(n):
            if totals[i] > 0:
                ax.text(i, totals[i] + 0.3, str(totals[i]),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
            if no_ack[i] > 0:
                ax.text(i, no_ack[i] / 2, str(no_ack[i]),
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            if ack_counts[i] > 0:
                ax.text(i, no_ack[i] + ack_counts[i] / 2, str(ack_counts[i]),
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")

        ax.set_title(condition_name, fontsize=12)
        ax.set_ylabel("Cases (absolute count)" if panel_idx == 0 else "", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylim(0, max(totals) * 1.2 if totals else 10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "flips_cot_acknowledgment.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_accuracy_comparison(
    hint_analysis: dict,
    misleading_analysis: dict,
    output_dir: Path,
):
    """Grouped bar chart comparing baseline/hint/misleading accuracy."""
    models = [m for m in EXP5_MODELS if m in hint_analysis]
    display_names = [MODEL_DISPLAY[m] for m in models]
    n = len(models)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_acc = [hint_analysis[m]["baseline_accuracy"] * 100 for m in models]
    hint_acc = [hint_analysis[m]["hint_accuracy"] * 100 for m in models]
    misl_acc = [
        misleading_analysis[m]["misleading_accuracy"] * 100
        if m in misleading_analysis else 0
        for m in models
    ]

    bars1 = ax.bar(x - width, baseline_acc, width, label="Baseline", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x, hint_acc, width, label="Helpful hint", color="#55A868", alpha=0.85)
    bars3 = ax.bar(x + width, misl_acc, width, label="Misleading hint", color="#C44E52", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("HLE: Accuracy by Condition (multiturn with planted reasoning)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(baseline_acc + hint_acc + misl_acc) * 1.3)
    plt.tight_layout()
    path = output_dir / "accuracy_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_planted_hit_rate(
    misleading_analysis: dict,
    output_dir: Path,
):
    """Bar chart: when misleading hint changes the answer, how often to the planted position?"""
    models = [m for m in EXP5_MODELS if m in misleading_analysis]
    display_names = [MODEL_DISPLAY[m] for m in models]
    n = len(models)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5))

    plant_rates = []
    for m in models:
        a = misleading_analysis[m]
        effect = len(a.get("all_effect_cases", []))
        planted = a.get("total_planted", 0)
        plant_rates.append(planted / effect * 100 if effect > 0 else 0)

    bars = ax.bar(x, plant_rates, 0.5, color="#C44E52", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")
    ax.set_ylabel("Planted Position Hit Rate (%)", fontsize=12)
    ax.set_title(
        "HLE: When Hint Changes Answer,\nHow Often to Planted Position? (multiturn)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylim(0, max(plant_rates) * 1.3 if plant_rates else 50)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = output_dir / "planted_hit_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_flip_rate(
    hint_analysis: dict,
    misleading_analysis: dict,
    output_dir: Path,
):
    """Bar chart comparing flip rates between helpful and misleading."""
    models = [m for m in EXP5_MODELS if m in hint_analysis]
    display_names = [MODEL_DISPLAY[m] for m in models]
    n = len(models)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    hint_flip = []
    misl_flip = []
    for m in models:
        a = hint_analysis[m]
        baseline_wrong = a["n"] - int(a["baseline_accuracy"] * a["n"])
        hint_flip.append(a["hint_helped"] / baseline_wrong * 100 if baseline_wrong > 0 else 0)

        if m in misleading_analysis:
            ma = misleading_analysis[m]
            baseline_right = int(ma["baseline_accuracy"] * ma["n"])
            misl_flip.append(ma["right_to_wrong"] / baseline_right * 100 if baseline_right > 0 else 0)
        else:
            misl_flip.append(0)

    b1 = ax.bar(x - width / 2, hint_flip, width, label="Helpful: wrong→right rate", color="#55A868", alpha=0.85)
    b2 = ax.bar(x + width / 2, misl_flip, width, label="Misleading: right→wrong rate", color="#C44E52", alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Flip Rate (%)", fontsize=12)
    ax.set_title("HLE: Hint Influence — Flip Rates (multiturn)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(hint_flip + misl_flip) * 1.3 if hint_flip else 50)
    plt.tight_layout()
    path = output_dir / "flip_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Plot exp5 multiturn results")
    parser.add_argument("--llm-cot", action="store_true",
                        help="Use GPT-4o-mini for CoT acknowledgment")
    args = parser.parse_args()

    log_dir = HLE_DIR / "logs" / "exp5_multiturn"
    output_dir = HLE_DIR / "plots" / "exp5"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {log_dir}...")
    results = load_results(log_dir)

    # Filter to only our 3 models
    for condition in results:
        results[condition] = {
            m: v for m, v in results[condition].items() if m in EXP5_MODELS
        }

    hint_analysis = analyze_hint_effect(results)
    misleading_analysis = analyze_misleading_effect(results)

    if args.llm_cot:
        print("Running LLM-based CoT acknowledgment...")
        asyncio.run(annotate_cot_acknowledgment(hint_analysis, misleading_analysis))

    # Print summary
    print(f"\n{'='*60}")
    print("EXP5 MULTITURN RESULTS")
    print(f"{'='*60}\n")
    print(f"{'Model':<15} {'N':>4} {'Base%':>7} {'Hint%':>7} {'Misl%':>7} {'Helped':>7} {'Hurt':>6}")
    print("-" * 60)
    for m in EXP5_MODELS:
        if m not in hint_analysis:
            continue
        a = hint_analysis[m]
        ma = misleading_analysis.get(m, {})
        misl = ma.get("misleading_accuracy", 0) * 100 if ma else 0
        display = MODEL_DISPLAY[m]
        print(
            f"{display:<15} {a['n']:>4} {a['baseline_accuracy']*100:>6.1f}% "
            f"{a['hint_accuracy']*100:>6.1f}% {misl:>6.1f}% "
            f"{a['hint_helped']:>7} {a['hint_hurt']:>6}"
        )

    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    plot_baseline_and_flips(hint_analysis, misleading_analysis, output_dir)
    plot_flips_cot_acknowledgment(hint_analysis, misleading_analysis, output_dir)
    plot_accuracy_comparison(hint_analysis, misleading_analysis, output_dir)
    plot_planted_hit_rate(misleading_analysis, output_dir)
    plot_flip_rate(hint_analysis, misleading_analysis, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

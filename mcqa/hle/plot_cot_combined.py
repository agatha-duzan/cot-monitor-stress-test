#!/usr/bin/env python3
"""Combined flips + 3-level CoT acknowledgment plots across exp3, exp4, exp5.

Runs the 3-level classifier (none/noticed/influenced) on all experiments
and generates combined comparison plots.

Usage:
    source ../../keys.sh
    python mcqa/hle/plot_cot_combined.py
"""

import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (
    HLE_DIR,
    MODEL_ORDER,
    MODEL_DISPLAY,
    _get_3level_counts,
    analyze_hint_effect,
    analyze_misleading_effect,
    annotate_cot_3level,
    load_results,
)

# Experiment configs: (log_dir_name, display_label, model_filter or None for all)
EXPERIMENTS = [
    ("exp3_sysprompt_1", "Exp3: strong prefill (8 fewshot)"),
    ("exp3_sysprompt_2", "Exp3: weak prefill (8 fewshot)"),
    ("exp4_sysprompt_1_16", "Exp4: strong prefill (16 fewshot)"),
    ("exp4_sysprompt_2_16", "Exp4: weak prefill (16 fewshot)"),
    ("exp5_multiturn", "Exp5: multiturn + planted reasoning"),
]

# Use consistent model set: the 6 standard + grok_xai
# grok and grok_xai are both Grok 3 Mini but via different APIs
# For exp3/exp4: use grok_xai (xAI direct), for exp5: use grok_xai
MODELS = ["haiku", "sonnet", "opus", "kimi", "glm", "grok_xai"]
DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok_xai": "Grok 3 Mini",
}


def load_experiment(exp_name: str) -> dict:
    """Load and analyze a single experiment."""
    log_dir = HLE_DIR / "logs" / exp_name
    results = load_results(log_dir)

    # Filter to our model set
    for cond in results:
        results[cond] = {m: v for m, v in results[cond].items() if m in MODELS}

    hint = analyze_hint_effect(results)
    misl = analyze_misleading_effect(results)
    return {"hint": hint, "misleading": misl}


def plot_helpful_flips_3level(experiments: list[dict], output_dir: Path):
    """Stacked bars: wrong→right flips with 3-level CoT across experiments.

    One panel per experiment, bars = models, stacked by none/noticed/influenced.
    """
    n_exp = len(experiments)
    fig, axes = plt.subplots(1, n_exp, figsize=(4.2 * n_exp, 6), sharey=True)
    if n_exp == 1:
        axes = [axes]

    fig.suptitle(
        "Helpful Hint: Wrong→Right Flips — CoT Acknowledgment",
        fontsize=14, y=0.98,
    )

    colors = {"none": "#C44E52", "noticed": "#FF9800", "influenced": "#55A868"}

    for panel_idx, (exp, (exp_name, label)) in enumerate(
        zip(experiments, EXPERIMENTS)
    ):
        ax = axes[panel_idx]
        hint = exp["hint"]
        models = [m for m in MODELS if m in hint]
        display_names = [DISPLAY[m] for m in models]
        n = len(models)
        x = np.arange(n)

        none_vals, noticed_vals, influenced_vals = [], [], []
        totals = []

        for m in models:
            cases = hint[m].get("hint_helped_cases", [])
            counts = _get_3level_counts(cases)
            none_vals.append(counts["none"])
            noticed_vals.append(counts["noticed"])
            influenced_vals.append(counts["influenced"])
            totals.append(len(cases))

        b1 = ax.bar(x, none_vals, 0.6, color=colors["none"], alpha=0.85)
        b2 = ax.bar(x, noticed_vals, 0.6, bottom=none_vals,
                     color=colors["noticed"], alpha=0.85)
        bottom2 = [a + b for a, b in zip(none_vals, noticed_vals)]
        b3 = ax.bar(x, influenced_vals, 0.6, bottom=bottom2,
                     color=colors["influenced"], alpha=0.85)

        # Labels
        for i in range(n):
            if totals[i] > 0:
                ax.text(i, totals[i] + 0.3, str(totals[i]),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            for val, bot in [
                (none_vals[i], 0),
                (noticed_vals[i], none_vals[i]),
                (influenced_vals[i], bottom2[i]),
            ]:
                if val > 0:
                    ax.text(i, bot + val / 2, str(val),
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")

        ax.set_title(label, fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel("Wrong→Right cases", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["none"], alpha=0.85, label="No acknowledgment"),
        Patch(facecolor=colors["noticed"], alpha=0.85, label="Pattern noticed"),
        Patch(facecolor=colors["influenced"], alpha=0.85, label="Pattern influenced answer"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "helpful_flips_cot_3level.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_misleading_flips_3level(experiments: list[dict], output_dir: Path):
    """Stacked bars: right→wrong flips with 3-level CoT across experiments."""
    n_exp = len(experiments)
    fig, axes = plt.subplots(1, n_exp, figsize=(4.2 * n_exp, 6), sharey=True)
    if n_exp == 1:
        axes = [axes]

    fig.suptitle(
        "Misleading Hint: Right→Wrong Flips — CoT Acknowledgment",
        fontsize=14, y=0.98,
    )

    colors = {"none": "#C44E52", "noticed": "#FF9800", "influenced": "#55A868"}

    for panel_idx, (exp, (exp_name, label)) in enumerate(
        zip(experiments, EXPERIMENTS)
    ):
        ax = axes[panel_idx]
        misl = exp["misleading"]
        models = [m for m in MODELS if m in misl]
        display_names = [DISPLAY[m] for m in models]
        n = len(models)
        x = np.arange(n)

        none_vals, noticed_vals, influenced_vals = [], [], []
        totals = []

        for m in models:
            cases = misl[m].get("right_to_wrong_cases", [])
            counts = _get_3level_counts(cases)
            none_vals.append(counts["none"])
            noticed_vals.append(counts["noticed"])
            influenced_vals.append(counts["influenced"])
            totals.append(len(cases))

        b1 = ax.bar(x, none_vals, 0.6, color=colors["none"], alpha=0.85)
        b2 = ax.bar(x, noticed_vals, 0.6, bottom=none_vals,
                     color=colors["noticed"], alpha=0.85)
        bottom2 = [a + b for a, b in zip(none_vals, noticed_vals)]
        b3 = ax.bar(x, influenced_vals, 0.6, bottom=bottom2,
                     color=colors["influenced"], alpha=0.85)

        for i in range(n):
            if totals[i] > 0:
                ax.text(i, totals[i] + 0.3, str(totals[i]),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            for val, bot in [
                (none_vals[i], 0),
                (noticed_vals[i], none_vals[i]),
                (influenced_vals[i], bottom2[i]),
            ]:
                if val > 0:
                    ax.text(i, bot + val / 2, str(val),
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")

        ax.set_title(label, fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel("Right→Wrong cases", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["none"], alpha=0.85, label="No acknowledgment"),
        Patch(facecolor=colors["noticed"], alpha=0.85, label="Pattern noticed"),
        Patch(facecolor=colors["influenced"], alpha=0.85, label="Pattern influenced answer"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "misleading_flips_cot_3level.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def plot_cot_rates_by_model(experiments: list[dict], output_dir: Path):
    """Per-model comparison of noticed+influenced rates across experiments.

    Grouped bar chart: models on x-axis, one bar per experiment,
    bar height = (noticed + influenced) / total cases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "CoT Pattern Recognition Rate by Model Across Experiments",
        fontsize=14, y=0.98,
    )

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for panel_idx, (case_key, title, analysis_key) in enumerate([
        ("hint_helped_cases", "Helpful hint (wrong→right)", "hint"),
        ("right_to_wrong_cases", "Misleading hint (right→wrong)", "misleading"),
    ]):
        ax = axes[panel_idx]
        all_models = MODELS
        display_names = [DISPLAY[m] for m in all_models]
        x = np.arange(len(all_models))
        n_exp = len(experiments)
        width = 0.7 / n_exp

        for i, (exp, (exp_name, label)) in enumerate(
            zip(experiments, EXPERIMENTS)
        ):
            analysis = exp[analysis_key]
            rates = []
            for m in all_models:
                if m not in analysis:
                    rates.append(0)
                    continue
                cases = analysis[m].get(case_key, [])
                if not cases:
                    rates.append(0)
                    continue
                counts = _get_3level_counts(cases)
                recognized = counts["noticed"] + counts["influenced"]
                rates.append(recognized / len(cases) * 100)

            offset = (i - (n_exp - 1) / 2) * width
            # Stack noticed (orange bottom) and influenced (red top)
            bars = ax.bar(x + offset, rates, width, label=label,
                         color=colors[i], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("% cases with pattern recognized" if panel_idx == 0 else "",
                      fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15 if ax.get_ylim()[1] > 0 else 10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.94))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "cot_recognition_rate_by_model.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def main():
    output_dir = HLE_DIR / "plots" / "combined_cot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiments
    print("Loading experiments...")
    experiments = []
    for exp_name, label in EXPERIMENTS:
        print(f"  Loading {exp_name}...")
        experiments.append(load_experiment(exp_name))

    # Run 3-level classifier on all
    print("\nRunning 3-level CoT classifier on all experiments...")
    for exp, (exp_name, label) in zip(experiments, EXPERIMENTS):
        print(f"  {exp_name}:")
        asyncio.run(annotate_cot_3level(exp["hint"], exp["misleading"]))

    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    plot_helpful_flips_3level(experiments, output_dir)
    plot_misleading_flips_3level(experiments, output_dir)
    plot_cot_rates_by_model(experiments, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

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


def plot_combined_report(experiments: list[dict], output_dir: Path):
    """Single combined plot for report: all flips by model and method.

    X-axis: 6 model clusters, each bar labelled with method abbreviation.
    Each bar: stacked by CoT acknowledgment (none / noticed / influenced).
    Y-axis: number of answer flips (helpful + misleading combined).
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n_models = len(MODELS)
    n_methods = len(EXPERIMENTS)

    method_abbrevs = ["S8", "W8", "S16", "W16", "MT"]
    method_full = [
        "S8  = Strong prefill (8 fewshot)",
        "W8  = Weak prefill (8 fewshot)",
        "S16 = Strong prefill (16 fewshot)",
        "W16 = Weak prefill (16 fewshot)",
        "MT  = Multiturn + planted reasoning",
    ]

    # Hatches to reinforce method identity (in addition to x-axis labels)
    hatches = ["", "///", r"\\\\", "xx", ".."]
    plt.rcParams["hatch.linewidth"] = 0.7

    cot_colors = {"none": "#C44E52", "noticed": "#FF9800", "influenced": "#55A868"}

    fig, ax = plt.subplots(figsize=(15, 7))

    bar_width = 0.16
    bar_gap = 0.02  # small gap between bars within a cluster
    cluster_spacing = 0.45
    step = bar_width + bar_gap
    cluster_width = n_methods * step
    x_centers = np.arange(n_models) * (cluster_width + cluster_spacing)

    # Collect all bar positions and their method abbreviation labels
    all_tick_pos = []
    all_tick_labels = []
    max_y = 0

    for method_idx, (exp, (exp_name, label)) in enumerate(
        zip(experiments, EXPERIMENTS)
    ):
        hint = exp["hint"]
        misl = exp["misleading"]

        none_vals, noticed_vals, influenced_vals = [], [], []

        for m in MODELS:
            helpful_cases = hint.get(m, {}).get("hint_helped_cases", [])
            misleading_cases = misl.get(m, {}).get("right_to_wrong_cases", [])
            all_cases = helpful_cases + misleading_cases
            counts = _get_3level_counts(all_cases)
            none_vals.append(counts["none"])
            noticed_vals.append(counts["noticed"])
            influenced_vals.append(counts["influenced"])

        offset = (method_idx - (n_methods - 1) / 2) * step
        xpos = x_centers + offset
        hatch = hatches[method_idx]
        hatch_ec = "#444444"

        ax.bar(
            xpos, none_vals, bar_width,
            color=cot_colors["none"], hatch=hatch,
            edgecolor=hatch_ec, linewidth=0.5,
        )
        ax.bar(
            xpos, noticed_vals, bar_width, bottom=none_vals,
            color=cot_colors["noticed"], hatch=hatch,
            edgecolor=hatch_ec, linewidth=0.5,
        )
        bottom2 = [a + b for a, b in zip(none_vals, noticed_vals)]
        ax.bar(
            xpos, influenced_vals, bar_width, bottom=bottom2,
            color=cot_colors["influenced"], hatch=hatch,
            edgecolor=hatch_ec, linewidth=0.5,
        )

        for i in range(n_models):
            total = none_vals[i] + noticed_vals[i] + influenced_vals[i]
            max_y = max(max_y, total)
            all_tick_pos.append(xpos[i])
            all_tick_labels.append(method_abbrevs[method_idx])

    # --- X-axis: method abbreviations per bar (minor) + model names (major) ---
    ax.set_xticks(all_tick_pos)
    ax.set_xticklabels(all_tick_labels, fontsize=7, fontfamily="monospace")
    ax.tick_params(axis="x", length=0, pad=2)

    # Model names below, centred on each cluster
    display_names = [DISPLAY[m] for m in MODELS]
    for i, name in enumerate(display_names):
        ax.text(
            x_centers[i], -0.08, name,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Number of answer flips", fontsize=12)
    ax.set_ylim(0, max_y * 1.10 if max_y > 0 else 10)
    ax.set_title(
        "Planted Positional Hints: Answer Flips by Model and Method\n"
        "(Helpful + Misleading Combined)",
        fontsize=13, pad=12,
    )

    # Legend: CoT colours only (method keys explained as annotation)
    cot_handles = [
        Patch(facecolor=cot_colors["none"], label="No acknowledgment"),
        Patch(facecolor=cot_colors["noticed"], label="Pattern noticed"),
        Patch(facecolor=cot_colors["influenced"], label="Pattern influenced"),
    ]
    ax.legend(
        handles=cot_handles, loc="upper left", fontsize=9,
        title="CoT acknowledgment", title_fontsize=9, framealpha=0.92,
    )

    # Method key as compact text in lower-right
    key_text = "  |  ".join(method_full)
    fig.text(
        0.5, 0.01, key_text,
        ha="center", va="bottom", fontsize=8.5, fontstyle="italic",
        color="#222222",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "combined_flips_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
    plot_combined_report(experiments, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

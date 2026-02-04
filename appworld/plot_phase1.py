#!/usr/bin/env python3
"""Plotting for Phase 1 experiment results.

Generates multi-model visualizations of constraint-driven behavioral changes.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APPWORLD_EXP_DIR = Path(__file__).parent
PLOTS_DIR = APPWORLD_EXP_DIR / "plots" / "phase1"

# Colors per model
MODEL_COLORS = {
    "haiku": "#1f77b4",
    "sonnet": "#ff7f0e",
    "opus": "#2ca02c",
    "kimi": "#d62728",
    "glm": "#9467bd",
    "grok": "#8c564b",
}

MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok": "Grok 3 Mini",
}

# Constraint display info: (short_label, category)
# Loaded dynamically from data, but we define ordering
CATEGORY_ORDER = ["user", "model", "values"]
DIRECTION_ORDER = ["negative", "positive"]


def _get_constraint_info(all_results: dict) -> list[dict]:
    """Extract constraint metadata from results, ordered by category."""
    # Collect all constraint names from the first model's results
    first_model = next(iter(all_results.values()))
    conditions = first_model["metadata"]["conditions"]

    # Parse constraint info
    constraints = []
    for cond in conditions:
        if cond == "baseline":
            continue
        # Find this constraint in any result to get metadata
        for r in first_model["results"]:
            if r["constraint_name"] == cond:
                constraints.append({
                    "name": cond,
                    "category": r.get("constraint_category", ""),
                    "theme": r.get("constraint_theme", ""),
                    "direction": r.get("constraint_direction", ""),
                })
                break

    # Sort: by category order, then by theme, then negative before positive
    def sort_key(c):
        cat_idx = CATEGORY_ORDER.index(c["category"]) if c["category"] in CATEGORY_ORDER else 99
        dir_idx = DIRECTION_ORDER.index(c["direction"]) if c["direction"] in DIRECTION_ORDER else 99
        return (cat_idx, c["theme"], dir_idx)

    constraints.sort(key=sort_key)
    return constraints


def plot_all(all_results: dict):
    """Generate all phase 1 plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_api_calls_by_model(all_results, PLOTS_DIR / "api_calls_by_model.png")
    plot_delta_heatmap(all_results, PLOTS_DIR / "delta_heatmap.png")
    plot_per_model_bars(all_results, PLOTS_DIR)

    print(f"\nPlots saved to: {PLOTS_DIR}")


def plot_api_calls_by_model(all_results: dict, output_file: Path):
    """Bar chart: API calls per constraint, grouped by model."""
    models = [m for m in ["haiku", "sonnet", "opus", "kimi", "glm", "grok"] if m in all_results]
    constraints = _get_constraint_info(all_results)

    # Conditions: baseline + all constraints
    condition_names = ["baseline"] + [c["name"] for c in constraints]
    n_conditions = len(condition_names)
    n_models = len(models)

    # Build display labels
    display_labels = ["Baseline"]
    for c in constraints:
        direction_symbol = "-" if c["direction"] == "negative" else "+"
        display_labels.append(f"{c['theme']}\n({direction_symbol})")

    # Figure
    fig, ax = plt.subplots(figsize=(max(20, n_conditions * 1.2), 8))

    bar_width = 0.8 / n_models
    x = np.arange(n_conditions)

    for i, model_key in enumerate(models):
        data = all_results[model_key]
        by_cond = data["analysis"]["by_condition"]

        means = []
        stds = []
        for cond in condition_names:
            if cond in by_cond:
                means.append(by_cond[cond]["avg_api_calls"])
                stds.append(by_cond[cond]["std_api_calls"])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=2,
               label=MODEL_DISPLAY.get(model_key, model_key),
               color=MODEL_COLORS.get(model_key, f"C{i}"),
               edgecolor="black", linewidth=0.5, alpha=0.85)

    # Category separators
    prev_cat = None
    for i, c in enumerate(constraints):
        if prev_cat is not None and c["category"] != prev_cat:
            ax.axvline(x=i + 0.5, color="gray", linestyle=":", alpha=0.4)
        prev_cat = c["category"]
    # Separator after baseline
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

    # Category labels at top
    cat_positions = {"baseline": [0]}
    for i, c in enumerate(constraints):
        cat_positions.setdefault(c["category"], []).append(i + 1)

    ylim = ax.get_ylim()
    for cat, positions in cat_positions.items():
        mid = (min(positions) + max(positions)) / 2
        label = cat.capitalize() if cat != "baseline" else "Base"
        ax.text(mid, ylim[1] * 0.97, label, ha="center", fontsize=10, style="italic", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("API Calls (mean ± std)", fontsize=12)
    ax.set_title("API Call Count by Constraint and Model", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_delta_heatmap(all_results: dict, output_file: Path):
    """Heatmap of % delta from baseline for each model × constraint."""
    models = [m for m in ["haiku", "sonnet", "opus", "kimi", "glm", "grok"] if m in all_results]
    constraints = _get_constraint_info(all_results)
    constraint_names = [c["name"] for c in constraints]

    # Build matrix
    matrix = np.full((len(models), len(constraint_names)), np.nan)
    for i, model_key in enumerate(models):
        deltas = all_results[model_key]["analysis"]["behavioral_deltas"]
        for j, cname in enumerate(constraint_names):
            if cname in deltas:
                matrix[i, j] = deltas[cname]["api_call_delta_pct"]

    # Display labels
    y_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    x_labels = []
    for c in constraints:
        d = "-" if c["direction"] == "negative" else "+"
        x_labels.append(f"{c['theme']} ({d})")

    fig, ax = plt.subplots(figsize=(max(16, len(constraint_names) * 0.8), max(5, len(models) * 0.8)))

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 50)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(constraint_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(len(constraint_names)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_title("% Change in API Calls vs Baseline", fontsize=14)

    # Category separators
    prev_cat = None
    for j, c in enumerate(constraints):
        if prev_cat is not None and c["category"] != prev_cat:
            ax.axvline(x=j - 0.5, color="black", linewidth=1.5)
        prev_cat = c["category"]

    plt.colorbar(im, ax=ax, label="% Delta from Baseline", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_per_model_bars(all_results: dict, output_dir: Path):
    """Individual bar chart per model (simpler view)."""
    for model_key, data in all_results.items():
        constraints = _get_constraint_info(all_results)
        condition_names = ["baseline"] + [c["name"] for c in constraints]
        by_cond = data["analysis"]["by_condition"]

        display_labels = ["Baseline"]
        colors = ["#888888"]
        cat_colors = {"user": "#4a86c7", "model": "#8a5ac7", "values": "#c74a4a"}
        for c in constraints:
            d = "-" if c["direction"] == "negative" else "+"
            display_labels.append(f"{c['theme']}\n({d})")
            colors.append(cat_colors.get(c["category"], "#888888"))

        means = []
        stds = []
        for cond in condition_names:
            if cond in by_cond:
                means.append(by_cond[cond]["avg_api_calls"])
                stds.append(by_cond[cond]["std_api_calls"])
            else:
                means.append(0)
                stds.append(0)

        fig, ax = plt.subplots(figsize=(max(14, len(condition_names) * 0.7), 6))
        x = np.arange(len(condition_names))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor="black", linewidth=0.5)

        # Baseline reference
        baseline_mean = by_cond.get("baseline", {}).get("avg_api_calls", 0)
        if baseline_mean > 0:
            ax.axhline(y=baseline_mean, color="black", linestyle="--",
                       linewidth=1.5, alpha=0.6, label=f"Baseline ({baseline_mean:.1f})")

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("API Calls", fontsize=11)
        display = MODEL_DISPLAY.get(model_key, model_key)
        ax.set_title(f"{display}: API Calls by Constraint (n={data['metadata']['samples_per_condition']})", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        out = output_dir / f"api_calls_{model_key}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=str(PHASE1_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_results = {}
    for model_key in ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]:
        files = sorted(results_dir.glob(f"phase1_{model_key}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                all_results[model_key] = json.load(f)
            print(f"Loaded {model_key}: {files[0].name}")

    if all_results:
        plot_all(all_results)
    else:
        print("No results found")

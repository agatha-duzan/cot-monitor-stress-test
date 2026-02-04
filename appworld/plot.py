#!/usr/bin/env python3
"""Unified plotting and analysis for AppWorld experiments.

Subcommands:
    pilot         - Plot pilot API calls and monitor performance
    phase1        - Multi-model constraint effect visualizations
    phase2        - Per-task and aggregate constraint plots
    survey        - Task survey success rates and API call variance
    analyze-pilot - Run monitors on pilot trajectories and report

Usage:
    python appworld/plot.py pilot --latest
    python appworld/plot.py phase1
    python appworld/plot.py phase2 --results-file logs/phase2/phase2_xxx.json
    python appworld/plot.py survey --results-file logs/easy_test/easy_test_xxx.json
    python appworld/plot.py analyze-pilot --latest --save
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import APPWORLD_EXP_DIR, load_local_module

config = load_local_module("config")
CONSTRAINTS = config.CONSTRAINTS
MODEL_ORDER = config.MODEL_ORDER

# ──────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ──────────────────────────────────────────────────────────────────────

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

CATEGORY_ORDER = ["user", "model", "values"]
DIRECTION_ORDER = ["negative", "positive"]


# ──────────────────────────────────────────────────────────────────────
# PILOT PLOTTING (from plot_results.py)
# ──────────────────────────────────────────────────────────────────────

def plot_pilot_api_calls(data: dict, output_file: Path | None = None):
    """Create bar plot of API calls by condition."""
    analysis = data["analysis"]
    by_condition = analysis["by_condition"]

    condition_order = [
        "baseline",
        "user_negative", "user_positive",
        "model_negative", "model_positive",
        "values_negative", "values_positive",
    ]
    condition_order = [c for c in condition_order if c in by_condition]

    means = [by_condition[c]["avg_api_calls"] for c in condition_order]
    stds = [by_condition[c].get("std_api_calls", 0) for c in condition_order]

    colors = []
    for c in condition_order:
        if c == "baseline":
            colors.append("#888888")
        elif "negative" in c:
            colors.append("#4a86c7")
        else:
            colors.append("#c74a4a")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(condition_order))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1)

    baseline_mean = by_condition["baseline"]["avg_api_calls"]
    ax.axhline(y=baseline_mean, color="black", linestyle="--", linewidth=2, label=f"Baseline ({baseline_mean:.1f})")

    display_labels = []
    for c in condition_order:
        if c == "baseline":
            display_labels.append("Baseline")
        else:
            parts = c.split("_")
            category = parts[0].capitalize()
            direction = "(-)" if parts[1] == "negative" else "(+)"
            display_labels.append(f"{category}\n{direction}")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_ylabel("Average API Calls", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("API Call Behavior by Constraint Condition (n=5 per condition)", fontsize=14)
    ax.legend(loc="upper right")

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        delta = ""
        if condition_order[i] != "baseline":
            delta_val = analysis["behavioral_deltas"].get(condition_order[i], {}).get("api_call_delta_pct", 0)
            delta = f"\n({delta_val:+.0f}%)"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                f"{mean:.1f}{delta}", ha="center", va="bottom", fontsize=10)

    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=2.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.5)

    ax.text(0, ax.get_ylim()[1] * 0.95, "Baseline", ha="center", fontsize=10, style="italic")
    ax.text(1.5, ax.get_ylim()[1] * 0.95, "User Cost", ha="center", fontsize=10, style="italic")
    ax.text(3.5, ax.get_ylim()[1] * 0.95, "Model Incentive", ha="center", fontsize=10, style="italic")
    ax.text(5.5, ax.get_ylim()[1] * 0.95, "Values-Based", ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
    plt.close(fig)
    return fig


def plot_pilot_monitors(data: dict, output_file: Path | None = None):
    """Run monitors on all trajectories and plot suspicion score distributions."""
    import os
    import sys

    monitor_module = load_local_module("monitor")
    analyze_trajectory_with_monitors = monitor_module.analyze_trajectory_with_monitors

    results_by_condition = {}
    for result in data.get("results", []):
        constraint_name = result.get("constraint_name", "unknown")
        trajectory = result.get("trajectory", {})

        if not trajectory.get("turns"):
            continue

        true_constraint = None if constraint_name == "baseline" else constraint_name

        try:
            monitor_result = analyze_trajectory_with_monitors(trajectory, true_constraint)

            if constraint_name not in results_by_condition:
                results_by_condition[constraint_name] = {
                    "internal_scores": [], "external_scores": [],
                    "internal_explanations": [], "external_explanations": [],
                    "internal_categories": [], "external_categories": [],
                }

            internal_score = monitor_result.get("internal_monitor", {}).get("result", {}).get("suspicion_score", 0)
            external_score = monitor_result.get("external_monitor", {}).get("result", {}).get("suspicion_score", 0)
            internal_explanation = monitor_result.get("internal_monitor", {}).get("result", {}).get("explanation", "")
            external_explanation = monitor_result.get("external_monitor", {}).get("result", {}).get("explanation", "")
            internal_category = monitor_result.get("internal_monitor", {}).get("result", {}).get("influence_category", "none")
            external_category = monitor_result.get("external_monitor", {}).get("result", {}).get("influence_category", "none")

            results_by_condition[constraint_name]["internal_scores"].append(internal_score)
            results_by_condition[constraint_name]["external_scores"].append(external_score)
            results_by_condition[constraint_name]["internal_explanations"].append(internal_explanation)
            results_by_condition[constraint_name]["external_explanations"].append(external_explanation)
            results_by_condition[constraint_name]["internal_categories"].append(internal_category)
            results_by_condition[constraint_name]["external_categories"].append(external_category)

            print(f"  {constraint_name}: internal={internal_score:.2f}, external={external_score:.2f}")

        except Exception as e:
            print(f"  Error analyzing {constraint_name}: {e}")

    condition_order = [
        "baseline",
        "user_negative", "user_positive",
        "model_negative", "model_positive",
        "values_negative", "values_positive",
    ]
    condition_order = [c for c in condition_order if c in results_by_condition]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    internal_data = [results_by_condition[c]["internal_scores"] for c in condition_order]
    external_data = [results_by_condition[c]["external_scores"] for c in condition_order]

    display_labels = []
    for c in condition_order:
        if c == "baseline":
            display_labels.append("Baseline")
        else:
            parts = c.split("_")
            category = parts[0].capitalize()
            direction = "(-)" if parts[1] == "negative" else "(+)"
            display_labels.append(f"{category}\n{direction}")

    colors = []
    for c in condition_order:
        if c == "baseline":
            colors.append("#888888")
        elif "user" in c:
            colors.append("#4a86c7")
        elif "model" in c:
            colors.append("#8a5ac7")
        else:
            colors.append("#c74a4a")

    # Internal monitor
    bp1 = ax1.boxplot(internal_data, positions=range(len(condition_order)), widths=0.6, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (scores, color) in enumerate(zip(internal_data, colors)):
        jittered_x = np.random.normal(i, 0.08, len(scores))
        ax1.scatter(jittered_x, scores, color=color, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, zorder=3)
    ax1.set_xticks(range(len(condition_order)))
    ax1.set_xticklabels(display_labels, fontsize=10)
    ax1.set_ylabel("Suspicion Score", fontsize=12)
    ax1.set_xlabel("Condition", fontsize=12)
    ax1.set_title("Internal Monitor (thinking blocks)\nSuspicion Scores by Condition", fontsize=13)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    ax1.axhline(y=0.3, color="orange", linestyle=":", alpha=0.5, label="Low threshold (0.3)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)
    for i, scores in enumerate(internal_data):
        ax1.text(i, -0.12, f"μ={np.mean(scores):.2f}", ha="center", fontsize=9, color="black")

    # External monitor
    bp2 = ax2.boxplot(external_data, positions=range(len(condition_order)), widths=0.6, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (scores, color) in enumerate(zip(external_data, colors)):
        jittered_x = np.random.normal(i, 0.08, len(scores))
        ax2.scatter(jittered_x, scores, color=color, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, zorder=3)
    ax2.set_xticks(range(len(condition_order)))
    ax2.set_xticklabels(display_labels, fontsize=10)
    ax2.set_ylabel("Suspicion Score", fontsize=12)
    ax2.set_xlabel("Condition", fontsize=12)
    ax2.set_title("External Monitor (ReAct thoughts)\nSuspicion Scores by Condition", fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    ax2.axhline(y=0.3, color="orange", linestyle=":", alpha=0.5, label="Low threshold (0.3)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)
    for i, scores in enumerate(external_data):
        ax2.text(i, -0.12, f"μ={np.mean(scores):.2f}", ha="center", fontsize=9, color="black")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
    plt.close(fig)

    # Print summary
    print("\n--- Suspicion Score Summary ---")
    print(f"{'Condition':<20} {'Internal (mean±std)':<20} {'External (mean±std)':<20}")
    print("-" * 60)
    for c in condition_order:
        int_scores = results_by_condition[c]["internal_scores"]
        ext_scores = results_by_condition[c]["external_scores"]
        int_mean, int_std = np.mean(int_scores), np.std(int_scores)
        ext_mean, ext_std = np.mean(ext_scores), np.std(ext_scores)
        print(f"{c:<20} {int_mean:.2f} ± {int_std:.2f}             {ext_mean:.2f} ± {ext_std:.2f}")

    return fig, results_by_condition


def cmd_pilot(args):
    """Plot pilot results."""
    if args.results_file:
        results_file = Path(args.results_file)
    elif args.latest:
        pilot_dir = APPWORLD_EXP_DIR / "logs" / "pilot"
        pilot_files = sorted(pilot_dir.glob("pilot_*.json"), reverse=True)
        if not pilot_files:
            print("No pilot files found")
            return
        results_file = pilot_files[0]
    else:
        print("Specify --results-file or --latest")
        return

    print(f"Loading: {results_file}")
    with open(results_file) as f:
        data = json.load(f)

    output_dir = APPWORLD_EXP_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating API calls plot...")
    plot_pilot_api_calls(data, output_dir / "api_calls_by_condition.png")

    if not args.no_monitor:
        print("\nRunning monitors and generating performance plot...")
        print("(This may take a few minutes as it calls GPT-4o for each trajectory)")
        plot_pilot_monitors(data, output_dir / "monitor_performance.png")

    print(f"\nPlots saved to: {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# PHASE 1 PLOTTING (from plot_phase1.py)
# ──────────────────────────────────────────────────────────────────────

def _get_constraint_info(all_results: dict) -> list[dict]:
    """Extract constraint metadata from results, ordered by category."""
    first_model = next(iter(all_results.values()))
    conditions = first_model["metadata"]["conditions"]

    constraints = []
    for cond in conditions:
        if cond == "baseline":
            continue
        for r in first_model["results"]:
            if r["constraint_name"] == cond:
                constraints.append({
                    "name": cond,
                    "category": r.get("constraint_category", ""),
                    "theme": r.get("constraint_theme", ""),
                    "direction": r.get("constraint_direction", ""),
                })
                break

    def sort_key(c):
        cat_idx = CATEGORY_ORDER.index(c["category"]) if c["category"] in CATEGORY_ORDER else 99
        dir_idx = DIRECTION_ORDER.index(c["direction"]) if c["direction"] in DIRECTION_ORDER else 99
        return (cat_idx, c["theme"], dir_idx)

    constraints.sort(key=sort_key)
    return constraints


def plot_phase1_api_calls_by_model(all_results: dict, output_file: Path):
    """Bar chart: API calls per constraint, grouped by model."""
    models = [m for m in MODEL_ORDER if m in all_results]
    constraints = _get_constraint_info(all_results)

    condition_names = ["baseline"] + [c["name"] for c in constraints]
    n_conditions = len(condition_names)
    n_models = len(models)

    display_labels = ["Baseline"]
    for c in constraints:
        direction_symbol = "-" if c["direction"] == "negative" else "+"
        display_labels.append(f"{c['theme']}\n({direction_symbol})")

    fig, ax = plt.subplots(figsize=(max(20, n_conditions * 1.2), 8))
    bar_width = 0.8 / n_models
    x = np.arange(n_conditions)

    for i, model_key in enumerate(models):
        data = all_results[model_key]
        by_cond = data["analysis"]["by_condition"]
        means, stds = [], []
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

    prev_cat = None
    for i, c in enumerate(constraints):
        if prev_cat is not None and c["category"] != prev_cat:
            ax.axvline(x=i + 0.5, color="gray", linestyle=":", alpha=0.4)
        prev_cat = c["category"]
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

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


def plot_phase1_delta_heatmap(all_results: dict, output_file: Path):
    """Heatmap of % delta from baseline for each model × constraint."""
    models = [m for m in MODEL_ORDER if m in all_results]
    constraints = _get_constraint_info(all_results)
    constraint_names = [c["name"] for c in constraints]

    matrix = np.full((len(models), len(constraint_names)), np.nan)
    for i, model_key in enumerate(models):
        deltas = all_results[model_key]["analysis"]["behavioral_deltas"]
        for j, cname in enumerate(constraint_names):
            if cname in deltas:
                matrix[i, j] = deltas[cname]["api_call_delta_pct"]

    y_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    x_labels = []
    for c in constraints:
        d = "-" if c["direction"] == "negative" else "+"
        x_labels.append(f"{c['theme']} ({d})")

    fig, ax = plt.subplots(figsize=(max(16, len(constraint_names) * 0.8), max(5, len(models) * 0.8)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 50)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

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


def plot_phase1_per_model_bars(all_results: dict, output_dir: Path):
    """Individual bar chart per model."""
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

        means, stds = [], []
        for cond in condition_names:
            if cond in by_cond:
                means.append(by_cond[cond]["avg_api_calls"])
                stds.append(by_cond[cond]["std_api_calls"])
            else:
                means.append(0)
                stds.append(0)

        fig, ax = plt.subplots(figsize=(max(14, len(condition_names) * 0.7), 6))
        x = np.arange(len(condition_names))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)

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


def plot_phase1_all(all_results: dict):
    """Generate all phase 1 plots."""
    plots_dir = APPWORLD_EXP_DIR / "plots" / "phase1"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_phase1_api_calls_by_model(all_results, plots_dir / "api_calls_by_model.png")
    plot_phase1_delta_heatmap(all_results, plots_dir / "delta_heatmap.png")
    plot_phase1_per_model_bars(all_results, plots_dir)

    print(f"\nPlots saved to: {plots_dir}")


def load_phase1_results() -> dict[str, dict]:
    """Load the latest phase1 results for each model."""
    phase1_dir = APPWORLD_EXP_DIR / "logs" / "phase1"
    results = {}
    if not phase1_dir.exists():
        return results
    for model_key in MODEL_ORDER:
        files = sorted(phase1_dir.glob(f"phase1_{model_key}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                results[model_key] = json.load(f)
            print(f"Loaded {model_key}: {files[0].name}")
    return results


def cmd_phase1(args):
    """Plot phase1 results."""
    if hasattr(args, 'results_dir') and args.results_dir:
        results_dir = Path(args.results_dir)
        all_results = {}
        for model_key in MODEL_ORDER:
            files = sorted(results_dir.glob(f"phase1_{model_key}_*.json"), reverse=True)
            if files:
                with open(files[0]) as f:
                    all_results[model_key] = json.load(f)
                print(f"Loaded {model_key}: {files[0].name}")
    else:
        all_results = load_phase1_results()

    if not all_results:
        print("No results found in logs/phase1/")
        return
    plot_phase1_all(all_results)


# ──────────────────────────────────────────────────────────────────────
# PHASE 2 PLOTTING (from run_phase2.py)
# ──────────────────────────────────────────────────────────────────────

PHASE2_TASKS = ["afc4005_1", "425a494_1", "7847649_1", "dac78d9_1", "1150ed6_1"]


def load_phase2_baseline():
    """Load baseline results for opus on the 5 tasks from easy_test."""
    baseline = defaultdict(list)
    log_dir = APPWORLD_EXP_DIR / "logs" / "easy_test"
    for f in sorted(log_dir.glob("easy_test_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if r["model_key"] == "opus" and r["task_id"] in PHASE2_TASKS:
                if r["total_api_calls"] > 0 or r["turn_count"] > 0:
                    baseline[r["task_id"]].append(r)
    return baseline


def plot_phase2_results(results, baseline):
    """Generate all phase 2 plots."""
    constraint_names = sorted(CONSTRAINTS.keys())
    constraint_ids = sorted(set(CONSTRAINTS[c]["constraint_id"] for c in constraint_names))

    plot_dir = APPWORLD_EXP_DIR / "plots" / "phase2"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Per-task bar charts ---
    for task_id in PHASE2_TASKS:
        bl_runs = baseline.get(task_id, [])
        bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 0

        task_results = [r for r in results if r["task_id"] == task_id]
        by_constraint = defaultdict(list)
        for r in task_results:
            by_constraint[r["constraint_name"]].append(r)

        fig, ax = plt.subplots(figsize=(16, 5))
        x = np.arange(len(constraint_ids))
        w = 0.35

        neg_means, neg_stds = [], []
        pos_means, pos_stds = [], []

        for cid in constraint_ids:
            neg_runs = by_constraint.get(f"{cid}_negative", [])
            pos_runs = by_constraint.get(f"{cid}_positive", [])
            neg_apis = [r["total_api_calls"] for r in neg_runs] if neg_runs else [0]
            pos_apis = [r["total_api_calls"] for r in pos_runs] if pos_runs else [0]
            neg_means.append(np.mean(neg_apis))
            neg_stds.append(np.std(neg_apis))
            pos_means.append(np.mean(pos_apis))
            pos_stds.append(np.std(pos_apis))

        ax.bar(x - w/2, neg_means, w, yerr=neg_stds, label="Negative", color="#3498db", alpha=0.85, capsize=2)
        ax.bar(x + w/2, pos_means, w, yerr=pos_stds, label="Positive", color="#e74c3c", alpha=0.85, capsize=2)
        ax.axhline(bl_api, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({bl_api:.1f})")

        ax.set_ylabel("API Calls (mean ± std)")
        ax.set_title(f"Opus — {task_id}: API Calls by Constraint")
        ax.set_xticks(x)
        ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"api_calls_{task_id}.png", dpi=150)
        print(f"Saved: {plot_dir}/api_calls_{task_id}.png")
        plt.close(fig)

    # --- Plot 2: Delta heatmap ---
    fig, ax = plt.subplots(figsize=(18, 6))
    col_labels = []
    for cid in constraint_ids:
        col_labels.append(f"{cid}_neg")
        col_labels.append(f"{cid}_pos")

    matrix = np.zeros((len(PHASE2_TASKS), len(col_labels)))
    for i, task_id in enumerate(PHASE2_TASKS):
        bl_runs = baseline.get(task_id, [])
        bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 1
        task_results_list = [r for r in results if r["task_id"] == task_id]
        by_constraint = defaultdict(list)
        for r in task_results_list:
            by_constraint[r["constraint_name"]].append(r)

        for j, cid in enumerate(constraint_ids):
            for k, direction in enumerate(["negative", "positive"]):
                cname = f"{cid}_{direction}"
                runs = by_constraint.get(cname, [])
                avg_api = np.mean([r["total_api_calls"] for r in runs]) if runs else bl_api
                delta_pct = (avg_api - bl_api) / bl_api * 100 if bl_api > 0 else 0
                matrix[i, j * 2 + k] = delta_pct

    vmax = max(abs(matrix.min()), abs(matrix.max()), 50)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(PHASE2_TASKS)))
    ax.set_yticklabels([t.split("_")[0] for t in PHASE2_TASKS], fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_title("% Change in API Calls vs Baseline (Opus)")
    fig.colorbar(im, ax=ax, label="% change", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_heatmap.png", dpi=150)
    print(f"Saved: {plot_dir}/delta_heatmap.png")
    plt.close(fig)

    # --- Plot 3: Aggregated deltas ---
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(constraint_ids))
    w = 0.35

    neg_deltas, pos_deltas = [], []
    for cid in constraint_ids:
        neg_d, pos_d = [], []
        for task_id in PHASE2_TASKS:
            bl_runs = baseline.get(task_id, [])
            bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 1
            task_res = [r for r in results if r["task_id"] == task_id]
            by_c = defaultdict(list)
            for r in task_res:
                by_c[r["constraint_name"]].append(r)
            neg_runs = by_c.get(f"{cid}_negative", [])
            pos_runs = by_c.get(f"{cid}_positive", [])
            if neg_runs and bl_api > 0:
                neg_d.append((np.mean([r["total_api_calls"] for r in neg_runs]) - bl_api) / bl_api * 100)
            if pos_runs and bl_api > 0:
                pos_d.append((np.mean([r["total_api_calls"] for r in pos_runs]) - bl_api) / bl_api * 100)
        neg_deltas.append(np.mean(neg_d) if neg_d else 0)
        pos_deltas.append(np.mean(pos_d) if pos_d else 0)

    ax.bar(x - w/2, neg_deltas, w, label="Negative (expect fewer)", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, pos_deltas, w, label="Positive (expect more)", color="#e74c3c", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Avg % Change in API Calls vs Baseline")
    ax.set_title("Opus: Average Effect of Constraints Across 5 Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "avg_delta_by_constraint.png", dpi=150)
    print(f"Saved: {plot_dir}/avg_delta_by_constraint.png")
    plt.close(fig)

    # --- Plot 4: Success rate comparison ---
    fig, ax = plt.subplots(figsize=(16, 5))
    neg_sr, pos_sr = [], []
    for cid in constraint_ids:
        neg_all, pos_all = [], []
        for task_id in PHASE2_TASKS:
            task_res = [r for r in results if r["task_id"] == task_id]
            by_c = defaultdict(list)
            for r in task_res:
                by_c[r["constraint_name"]].append(r)
            neg_all.extend(by_c.get(f"{cid}_negative", []))
            pos_all.extend(by_c.get(f"{cid}_positive", []))
        neg_sr.append(sum(1 for r in neg_all if r["success"]) / max(len(neg_all), 1))
        pos_sr.append(sum(1 for r in pos_all if r["success"]) / max(len(pos_all), 1))

    all_bl = [r for bl in baseline.values() for r in bl]
    bl_sr_val = sum(1 for r in all_bl if r["success"]) / max(len(all_bl), 1)

    ax.bar(x - w/2, neg_sr, w, label="Negative", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, pos_sr, w, label="Positive", color="#e74c3c", alpha=0.85)
    ax.axhline(bl_sr_val, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({bl_sr_val:.0%})")
    ax.set_ylabel("Success Rate")
    ax.set_title("Opus: Success Rate by Constraint (across 5 tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "success_by_constraint.png", dpi=150)
    print(f"Saved: {plot_dir}/success_by_constraint.png")
    plt.close(fig)


def cmd_phase2(args):
    """Plot phase2 results."""
    if hasattr(args, 'results_file') and args.results_file:
        results_file = Path(args.results_file)
    else:
        phase2_dir = APPWORLD_EXP_DIR / "logs" / "phase2"
        files = sorted(phase2_dir.glob("phase2_*.json"), reverse=True)
        if not files:
            print("No phase2 results found")
            return
        results_file = files[0]

    print(f"Loading: {results_file}")
    with open(results_file) as f:
        data = json.load(f)

    baseline = load_phase2_baseline()
    plot_phase2_results(data["results"], baseline)


# ──────────────────────────────────────────────────────────────────────
# SURVEY PLOTTING (from run_task_survey.py)
# ──────────────────────────────────────────────────────────────────────

SURVEY_MODELS = ["opus", "kimi"]


def plot_survey_results(results: list[dict]):
    """Plot success rates and API call variance per task."""
    plot_dir = APPWORLD_EXP_DIR / "plots" / "easy_test"
    plot_dir.mkdir(parents=True, exist_ok=True)

    task_model_stats = {}
    for r in results:
        key = (r["task_id"], r["model_key"])
        task_model_stats.setdefault(key, []).append(r)

    tasks = sorted(set(r["task_id"] for r in results))
    model_colors = {"opus": "#2ecc71", "kimi": "#e74c3c"}

    # --- Plot 1: Success rate per task ---
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(tasks))
    bar_w = 0.35

    for i, model_key in enumerate(SURVEY_MODELS):
        rates = []
        for task_id in tasks:
            runs = task_model_stats.get((task_id, model_key), [])
            sr = sum(1 for r in runs if r["success"]) / max(len(runs), 1)
            rates.append(sr)
        ax.bar(x + i * bar_w, rates, bar_w, label=model_key,
               color=model_colors.get(model_key, f"C{i}"), alpha=0.85)

    ax.set_ylabel("Success Rate")
    ax.set_title("Baseline Success Rate per Task (10 samples)")
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels([t.split("_")[0] for t in tasks], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "success_rate.png", dpi=150)
    print(f"Saved: {plot_dir / 'success_rate.png'}")
    plt.close(fig)

    # --- Plot 2: API calls with variance ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, model_key in enumerate(SURVEY_MODELS):
        means, stds = [], []
        for task_id in tasks:
            runs = task_model_stats.get((task_id, model_key), [])
            api_calls = [r["total_api_calls"] for r in runs]
            means.append(np.mean(api_calls) if api_calls else 0)
            stds.append(np.std(api_calls) if api_calls else 0)
        ax.bar(x + i * bar_w, means, bar_w, yerr=stds,
               label=model_key, color=model_colors.get(model_key, f"C{i}"),
               alpha=0.85, capsize=3)

    ax.set_ylabel("API Calls (mean ± std)")
    ax.set_title("API Calls per Task — Variance across 10 samples")
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels([t.split("_")[0] for t in tasks], rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "api_calls_variance.png", dpi=150)
    print(f"Saved: {plot_dir / 'api_calls_variance.png'}")
    plt.close(fig)

    # --- Plot 3: Viable tasks only ---
    good_tasks = []
    for task_id in tasks:
        for model_key in SURVEY_MODELS:
            runs = task_model_stats.get((task_id, model_key), [])
            sr = sum(1 for r in runs if r["success"]) / max(len(runs), 1)
            if sr >= 0.5:
                good_tasks.append(task_id)
                break

    if good_tasks:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(good_tasks) * 0.8), 8), sharex=True)
        xg = np.arange(len(good_tasks))

        for i, model_key in enumerate(SURVEY_MODELS):
            rates, means, stds = [], [], []
            for task_id in good_tasks:
                runs = task_model_stats.get((task_id, model_key), [])
                api_calls = [r["total_api_calls"] for r in runs]
                rates.append(sum(1 for r in runs if r["success"]) / max(len(runs), 1))
                means.append(np.mean(api_calls) if api_calls else 0)
                stds.append(np.std(api_calls) if api_calls else 0)

            ax1.bar(xg + i * bar_w, rates, bar_w, label=model_key,
                    color=model_colors.get(model_key, f"C{i}"), alpha=0.85)
            ax2.bar(xg + i * bar_w, means, bar_w, yerr=stds,
                    label=model_key, color=model_colors.get(model_key, f"C{i}"),
                    alpha=0.85, capsize=3)

        ax1.set_ylabel("Success Rate")
        ax1.set_title("Viable Tasks (≥50% success on at least one model)")
        ax1.set_ylim(0, 1.05)
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.set_ylabel("API Calls (mean ± std)")
        ax2.set_xticks(xg + bar_w / 2)
        ax2.set_xticklabels([t.split("_")[0] for t in good_tasks], rotation=45, ha="right")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(plot_dir / "viable_tasks.png", dpi=150)
        print(f"Saved: {plot_dir / 'viable_tasks.png'}")
        plt.close(fig)
    else:
        print("No viable tasks found (none with >=50% success).")


def cmd_survey(args):
    """Plot survey results."""
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        survey_dir = APPWORLD_EXP_DIR / "logs" / "easy_test"
        files = sorted(survey_dir.glob("easy_test_*.json"), reverse=True)
        if not files:
            print("No survey results found")
            return
        results_file = files[0]

    print(f"Loading: {results_file}")
    with open(results_file) as f:
        data = json.load(f)

    plot_survey_results(data["results"])


# ──────────────────────────────────────────────────────────────────────
# ANALYZE PILOT (from analyze_pilot.py)
# ──────────────────────────────────────────────────────────────────────

def analyze_pilot_file(pilot_file: Path) -> dict:
    """Analyze all trajectories in a pilot file with monitors."""
    monitor_module = load_local_module("monitor")
    analyze_trajectory_with_monitors = monitor_module.analyze_trajectory_with_monitors

    with open(pilot_file) as f:
        pilot_data = json.load(f)

    results = []
    for i, exp_result in enumerate(pilot_data.get("results", [])):
        constraint_name = exp_result.get("constraint_name")
        trajectory = exp_result.get("trajectory", {})

        if not trajectory.get("turns"):
            print(f"  Skipping {constraint_name}: no turns in trajectory")
            continue

        print(f"  Analyzing {constraint_name}...")
        true_constraint = None if constraint_name == "baseline" else constraint_name

        try:
            monitor_result = analyze_trajectory_with_monitors(trajectory, true_constraint)
            monitor_result["experiment_index"] = i
            monitor_result["constraint_name"] = constraint_name
            monitor_result["task_success"] = exp_result.get("success", False)
            results.append(monitor_result)
        except Exception as e:
            print(f"    Error analyzing {constraint_name}: {e}")

    return {
        "metadata": pilot_data.get("metadata", {}),
        "pilot_analysis": pilot_data.get("analysis", {}),
        "monitor_results": results,
    }


def compute_pilot_summary_statistics(results: list[dict]) -> dict:
    """Compute summary statistics across all monitor results."""
    stats = {
        "total_samples": len(results),
        "by_constraint": defaultdict(lambda: {
            "count": 0,
            "internal_level_1": 0, "internal_level_2": 0, "internal_level_3": 0,
            "external_level_1": 0, "external_level_2": 0, "external_level_3": 0,
        }),
        "overall_internal": {"level_1": 0, "level_2": 0, "level_3": 0},
        "overall_external": {"level_1": 0, "level_2": 0, "level_3": 0},
    }

    for result in results:
        constraint = result.get("constraint_name", "unknown")
        internal_level = result.get("internal_monitor", {}).get("detection", {}).get("level", 1)
        external_level = result.get("external_monitor", {}).get("detection", {}).get("level", 1)

        stats["by_constraint"][constraint]["count"] += 1
        stats["by_constraint"][constraint][f"internal_level_{internal_level}"] += 1
        stats["by_constraint"][constraint][f"external_level_{external_level}"] += 1
        stats["overall_internal"][f"level_{internal_level}"] += 1
        stats["overall_external"][f"level_{external_level}"] += 1

    stats["by_constraint"] = dict(stats["by_constraint"])
    return stats


def print_pilot_report(analysis: dict, stats: dict):
    """Print a formatted report of the pilot analysis."""
    print("\n" + "=" * 70)
    print("PILOT ANALYSIS REPORT")
    print("=" * 70)

    metadata = analysis.get("metadata", {})
    print(f"\nModel: {metadata.get('model_key', 'unknown')} ({metadata.get('model_id', 'unknown')})")
    print(f"Tasks: {metadata.get('task_ids', [])}")
    print(f"Conditions: {metadata.get('conditions', [])}")
    print(f"Total samples: {stats['total_samples']}")

    pilot_analysis = analysis.get("pilot_analysis", {})
    if pilot_analysis.get("behavioral_deltas"):
        print("\n--- Behavioral Deltas (vs Baseline) ---")
        for cond, delta in pilot_analysis["behavioral_deltas"].items():
            direction = "↓" if delta["api_call_delta"] < 0 else "↑"
            print(f"  {cond}: {delta['api_call_delta']:+.1f} API calls ({delta['api_call_delta_pct']:+.1f}%) {direction}")

    print("\n--- Monitor Detection Levels ---")
    print(f"{'Condition':<20} {'Internal (L1/L2/L3)':<25} {'External (L1/L2/L3)':<25}")
    print("-" * 70)

    for constraint, cstats in stats["by_constraint"].items():
        n = cstats["count"]
        int_l1 = cstats["internal_level_1"]
        int_l2 = cstats["internal_level_2"]
        int_l3 = cstats["internal_level_3"]
        ext_l1 = cstats["external_level_1"]
        ext_l2 = cstats["external_level_2"]
        ext_l3 = cstats["external_level_3"]
        print(f"{constraint:<20} {int_l1}/{int_l2}/{int_l3} ({n})               {ext_l1}/{ext_l2}/{ext_l3} ({n})")

    print("\n--- Overall Detection Rates ---")
    total = stats["total_samples"]
    if total > 0:
        int_stats = stats["overall_internal"]
        ext_stats = stats["overall_external"]

        print(f"\nInternal Monitor:")
        print(f"  Level 1 (No detection):      {int_stats['level_1']}/{total} ({100*int_stats['level_1']/total:.0f}%)")
        print(f"  Level 2 (Behavior only):     {int_stats['level_2']}/{total} ({100*int_stats['level_2']/total:.0f}%)")
        print(f"  Level 3 (Full detection):    {int_stats['level_3']}/{total} ({100*int_stats['level_3']/total:.0f}%)")

        print(f"\nExternal Monitor:")
        print(f"  Level 1 (No detection):      {ext_stats['level_1']}/{total} ({100*ext_stats['level_1']/total:.0f}%)")
        print(f"  Level 2 (Behavior only):     {ext_stats['level_2']}/{total} ({100*ext_stats['level_2']/total:.0f}%)")
        print(f"  Level 3 (Full detection):    {ext_stats['level_3']}/{total} ({100*ext_stats['level_3']/total:.0f}%)")

    print("\n--- Key Finding ---")
    constrained_results = [r for r in analysis.get("monitor_results", [])
                          if r.get("constraint_name") not in ["baseline", None]]

    if constrained_results:
        internal_detected = sum(1 for r in constrained_results
                               if r.get("internal_monitor", {}).get("detection", {}).get("level", 1) >= 2)
        external_detected = sum(1 for r in constrained_results
                               if r.get("external_monitor", {}).get("detection", {}).get("level", 1) >= 2)
        internal_full = sum(1 for r in constrained_results
                          if r.get("internal_monitor", {}).get("detection", {}).get("level", 1) == 3)
        external_full = sum(1 for r in constrained_results
                          if r.get("external_monitor", {}).get("detection", {}).get("level", 1) == 3)

        n_constrained = len(constrained_results)
        print(f"For constrained runs (n={n_constrained}):")
        print(f"  Internal monitor detected behavior in {internal_detected}/{n_constrained} ({100*internal_detected/n_constrained:.0f}%)")
        print(f"  Internal monitor detected correct reason in {internal_full}/{n_constrained} ({100*internal_full/n_constrained:.0f}%)")
        print(f"  External monitor detected behavior in {external_detected}/{n_constrained} ({100*external_detected/n_constrained:.0f}%)")
        print(f"  External monitor detected correct reason in {external_full}/{n_constrained} ({100*external_full/n_constrained:.0f}%)")

        if internal_full == 0 and external_full == 0:
            print("\n  ⚠️  CONCERNING: Neither monitor correctly identified the constraint reasoning")
            print("     despite measurable behavioral changes.")


def cmd_analyze_pilot(args):
    """Run monitor analysis on pilot trajectories."""
    if args.pilot_file:
        pilot_file = Path(args.pilot_file)
    elif args.latest:
        pilot_dir = APPWORLD_EXP_DIR / "logs" / "pilot"
        pilot_files = sorted(pilot_dir.glob("pilot_*.json"), reverse=True)
        if not pilot_files:
            print("No pilot files found")
            return
        pilot_file = pilot_files[0]
    else:
        print("Specify --pilot-file or --latest")
        return

    print(f"Analyzing: {pilot_file}")
    analysis = analyze_pilot_file(pilot_file)
    stats = compute_pilot_summary_statistics(analysis.get("monitor_results", []))
    print_pilot_report(analysis, stats)

    if args.save:
        output_file = pilot_file.with_suffix(".analysis.json")
        output = {"analysis": analysis, "statistics": stats}
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved analysis to: {output_file}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AppWorld experiment plotting and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- pilot ---
    p_pilot = subparsers.add_parser("pilot", help="Plot pilot experiment results")
    p_pilot.add_argument("--results-file", type=str, help="Path to results JSON file")
    p_pilot.add_argument("--latest", action="store_true", help="Use the latest results file")
    p_pilot.add_argument("--no-monitor", action="store_true", help="Skip monitor analysis (faster)")

    # --- phase1 ---
    p_phase1 = subparsers.add_parser("phase1", help="Plot phase 1 results")
    p_phase1.add_argument("--results-dir", type=str, default=None, help="Results directory")

    # --- phase2 ---
    p_phase2 = subparsers.add_parser("phase2", help="Plot phase 2 results")
    p_phase2.add_argument("--results-file", type=str, default=None, help="Results JSON file")

    # --- survey ---
    p_survey = subparsers.add_parser("survey", help="Plot task survey results")
    p_survey.add_argument("--results-file", type=str, default=None, help="Results JSON file")

    # --- analyze-pilot ---
    p_analyze = subparsers.add_parser("analyze-pilot", help="Run monitors on pilot trajectories")
    p_analyze.add_argument("--pilot-file", type=str, help="Path to pilot results JSON")
    p_analyze.add_argument("--latest", action="store_true", help="Use the latest pilot file")
    p_analyze.add_argument("--save", action="store_true", help="Save analysis to file")

    args = parser.parse_args(argv)

    if args.command == "pilot":
        cmd_pilot(args)
    elif args.command == "phase1":
        cmd_phase1(args)
    elif args.command == "phase2":
        cmd_phase2(args)
    elif args.command == "survey":
        cmd_survey(args)
    elif args.command == "analyze-pilot":
        cmd_analyze_pilot(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate visualizations for pilot experiment results."""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

APPWORLD_EXP_DIR = Path(__file__).parent


def load_results(results_file: Path) -> dict:
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def plot_api_calls(data: dict, output_file: Path | None = None):
    """Create bar plot of API calls by condition."""
    analysis = data["analysis"]
    by_condition = analysis["by_condition"]

    # Order: baseline, user (neg/pos), model (neg/pos), values (neg/pos)
    condition_order = [
        "baseline",
        "user_negative", "user_positive",
        "model_negative", "model_positive",
        "values_negative", "values_positive",
    ]

    # Filter to only conditions that exist
    condition_order = [c for c in condition_order if c in by_condition]

    # Extract data
    means = [by_condition[c]["avg_api_calls"] for c in condition_order]
    stds = [by_condition[c].get("std_api_calls", 0) for c in condition_order]

    # Colors: gray for baseline, blue shades for negative, red shades for positive
    colors = []
    for c in condition_order:
        if c == "baseline":
            colors.append("#888888")
        elif "negative" in c:
            colors.append("#4a86c7")  # blue
        else:
            colors.append("#c74a4a")  # red

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(condition_order))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1)

    # Baseline reference line
    baseline_mean = by_condition["baseline"]["avg_api_calls"]
    ax.axhline(y=baseline_mean, color="black", linestyle="--", linewidth=2, label=f"Baseline ({baseline_mean:.1f})")

    # Labels
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

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        delta = ""
        if condition_order[i] != "baseline":
            delta_val = analysis["behavioral_deltas"].get(condition_order[i], {}).get("api_call_delta_pct", 0)
            delta = f"\n({delta_val:+.0f}%)"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                f"{mean:.1f}{delta}", ha="center", va="bottom", fontsize=10)

    # Add category separators
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=2.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.5)

    # Category labels at top
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


def run_monitors_and_plot(data: dict, output_file: Path | None = None):
    """Run monitors on all trajectories and plot suspicion score distributions."""
    import importlib.util
    import os
    import sys

    # Setup paths for local imports
    sys.path.insert(0, str(APPWORLD_EXP_DIR.parent))
    os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)

    def load_local_module(name: str):
        module_path = APPWORLD_EXP_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monitor_module = load_local_module("monitor")
    analyze_trajectory_with_monitors = monitor_module.analyze_trajectory_with_monitors

    # Run monitors on all trajectories
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
                    "internal_scores": [],
                    "external_scores": [],
                    "internal_explanations": [],
                    "external_explanations": [],
                    "internal_categories": [],
                    "external_categories": [],
                }

            # Extract suspicion scores
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

    # Order conditions for plotting
    condition_order = [
        "baseline",
        "user_negative", "user_positive",
        "model_negative", "model_positive",
        "values_negative", "values_positive",
    ]
    condition_order = [c for c in condition_order if c in results_by_condition]

    # Create figure with box plots showing suspicion score distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Prepare data for box plots
    internal_data = [results_by_condition[c]["internal_scores"] for c in condition_order]
    external_data = [results_by_condition[c]["external_scores"] for c in condition_order]

    # Display labels
    display_labels = []
    for c in condition_order:
        if c == "baseline":
            display_labels.append("Baseline")
        else:
            parts = c.split("_")
            category = parts[0].capitalize()
            direction = "(-)" if parts[1] == "negative" else "(+)"
            display_labels.append(f"{category}\n{direction}")

    # Colors for each condition
    colors = []
    for c in condition_order:
        if c == "baseline":
            colors.append("#888888")  # Gray
        elif "user" in c:
            colors.append("#4a86c7")  # Blue
        elif "model" in c:
            colors.append("#8a5ac7")  # Purple
        else:  # values
            colors.append("#c74a4a")  # Red

    # ===== INTERNAL MONITOR (ax1) =====
    bp1 = ax1.boxplot(internal_data, positions=range(len(condition_order)), widths=0.6, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
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

    # Add vertical separator after baseline
    ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

    # Add mean labels
    for i, scores in enumerate(internal_data):
        mean_val = np.mean(scores)
        ax1.text(i, -0.12, f"μ={mean_val:.2f}", ha="center", fontsize=9, color="black")

    # ===== EXTERNAL MONITOR (ax2) =====
    bp2 = ax2.boxplot(external_data, positions=range(len(condition_order)), widths=0.6, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
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

    # Add vertical separator after baseline
    ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

    # Add mean labels
    for i, scores in enumerate(external_data):
        mean_val = np.mean(scores)
        ax2.text(i, -0.12, f"μ={mean_val:.2f}", ha="center", fontsize=9, color="black")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

    plt.close(fig)

    # Also print summary statistics
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


def main():
    parser = argparse.ArgumentParser(description="Plot pilot experiment results")
    parser.add_argument("--results-file", type=str, help="Path to results JSON file")
    parser.add_argument("--latest", action="store_true", help="Use the latest results file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output plots")
    parser.add_argument("--no-monitor", action="store_true", help="Skip monitor analysis (faster)")
    args = parser.parse_args()

    # Find results file
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
    data = load_results(results_file)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = APPWORLD_EXP_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: API calls bar chart
    print("\nGenerating API calls plot...")
    plot_api_calls(data, output_dir / "api_calls_by_condition.png")

    # Plot 2: Monitor performance
    if not args.no_monitor:
        print("\nRunning monitors and generating performance plot...")
        print("(This may take a few minutes as it calls GPT-4o for each trajectory)")
        run_monitors_and_plot(data, output_dir / "monitor_performance.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

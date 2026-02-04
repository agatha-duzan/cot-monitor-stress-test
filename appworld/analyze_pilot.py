#!/usr/bin/env python3
"""Analyze pilot experiment results with monitors.

This script:
1. Loads pilot experiment results
2. Runs both monitors on each trajectory
3. Computes detection levels and aggregate statistics
4. Produces a summary report
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
APPWORLD_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)


def load_local_module(name: str):
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor_module = load_local_module("monitor")
analyze_trajectory_with_monitors = monitor_module.analyze_trajectory_with_monitors


def analyze_pilot_file(pilot_file: Path) -> dict:
    """Analyze all trajectories in a pilot file."""
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

        # Get the true constraint (None for baseline)
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


def compute_summary_statistics(results: list[dict]) -> dict:
    """Compute summary statistics across all monitor results."""
    stats = {
        "total_samples": len(results),
        "by_constraint": defaultdict(lambda: {
            "count": 0,
            "internal_level_1": 0,
            "internal_level_2": 0,
            "internal_level_3": 0,
            "external_level_1": 0,
            "external_level_2": 0,
            "external_level_3": 0,
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

    # Convert defaultdict to regular dict
    stats["by_constraint"] = dict(stats["by_constraint"])

    return stats


def print_report(analysis: dict, stats: dict):
    """Print a formatted report of the analysis."""
    print("\n" + "=" * 70)
    print("PILOT ANALYSIS REPORT")
    print("=" * 70)

    metadata = analysis.get("metadata", {})
    print(f"\nModel: {metadata.get('model_key', 'unknown')} ({metadata.get('model_id', 'unknown')})")
    print(f"Tasks: {metadata.get('task_ids', [])}")
    print(f"Conditions: {metadata.get('conditions', [])}")
    print(f"Total samples: {stats['total_samples']}")

    # Behavioral analysis from pilot
    pilot_analysis = analysis.get("pilot_analysis", {})
    if pilot_analysis.get("behavioral_deltas"):
        print("\n--- Behavioral Deltas (vs Baseline) ---")
        for cond, delta in pilot_analysis["behavioral_deltas"].items():
            direction = "↓" if delta["api_call_delta"] < 0 else "↑"
            print(f"  {cond}: {delta['api_call_delta']:+.1f} API calls ({delta['api_call_delta_pct']:+.1f}%) {direction}")

    # Monitor detection results
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

    # Overall summary
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

    # Key finding
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


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot experiment with monitors")
    parser.add_argument("--pilot-file", type=str, help="Path to pilot results JSON")
    parser.add_argument("--latest", action="store_true", help="Use the latest pilot file")
    parser.add_argument("--save", action="store_true", help="Save analysis to file")
    args = parser.parse_args()

    # Find pilot file
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

    # Run analysis
    analysis = analyze_pilot_file(pilot_file)
    stats = compute_summary_statistics(analysis.get("monitor_results", []))

    # Print report
    print_report(analysis, stats)

    # Save if requested
    if args.save:
        output_file = pilot_file.with_suffix(".analysis.json")
        output = {
            "analysis": analysis,
            "statistics": stats,
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved analysis to: {output_file}")


if __name__ == "__main__":
    main()

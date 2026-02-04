#!/usr/bin/env python3
"""Pilot experiment runner for AppWorld CoT monitoring.

This script runs a small pilot to validate:
1. The agent can complete tasks with different constraints
2. Constraints cause measurable behavioral changes
3. Internal reasoning is captured correctly

Usage:
    python appworld/run_pilot.py --task-id 024c982_1 --model kimi --conditions all --samples 5
    python appworld/run_pilot.py --all-simple-tasks --model kimi --conditions all
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
APPWORLD_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)


def load_local_module(name: str):
    """Load a module from the local appworld experiment folder."""
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules
config = load_local_module("config")
agent_module = load_local_module("agent")

MODEL_CONFIGS = config.MODEL_CONFIGS
CONSTRAINTS = config.CONSTRAINTS
ALL_CONDITIONS = config.ALL_CONDITIONS

run_appworld_agent = agent_module.run_appworld_agent
Trajectory = agent_module.Trajectory

# Simple tasks that are more likely to complete in few turns
SIMPLE_TASKS = [
    "024c982_1",  # Request $13 from Stacy on Venmo
    "024c982_2",  # Similar Venmo task variant
    "042a9fc_1",  # Another simple task
]


def get_model_id(model_key: str) -> str:
    """Get full model ID from short key."""
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]["model_id"]
    return model_key


def run_single_experiment(
    task_id: str,
    model_id: str,
    constraint_name: str | None,
    max_turns: int = 10,
    verbose: bool = False,
) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Model: {model_id}")
    print(f"Constraint: {constraint_name or 'baseline'}")
    print(f"{'='*60}")

    try:
        trajectory = run_appworld_agent(
            task_id=task_id,
            model_id=model_id,
            constraint_name=constraint_name,
            max_turns=max_turns,
            enable_thinking=True,
            verbose=verbose,
        )

        result = {
            "task_id": task_id,
            "model_id": model_id,
            "constraint_name": constraint_name or "baseline",
            "constraint_text": CONSTRAINTS.get(constraint_name, {}).get("text") if constraint_name else None,
            "constraint_category": CONSTRAINTS.get(constraint_name, {}).get("category") if constraint_name else None,
            "success": trajectory.task_success,
            "error": trajectory.error,
            "turn_count": trajectory.turn_count,
            "total_api_calls": trajectory.total_api_calls,
            "unique_api_count": len(trajectory.unique_apis),
            "unique_apis": list(trajectory.unique_apis),
            "has_internal_reasoning": any(t.internal_reasoning for t in trajectory.turns),
            "internal_reasoning_length": sum(len(t.internal_reasoning or "") for t in trajectory.turns),
            "external_thought_length": sum(len(t.external_thought or "") for t in trajectory.turns),
            "trajectory": trajectory.to_dict(),
        }

        print(f"\nResult: {'SUCCESS' if trajectory.task_success else 'FAIL'}")
        print(f"Turns: {trajectory.turn_count}, API calls: {trajectory.total_api_calls}")
        print(f"Internal reasoning: {result['internal_reasoning_length']} chars")

        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        return {
            "task_id": task_id,
            "model_id": model_id,
            "constraint_name": constraint_name or "baseline",
            "success": False,
            "error": str(e),
            "turn_count": 0,
            "total_api_calls": 0,
        }


def run_pilot_experiment(
    task_ids: list[str],
    model_key: str,
    conditions: list[str],
    max_turns: int = 10,
    samples: int = 1,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """Run pilot experiment across tasks and conditions.

    Args:
        task_ids: List of task IDs to run
        model_key: Model key from MODEL_CONFIGS
        conditions: List of conditions to test (use 'all' for all)
        max_turns: Max turns per task
        samples: Number of samples per condition (for statistical reliability)
        verbose: Verbose output
        output_dir: Output directory for results
    """
    model_id = get_model_id(model_key)
    results = []

    # Resolve conditions
    if "all" in conditions:
        conditions = ALL_CONDITIONS

    # Ensure baseline is first for comparison
    if "baseline" in conditions:
        conditions = ["baseline"] + [c for c in conditions if c != "baseline"]

    total_runs = len(task_ids) * len(conditions) * samples
    print(f"\n{'#'*60}")
    print(f"# PILOT EXPERIMENT")
    print(f"# Tasks: {len(task_ids)}")
    print(f"# Model: {model_key} ({model_id})")
    print(f"# Conditions: {conditions}")
    print(f"# Samples per condition: {samples}")
    print(f"# Total runs: {total_runs}")
    print(f"{'#'*60}")

    run_count = 0
    for task_id in task_ids:
        for condition in conditions:
            for sample_idx in range(samples):
                run_count += 1
                print(f"\n[Run {run_count}/{total_runs}] Sample {sample_idx + 1}/{samples}")

                constraint_name = None if condition == "baseline" else condition
                result = run_single_experiment(
                    task_id=task_id,
                    model_id=model_id,
                    constraint_name=constraint_name,
                    max_turns=max_turns,
                    verbose=verbose,
                )
                result["sample_index"] = sample_idx
                results.append(result)

    # Analyze results
    analysis = analyze_results(results)

    # Save results
    if output_dir is None:
        output_dir = APPWORLD_EXP_DIR / "logs" / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pilot_{model_key}_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "model_key": model_key,
            "model_id": model_id,
            "task_ids": task_ids,
            "conditions": conditions,
            "samples_per_condition": samples,
            "max_turns": max_turns,
        },
        "analysis": analysis,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# RESULTS SAVED: {output_file}")
    print(f"{'#'*60}")

    # Print analysis summary
    print_analysis(analysis)

    return output


def analyze_results(results: list[dict]) -> dict:
    """Analyze pilot results for behavioral deltas."""
    import math

    analysis = {
        "by_condition": {},
        "behavioral_deltas": {},
    }

    # Group by condition
    by_condition = {}
    for r in results:
        cond = r["constraint_name"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    def compute_std(values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    # Compute stats per condition
    for cond, cond_results in by_condition.items():
        successful = [r for r in cond_results if r["success"]]
        api_calls = [r["total_api_calls"] for r in cond_results]
        turns = [r["turn_count"] for r in cond_results]

        analysis["by_condition"][cond] = {
            "total": len(cond_results),
            "success_count": len(successful),
            "success_rate": len(successful) / len(cond_results) if cond_results else 0,
            "avg_api_calls": sum(api_calls) / len(api_calls) if api_calls else 0,
            "std_api_calls": compute_std(api_calls),
            "min_api_calls": min(api_calls) if api_calls else 0,
            "max_api_calls": max(api_calls) if api_calls else 0,
            "avg_turns": sum(turns) / len(turns) if turns else 0,
            "std_turns": compute_std(turns),
            "avg_internal_reasoning_length": sum(r.get("internal_reasoning_length", 0) for r in cond_results) / len(cond_results) if cond_results else 0,
            "has_reasoning_rate": sum(1 for r in cond_results if r.get("has_internal_reasoning")) / len(cond_results) if cond_results else 0,
        }

    # Compute deltas vs baseline
    baseline_stats = analysis["by_condition"].get("baseline", {})
    baseline_api_calls = baseline_stats.get("avg_api_calls", 0)

    for cond, stats in analysis["by_condition"].items():
        if cond != "baseline" and baseline_api_calls > 0:
            delta = stats["avg_api_calls"] - baseline_api_calls
            delta_pct = (delta / baseline_api_calls) * 100
            analysis["behavioral_deltas"][cond] = {
                "api_call_delta": delta,
                "api_call_delta_pct": delta_pct,
                "expected_direction": CONSTRAINTS.get(cond, {}).get("expected_effect", "unknown"),
            }

    return analysis


def print_analysis(analysis: dict):
    """Print analysis summary."""
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print("\n--- Per-Condition Stats ---")
    for cond, stats in analysis["by_condition"].items():
        print(f"\n{cond} (n={stats['total']}):")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        if stats['total'] > 1:
            print(f"  API calls: {stats['avg_api_calls']:.1f} ± {stats['std_api_calls']:.1f} (range: {stats['min_api_calls']}-{stats['max_api_calls']})")
            print(f"  Turns: {stats['avg_turns']:.1f} ± {stats['std_turns']:.1f}")
        else:
            print(f"  API calls: {stats['avg_api_calls']:.1f}")
            print(f"  Turns: {stats['avg_turns']:.1f}")
        print(f"  Avg internal reasoning: {stats['avg_internal_reasoning_length']:.0f} chars")
        print(f"  Has reasoning: {stats['has_reasoning_rate']:.1%}")

    if analysis["behavioral_deltas"]:
        print("\n--- Behavioral Deltas (vs Baseline) ---")
        for cond, delta in analysis["behavioral_deltas"].items():
            direction = "↓" if delta["api_call_delta"] < 0 else "↑" if delta["api_call_delta"] > 0 else "="
            expected = delta["expected_direction"]
            match = ""
            if expected == "minimize_api_calls" and delta["api_call_delta"] < 0:
                match = " ✓ (as expected)"
            elif expected == "maximize_api_calls" and delta["api_call_delta"] > 0:
                match = " ✓ (as expected)"
            elif delta["api_call_delta"] != 0:
                match = " ✗ (unexpected direction)"

            print(f"\n{cond}:")
            print(f"  API call delta: {delta['api_call_delta']:+.1f} ({delta['api_call_delta_pct']:+.1f}%) {direction}{match}")
            print(f"  Expected effect: {expected}")


def main():
    parser = argparse.ArgumentParser(description="Run AppWorld CoT monitoring pilot")
    parser.add_argument("--task-id", type=str, help="Specific task ID to run")
    parser.add_argument("--all-simple-tasks", action="store_true", help="Run all simple tasks")
    parser.add_argument("--model", type=str, default="kimi",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "user_negative"],
                       help="Conditions to test (use 'all' for all conditions)")
    parser.add_argument("--samples", type=int, default=1,
                       help="Number of samples per condition (for statistical reliability)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per task")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Determine task IDs
    if args.all_simple_tasks:
        task_ids = SIMPLE_TASKS
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = SIMPLE_TASKS[:1]  # Default to first simple task

    run_pilot_experiment(
        task_ids=task_ids,
        model_key=args.model,
        conditions=args.conditions,
        samples=args.samples,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

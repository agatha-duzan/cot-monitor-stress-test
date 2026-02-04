#!/usr/bin/env python3
"""Phase 1 experiment: measure constraint-driven behavioral changes across models.

Tests 6 reasoning models on AppWorld with 12 constraint pairs (positive/negative)
plus baseline. Each config is run 5 times for statistical reliability.

Usage:
    # Validate all models work (1 baseline run each)
    python appworld/run_phase1.py --validate

    # Run full experiment (all models, all constraints, 5 samples)
    python appworld/run_phase1.py --run

    # Run a single model
    python appworld/run_phase1.py --run --models kimi

    # Plot results from saved logs
    python appworld/run_phase1.py --plot
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
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_local_module("config")
agent_module = load_local_module("agent")

MODEL_CONFIGS = config.MODEL_CONFIGS
MODEL_ORDER = config.MODEL_ORDER
CONSTRAINTS = config.CONSTRAINTS
ALL_CONDITIONS = config.ALL_CONDITIONS

run_appworld_agent = agent_module.run_appworld_agent

PHASE1_DIR = APPWORLD_EXP_DIR / "logs" / "phase1"
TASK_ID = "024c982_1"
SAMPLES = 5
MAX_TURNS = 10


def run_single(task_id: str, model_id: str, constraint_name: str | None,
               max_turns: int = MAX_TURNS) -> dict:
    """Run a single experiment and return results dict."""
    try:
        trajectory = run_appworld_agent(
            task_id=task_id,
            model_id=model_id,
            constraint_name=constraint_name,
            max_turns=max_turns,
            enable_thinking=True,
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "constraint_name": constraint_name or "baseline",
            "constraint_text": CONSTRAINTS.get(constraint_name, {}).get("text") if constraint_name else None,
            "constraint_category": CONSTRAINTS.get(constraint_name, {}).get("category") if constraint_name else None,
            "constraint_theme": CONSTRAINTS.get(constraint_name, {}).get("theme") if constraint_name else None,
            "constraint_direction": CONSTRAINTS.get(constraint_name, {}).get("direction") if constraint_name else None,
            "success": trajectory.task_success,
            "error": trajectory.error,
            "turn_count": trajectory.turn_count,
            "total_api_calls": trajectory.total_api_calls,
            "unique_apis": list(trajectory.unique_apis),
            "has_internal_reasoning": any(t.internal_reasoning for t in trajectory.turns),
            "internal_reasoning_length": sum(len(t.internal_reasoning or "") for t in trajectory.turns),
            "trajectory": trajectory.to_dict(),
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "task_id": task_id,
            "model_id": model_id,
            "constraint_name": constraint_name or "baseline",
            "success": False,
            "error": str(e),
            "turn_count": 0,
            "total_api_calls": 0,
            "has_internal_reasoning": False,
            "internal_reasoning_length": 0,
        }


def run_model_experiment(model_key: str, conditions: list[str], samples: int,
                         task_id: str = TASK_ID, max_turns: int = MAX_TURNS) -> dict:
    """Run the full experiment for a single model."""
    model_id = MODEL_CONFIGS[model_key]["model_id"]
    display_name = MODEL_CONFIGS[model_key]["display_name"]

    total_runs = len(conditions) * samples
    print(f"\n{'#' * 60}")
    print(f"# MODEL: {display_name} ({model_key})")
    print(f"# Conditions: {len(conditions)} ({samples} samples each)")
    print(f"# Total runs: {total_runs}")
    print(f"{'#' * 60}")

    results = []
    run_count = 0

    for condition in conditions:
        for sample_idx in range(samples):
            run_count += 1
            constraint_name = None if condition == "baseline" else condition
            label = constraint_name or "baseline"
            print(f"\n[{model_key}] Run {run_count}/{total_runs}: {label} (sample {sample_idx + 1}/{samples})")

            result = run_single(task_id, model_id, constraint_name, max_turns)
            result["sample_index"] = sample_idx

            status = "OK" if not result.get("error") else "ERR"
            reasoning = "yes" if result.get("has_internal_reasoning") else "NO"
            print(f"  -> {status} | turns={result['turn_count']} api_calls={result['total_api_calls']} reasoning={reasoning}")

            results.append(result)

    # Save per-model results
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PHASE1_DIR / f"phase1_{model_key}_{timestamp}.json"

    analysis = analyze_model_results(results)

    output = {
        "metadata": {
            "timestamp": timestamp,
            "model_key": model_key,
            "model_id": model_id,
            "display_name": display_name,
            "task_id": task_id,
            "conditions": conditions,
            "samples_per_condition": samples,
            "max_turns": max_turns,
        },
        "analysis": analysis,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=> Saved: {output_file}")
    print_model_summary(model_key, analysis)

    return output


def analyze_model_results(results: list[dict]) -> dict:
    """Compute per-condition stats and deltas vs baseline."""
    import math

    by_condition = {}
    for r in results:
        cond = r["constraint_name"]
        by_condition.setdefault(cond, []).append(r)

    def std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    stats = {}
    for cond, rs in by_condition.items():
        api_calls = [r["total_api_calls"] for r in rs]
        turns = [r["turn_count"] for r in rs]
        stats[cond] = {
            "n": len(rs),
            "success_rate": sum(1 for r in rs if r["success"]) / len(rs),
            "avg_api_calls": sum(api_calls) / len(api_calls),
            "std_api_calls": std(api_calls),
            "min_api_calls": min(api_calls),
            "max_api_calls": max(api_calls),
            "avg_turns": sum(turns) / len(turns),
            "std_turns": std(turns),
            "avg_reasoning_len": sum(r.get("internal_reasoning_length", 0) for r in rs) / len(rs),
            "reasoning_rate": sum(1 for r in rs if r.get("has_internal_reasoning")) / len(rs),
        }

    # Deltas vs baseline
    baseline_avg = stats.get("baseline", {}).get("avg_api_calls", 0)
    deltas = {}
    if baseline_avg > 0:
        for cond, s in stats.items():
            if cond != "baseline":
                delta = s["avg_api_calls"] - baseline_avg
                deltas[cond] = {
                    "api_call_delta": delta,
                    "api_call_delta_pct": (delta / baseline_avg) * 100,
                    "expected_effect": CONSTRAINTS.get(cond, {}).get("expected_effect", "unknown"),
                }

    return {"by_condition": stats, "behavioral_deltas": deltas}


def print_model_summary(model_key: str, analysis: dict):
    """Print a compact summary for one model."""
    stats = analysis["by_condition"]
    deltas = analysis["behavioral_deltas"]

    baseline = stats.get("baseline", {})
    print(f"\n  Baseline: {baseline.get('avg_api_calls', 0):.1f} ± {baseline.get('std_api_calls', 0):.1f} API calls")
    print(f"  Reasoning capture rate: {baseline.get('reasoning_rate', 0):.0%}")

    if deltas:
        # Group by constraint_id
        correct = sum(
            1 for d in deltas.values()
            if (d["expected_effect"] == "minimize_api_calls" and d["api_call_delta"] < 0)
            or (d["expected_effect"] == "maximize_api_calls" and d["api_call_delta"] > 0)
        )
        print(f"  Constraints in expected direction: {correct}/{len(deltas)}")


def validate_models(models: list[str]):
    """Run 1 baseline per model to check everything works."""
    print("\n" + "=" * 60)
    print("VALIDATION RUN: 1 baseline per model")
    print("=" * 60)

    all_ok = True
    for model_key in models:
        model_id = MODEL_CONFIGS[model_key]["model_id"]
        display = MODEL_CONFIGS[model_key]["display_name"]
        print(f"\n--- {display} ({model_key}) ---")

        result = run_single(TASK_ID, model_id, None, max_turns=MAX_TURNS)

        ok = not result.get("error")
        reasoning = result.get("has_internal_reasoning", False)
        api_calls = result.get("total_api_calls", 0)
        turns = result.get("turn_count", 0)
        reasoning_len = result.get("internal_reasoning_length", 0)

        status = "PASS" if ok and reasoning else "FAIL"
        if not ok:
            status = f"FAIL (error: {result.get('error', 'unknown')[:80]})"
        elif not reasoning:
            status = "WARN (no reasoning captured)"

        print(f"  Status: {status}")
        print(f"  Turns: {turns}, API calls: {api_calls}")
        print(f"  Reasoning: {reasoning_len} chars")

        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("VALIDATION PASSED: All models working")
    else:
        print("VALIDATION FAILED: Some models had errors")
    print("=" * 60)

    return all_ok


def load_phase1_results() -> dict[str, dict]:
    """Load the latest phase1 results for each model."""
    results = {}
    if not PHASE1_DIR.exists():
        return results

    for model_key in MODEL_ORDER:
        files = sorted(PHASE1_DIR.glob(f"phase1_{model_key}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                results[model_key] = json.load(f)
            print(f"Loaded {model_key}: {files[0].name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: constraint behavioral effects across models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true", help="Quick validation (1 baseline per model)")
    group.add_argument("--run", action="store_true", help="Run full experiment")
    group.add_argument("--plot", action="store_true", help="Plot results from saved logs")

    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Samples per condition")
    parser.add_argument("--task-id", type=str, default=TASK_ID, help="Task ID")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help="Max turns per run")
    args = parser.parse_args()

    models = args.models or MODEL_ORDER

    if args.validate:
        validate_models(models)

    elif args.run:
        conditions = ALL_CONDITIONS
        for model_key in models:
            run_model_experiment(
                model_key=model_key,
                conditions=conditions,
                samples=args.samples,
                task_id=args.task_id,
                max_turns=args.max_turns,
            )
        print("\nAll models complete. Run with --plot to visualize.")

    elif args.plot:
        plot_module = load_local_module("plot_phase1")
        all_results = load_phase1_results()
        if not all_results:
            print("No results found in logs/phase1/")
            return
        plot_module.plot_all(all_results)


if __name__ == "__main__":
    main()

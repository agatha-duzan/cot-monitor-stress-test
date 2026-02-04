#!/usr/bin/env python3
"""Focused experiment: values_animal + user_cost on Kimi, n=25, no baseline.

Tests the two strongest constraint pairs on the most susceptible model
with sufficient sample size for statistical power.

Design:
- Model: Kimi K2 (thinking)
- Task: 024c982_1
- Constraints: values_animal (neg/pos), user_cost (neg/pos)
- Samples: 25 per condition
- Total: 4 conditions × 25 = 100 runs
- No baseline (compare neg vs pos directly; existing baseline data available)

Usage:
    # Validate with 1 run per condition
    python appworld/run_focused.py --validate

    # Run full experiment
    python appworld/run_focused.py --run

    # Run with custom sample size
    python appworld/run_focused.py --run --samples 10

    # Analyze saved results
    python appworld/run_focused.py --analyze
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
APPWORLD_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)

VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def load_local_module(name: str):
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_local_module("config")
CONSTRAINTS = config.CONSTRAINTS
MODEL_CONFIGS = config.MODEL_CONFIGS

# Experiment parameters
MODEL_KEY = "kimi"
TASK_ID = "024c982_1"
SAMPLES = 25
MAX_TURNS = 10
MAX_WORKERS = 4  # Moderate parallelism to avoid rate limits
SUBPROCESS_TIMEOUT = 300  # 5 min per run (generous for Kimi)

# The two strongest constraint pairs from prior analysis
CONDITIONS = [
    "user_cost_negative",
    "user_cost_positive",
    "values_animal_negative",
    "values_animal_positive",
]

LOGS_DIR = APPWORLD_EXP_DIR / "logs" / "focused"


def run_single_subprocess(constraint_name: str, sample: int) -> dict:
    """Run a single job via subprocess (avoids SQLite threading issues)."""
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        TASK_ID, MODEL_KEY, str(sample), str(MAX_TURNS), constraint_name,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
            env={**os.environ},
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return {
            "task_id": TASK_ID, "model_key": MODEL_KEY,
            "constraint_name": constraint_name, "sample": sample,
            "success": False,
            "error": f"No result. stderr: {proc.stderr[-500:] if proc.stderr else 'empty'}",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": TASK_ID, "model_key": MODEL_KEY,
            "constraint_name": constraint_name, "sample": sample,
            "success": False, "error": "Timeout",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except Exception as e:
        return {
            "task_id": TASK_ID, "model_key": MODEL_KEY,
            "constraint_name": constraint_name, "sample": sample,
            "success": False, "error": str(e),
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }


def run_experiment(samples: int = SAMPLES, workers: int = MAX_WORKERS):
    """Run the full focused experiment."""
    total_runs = len(CONDITIONS) * samples
    print(f"\n{'#' * 60}")
    print(f"# FOCUSED EXPERIMENT")
    print(f"# Model: {MODEL_KEY} ({MODEL_CONFIGS[MODEL_KEY]['model_id']})")
    print(f"# Task: {TASK_ID}")
    print(f"# Conditions: {CONDITIONS}")
    print(f"# Samples per condition: {samples}")
    print(f"# Total runs: {total_runs}")
    print(f"# Workers: {workers}")
    print(f"{'#' * 60}\n")

    # Build job list
    jobs = []
    for constraint_name in CONDITIONS:
        for sample_idx in range(samples):
            jobs.append((constraint_name, sample_idx))

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(run_single_subprocess, cn, s): (cn, s)
            for cn, s in jobs
        }

        for future in as_completed(future_to_job):
            cn, s = future_to_job[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = "OK" if not result.get("error") else f"ERR({result['error'][:30]})"
                print(f"  [{completed}/{total_runs}] {cn} #{s}: "
                      f"calls={result['total_api_calls']} turns={result['turn_count']} {status}")
            except Exception as e:
                print(f"  [{completed}/{total_runs}] {cn} #{s}: EXCEPTION {e}")
                results.append({
                    "task_id": TASK_ID, "model_key": MODEL_KEY,
                    "constraint_name": cn, "sample": s,
                    "success": False, "error": str(e),
                    "turn_count": 0, "total_api_calls": 0,
                    "has_reasoning": False, "reasoning_len": 0,
                })

    # Save results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = LOGS_DIR / f"focused_{MODEL_KEY}_{timestamp}.json"

    output = {
        "metadata": {
            "experiment": "focused",
            "timestamp": timestamp,
            "model_key": MODEL_KEY,
            "model_id": MODEL_CONFIGS[MODEL_KEY]["model_id"],
            "task_id": TASK_ID,
            "conditions": CONDITIONS,
            "samples_per_condition": samples,
            "max_turns": MAX_TURNS,
            "total_runs": total_runs,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=> Saved: {output_file}")
    analyze_results(results)

    return output_file


def analyze_results(results: list[dict]):
    """Analyze and print results with statistical tests."""
    from collections import defaultdict
    try:
        from scipy import stats as scipy_stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["constraint_name"]].append(r)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    # Per-condition stats
    for cond in CONDITIONS:
        rs = by_condition[cond]
        calls = [r["total_api_calls"] for r in rs]
        errors = sum(1 for r in rs if r.get("error"))
        timeouts = sum(1 for r in rs if r.get("error") == "Timeout")
        n = len(calls)
        mean = sum(calls) / n if n else 0
        std = math.sqrt(sum((x - mean) ** 2 for x in calls) / (n - 1)) if n > 1 else 0
        median = sorted(calls)[n // 2] if n else 0
        print(f"\n  {cond}:")
        print(f"    n={n}  mean={mean:.1f}  median={median}  std={std:.1f}  "
              f"range=[{min(calls)},{max(calls)}]  errors={errors}  timeouts={timeouts}")

    # Paired comparisons (the key test)
    pairs = [
        ("user_cost", "user_cost_negative", "user_cost_positive"),
        ("values_animal", "values_animal_negative", "values_animal_positive"),
    ]

    print(f"\n{'=' * 60}")
    print("PAIRED COMPARISONS (H1: positive > negative)")
    print(f"{'=' * 60}")

    for pair_name, neg_key, pos_key in pairs:
        neg_calls = [r["total_api_calls"] for r in by_condition[neg_key]]
        pos_calls = [r["total_api_calls"] for r in by_condition[pos_key]]

        neg_mean = sum(neg_calls) / len(neg_calls)
        pos_mean = sum(pos_calls) / len(pos_calls)
        diff = pos_mean - neg_mean

        print(f"\n  {pair_name}:")
        print(f"    negative: {neg_mean:.1f}  positive: {pos_mean:.1f}  diff: {diff:+.1f}")

        if has_scipy:
            # Mann-Whitney U (non-parametric, one-sided)
            u_stat, u_p = scipy_stats.mannwhitneyu(pos_calls, neg_calls, alternative='greater')
            print(f"    Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f} {'*' if u_p < 0.05 else ''}")

            # Also run t-test for comparison
            t_stat, t_p_two = scipy_stats.ttest_ind(pos_calls, neg_calls)
            t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2  # one-sided
            print(f"    t-test (one-sided): t={t_stat:.2f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''}")

            # Effect size (Cohen's d)
            pooled_std = math.sqrt(
                (sum((x - neg_mean)**2 for x in neg_calls) +
                 sum((x - pos_mean)**2 for x in pos_calls)) /
                (len(neg_calls) + len(pos_calls) - 2)
            )
            if pooled_std > 0:
                cohens_d = diff / pooled_std
                print(f"    Cohen's d: {cohens_d:.2f}")

    # Clean analysis (excluding error/timeout runs)
    print(f"\n{'=' * 60}")
    print("CLEAN ANALYSIS (excluding error/timeout runs)")
    print(f"{'=' * 60}")

    for pair_name, neg_key, pos_key in pairs:
        neg_calls = [r["total_api_calls"] for r in by_condition[neg_key] if not r.get("error")]
        pos_calls = [r["total_api_calls"] for r in by_condition[pos_key] if not r.get("error")]

        if not neg_calls or not pos_calls:
            print(f"\n  {pair_name}: insufficient clean data (neg={len(neg_calls)}, pos={len(pos_calls)})")
            continue

        neg_mean = sum(neg_calls) / len(neg_calls)
        pos_mean = sum(pos_calls) / len(pos_calls)
        diff = pos_mean - neg_mean

        print(f"\n  {pair_name} (n_neg={len(neg_calls)}, n_pos={len(pos_calls)}):")
        print(f"    negative: {neg_mean:.1f}  positive: {pos_mean:.1f}  diff: {diff:+.1f}")

        if has_scipy and len(neg_calls) >= 2 and len(pos_calls) >= 2:
            u_stat, u_p = scipy_stats.mannwhitneyu(pos_calls, neg_calls, alternative='greater')
            print(f"    Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f} {'*' if u_p < 0.05 else ''}")


def validate():
    """Run 1 sample per condition to check everything works."""
    print("\n" + "=" * 60)
    print("VALIDATION: 1 run per condition")
    print("=" * 60)

    for cond in CONDITIONS:
        print(f"\n--- {cond} ---")
        result = run_single_subprocess(cond, 0)
        status = "OK" if not result.get("error") else f"ERR: {result['error'][:80]}"
        print(f"  {status} | calls={result['total_api_calls']} turns={result['turn_count']} "
              f"reasoning={result.get('has_reasoning', False)}")

    print("\nValidation complete. Run with --run to start the full experiment.")


def analyze_from_file():
    """Load and analyze the latest results file."""
    if not LOGS_DIR.exists():
        print("No results directory found.")
        return

    files = sorted(LOGS_DIR.glob("focused_*.json"), reverse=True)
    if not files:
        print("No result files found.")
        return

    latest = files[0]
    print(f"Analyzing: {latest.name}")

    with open(latest) as f:
        data = json.load(f)

    print(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
    analyze_results(data["results"])


def main():
    parser = argparse.ArgumentParser(description="Focused constraint experiment (Kimi, 2 pairs, n=25)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true", help="Quick validation (1 run per condition)")
    group.add_argument("--run", action="store_true", help="Run full experiment")
    group.add_argument("--analyze", action="store_true", help="Analyze latest saved results")

    parser.add_argument("--samples", type=int, default=SAMPLES, help="Samples per condition (default: 25)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    if args.validate:
        validate()
    elif args.run:
        run_experiment(samples=args.samples, workers=args.workers)
    elif args.analyze:
        analyze_from_file()


if __name__ == "__main__":
    main()

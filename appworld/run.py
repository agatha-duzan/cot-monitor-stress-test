#!/usr/bin/env python3
"""Unified experiment runner for AppWorld CoT monitoring.

Subcommands:
    pilot   - Small pilot to validate agent + constraints work
    phase1  - Constraint behavioral effects across all models
    phase2  - Multi-task constraint experiment on opus
    focused - High-n focused experiment on strongest constraint pairs
    survey  - Survey tasks to find solvable ones

Usage:
    python appworld/run.py pilot --task-id 024c982_1 --model kimi --conditions all --samples 5
    python appworld/run.py phase1 --validate
    python appworld/run.py phase1 --run --models kimi
    python appworld/run.py phase2 --workers 6
    python appworld/run.py focused --run --samples 25
    python appworld/run.py survey --workers 6
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from utils import (
    APPWORLD_EXP_DIR,
    compute_condition_stats,
    load_local_module,
    run_parallel_jobs,
    run_single_direct,
    run_single_subprocess,
    save_results,
    timestamp_str,
)

config = load_local_module("config")
MODEL_CONFIGS = config.MODEL_CONFIGS
MODEL_ORDER = config.MODEL_ORDER
CONSTRAINTS = config.CONSTRAINTS
ALL_CONDITIONS = config.ALL_CONDITIONS

# ──────────────────────────────────────────────────────────────────────
# PILOT
# ──────────────────────────────────────────────────────────────────────

SIMPLE_TASKS = [
    "024c982_1",
    "024c982_2",
    "042a9fc_1",
]


def print_pilot_analysis(analysis: dict):
    """Print analysis summary for pilot."""
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


def cmd_pilot(args):
    """Run pilot experiment."""
    model_key = args.model
    model_id = MODEL_CONFIGS[model_key]["model_id"]

    # Determine task IDs
    if args.all_simple_tasks:
        task_ids = SIMPLE_TASKS
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = SIMPLE_TASKS[:1]

    # Resolve conditions
    conditions = list(args.conditions)
    if "all" in conditions:
        conditions = ALL_CONDITIONS
    if "baseline" in conditions:
        conditions = ["baseline"] + [c for c in conditions if c != "baseline"]

    total_runs = len(task_ids) * len(conditions) * args.samples
    print(f"\n{'#'*60}")
    print(f"# PILOT EXPERIMENT")
    print(f"# Tasks: {len(task_ids)}")
    print(f"# Model: {model_key} ({model_id})")
    print(f"# Conditions: {conditions}")
    print(f"# Samples per condition: {args.samples}")
    print(f"# Total runs: {total_runs}")
    print(f"{'#'*60}")

    results = []
    run_count = 0
    for task_id in task_ids:
        for condition in conditions:
            for sample_idx in range(args.samples):
                run_count += 1
                print(f"\n[Run {run_count}/{total_runs}] Sample {sample_idx + 1}/{args.samples}")

                constraint_name = None if condition == "baseline" else condition
                result = run_single_direct(
                    task_id=task_id,
                    model_id=model_id,
                    constraint_name=constraint_name,
                    max_turns=args.max_turns,
                    verbose=args.verbose,
                    constraints_dict=CONSTRAINTS,
                )
                result["sample_index"] = sample_idx
                results.append(result)

                print(f"  Result: {'SUCCESS' if result.get('success') else 'FAIL'}")
                print(f"  Turns: {result['turn_count']}, API calls: {result['total_api_calls']}")

    analysis = compute_condition_stats(results, CONSTRAINTS)

    ts = timestamp_str()
    output_dir = APPWORLD_EXP_DIR / "logs" / "pilot"
    output = {
        "metadata": {
            "timestamp": ts,
            "model_key": model_key,
            "model_id": model_id,
            "task_ids": task_ids,
            "conditions": conditions,
            "samples_per_condition": args.samples,
            "max_turns": args.max_turns,
        },
        "analysis": analysis,
        "results": results,
    }

    output_file = output_dir / f"pilot_{model_key}_{ts}.json"
    save_results(output, output_file)
    print_pilot_analysis(analysis)


# ──────────────────────────────────────────────────────────────────────
# PHASE 1
# ──────────────────────────────────────────────────────────────────────

PHASE1_DIR = APPWORLD_EXP_DIR / "logs" / "phase1"
PHASE1_TASK_ID = "024c982_1"
PHASE1_SAMPLES = 5
PHASE1_MAX_TURNS = 10


def print_phase1_model_summary(model_key: str, analysis: dict):
    """Print a compact summary for one model."""
    stats = analysis["by_condition"]
    deltas = analysis["behavioral_deltas"]

    baseline = stats.get("baseline", {})
    print(f"\n  Baseline: {baseline.get('avg_api_calls', 0):.1f} ± {baseline.get('std_api_calls', 0):.1f} API calls")
    print(f"  Reasoning capture rate: {baseline.get('reasoning_rate', 0):.0%}")

    if deltas:
        correct = sum(
            1 for d in deltas.values()
            if (d["expected_effect"] == "minimize_api_calls" and d["api_call_delta"] < 0)
            or (d["expected_effect"] == "maximize_api_calls" and d["api_call_delta"] > 0)
        )
        print(f"  Constraints in expected direction: {correct}/{len(deltas)}")


def run_phase1_model(model_key: str, conditions: list[str], samples: int,
                     task_id: str, max_turns: int) -> dict:
    """Run the full phase1 experiment for a single model."""
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

            result = run_single_direct(
                task_id=task_id,
                model_id=model_id,
                constraint_name=constraint_name,
                max_turns=max_turns,
                constraints_dict=CONSTRAINTS,
            )
            result["sample_index"] = sample_idx

            status = "OK" if not result.get("error") else "ERR"
            reasoning = "yes" if result.get("has_internal_reasoning") else "NO"
            print(f"  -> {status} | turns={result['turn_count']} api_calls={result['total_api_calls']} reasoning={reasoning}")

            results.append(result)

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp_str()
    output_file = PHASE1_DIR / f"phase1_{model_key}_{ts}.json"

    analysis = compute_condition_stats(results, CONSTRAINTS)

    output = {
        "metadata": {
            "timestamp": ts,
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

    save_results(output, output_file)
    print_phase1_model_summary(model_key, analysis)
    return output


def validate_phase1_models(models: list[str], task_id: str, max_turns: int):
    """Run 1 baseline per model to check everything works."""
    print("\n" + "=" * 60)
    print("VALIDATION RUN: 1 baseline per model")
    print("=" * 60)

    all_ok = True
    for model_key in models:
        model_id = MODEL_CONFIGS[model_key]["model_id"]
        display = MODEL_CONFIGS[model_key]["display_name"]
        print(f"\n--- {display} ({model_key}) ---")

        result = run_single_direct(
            task_id=task_id,
            model_id=model_id,
            constraint_name=None,
            max_turns=max_turns,
            constraints_dict=CONSTRAINTS,
        )

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


def cmd_phase1(args):
    """Run phase1 experiment."""
    models = args.models or MODEL_ORDER
    task_id = args.task_id or PHASE1_TASK_ID
    max_turns = args.max_turns or PHASE1_MAX_TURNS
    samples = args.samples or PHASE1_SAMPLES

    if args.validate:
        validate_phase1_models(models, task_id, max_turns)
    elif args.run:
        conditions = ALL_CONDITIONS
        for model_key in models:
            run_phase1_model(
                model_key=model_key,
                conditions=conditions,
                samples=samples,
                task_id=task_id,
                max_turns=max_turns,
            )
        print("\nAll models complete. Run 'python appworld/plot.py phase1' to visualize.")
    elif args.plot:
        # Import plot module and delegate
        plot_module = load_local_module("plot")
        # Build sys.argv for phase1 subcommand
        plot_args = ["plot.py", "phase1"]
        plot_module.main(plot_args)


# ──────────────────────────────────────────────────────────────────────
# PHASE 2
# ──────────────────────────────────────────────────────────────────────

PHASE2_TASKS = ["afc4005_1", "425a494_1", "7847649_1", "dac78d9_1", "1150ed6_1"]
PHASE2_MODEL = "opus"
PHASE2_SAMPLES = 5
PHASE2_MAX_TURNS = 10
PHASE2_MAX_WORKERS = 6

import numpy as np


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


def print_phase2_summary(results, baseline):
    constraint_names = sorted(CONSTRAINTS.keys())
    print(f"\n{'='*80}")
    print("PHASE 2 SUMMARY")
    print(f"{'='*80}")

    for task_id in PHASE2_TASKS:
        bl_runs = baseline.get(task_id, [])
        bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 0
        bl_std = np.std([r["total_api_calls"] for r in bl_runs]) if bl_runs else 0
        bl_sr = sum(1 for r in bl_runs if r["success"]) / max(len(bl_runs), 1)
        print(f"\n--- {task_id} (baseline: {bl_api:.1f}±{bl_std:.1f} API calls, {bl_sr:.0%} success) ---")
        print(f"  {'Constraint':<30} {'Dir':<5} {'Success':<8} {'AvgAPI':<8} {'Delta%':<8}")
        print(f"  {'-'*60}")

        task_results = [r for r in results if r["task_id"] == task_id]
        by_constraint = defaultdict(list)
        for r in task_results:
            by_constraint[r["constraint_name"]].append(r)

        for cname in constraint_names:
            runs = by_constraint.get(cname, [])
            if not runs:
                continue
            sr = sum(1 for r in runs if r["success"]) / len(runs)
            avg_api = np.mean([r["total_api_calls"] for r in runs])
            delta = ((avg_api - bl_api) / bl_api * 100) if bl_api > 0 else 0
            direction = CONSTRAINTS[cname]["direction"][:3]
            print(f"  {cname:<30} {direction:<5} {sr:>5.0%}   {avg_api:>6.1f}  {delta:>+6.1f}%")


def cmd_phase2(args):
    """Run phase2 experiment."""
    constraint_names = sorted(CONSTRAINTS.keys())
    max_workers = args.workers or PHASE2_MAX_WORKERS

    output_dir = APPWORLD_EXP_DIR / "logs" / "phase2"

    jobs = []
    for task_id in PHASE2_TASKS:
        for cname in constraint_names:
            for sample in range(PHASE2_SAMPLES):
                jobs.append((task_id, PHASE2_MODEL, sample, PHASE2_MAX_TURNS, cname))

    total = len(jobs)
    print(f"Running {total} jobs ({len(PHASE2_TASKS)} tasks × {len(constraint_names)} constraints × {PHASE2_SAMPLES} samples)")
    print(f"Model: {PHASE2_MODEL} | Max turns: {PHASE2_MAX_TURNS} | Workers: {max_workers}\n", flush=True)

    def job_fn(task_id, model_key, sample, max_turns, constraint_name):
        return run_single_subprocess(task_id, model_key, sample, max_turns, constraint_name)

    def label_fn(job, result):
        task_id, _, sample, _, cname = job
        status = "PASS" if result.get("success") else "FAIL"
        return f"{task_id} | {cname} | s{sample+1} -> {status} | api={result.get('total_api_calls', 0)}"

    results = run_parallel_jobs(jobs, job_fn, max_workers=max_workers, label_fn=label_fn)

    ts = timestamp_str()
    out_file = output_dir / f"phase2_{ts}.json"
    save_results({
        "results": results, "tasks": PHASE2_TASKS, "model": PHASE2_MODEL,
        "constraints": constraint_names, "samples": PHASE2_SAMPLES,
    }, out_file)

    baseline = load_phase2_baseline()
    print_phase2_summary(results, baseline)

    # Auto-generate plots
    plot_module = load_local_module("plot")
    plot_phase2_cmd_args = argparse.Namespace(results_file=str(out_file))
    plot_module.cmd_phase2(plot_phase2_cmd_args)


# ──────────────────────────────────────────────────────────────────────
# FOCUSED
# ──────────────────────────────────────────────────────────────────────

FOCUSED_MODEL_KEY = "kimi"
FOCUSED_TASK_ID = "024c982_1"
FOCUSED_SAMPLES = 25
FOCUSED_MAX_TURNS = 10
FOCUSED_MAX_WORKERS = 4
FOCUSED_SUBPROCESS_TIMEOUT = 300

FOCUSED_CONDITIONS = [
    "user_cost_negative",
    "user_cost_positive",
    "values_animal_negative",
    "values_animal_positive",
]

FOCUSED_LOGS_DIR = APPWORLD_EXP_DIR / "logs" / "focused"


def analyze_focused_results(results: list[dict]):
    """Analyze and print results with statistical tests."""
    try:
        from scipy import stats as scipy_stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["constraint_name"]].append(r)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    for cond in FOCUSED_CONDITIONS:
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
            u_stat, u_p = scipy_stats.mannwhitneyu(pos_calls, neg_calls, alternative='greater')
            print(f"    Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f} {'*' if u_p < 0.05 else ''}")

            t_stat, t_p_two = scipy_stats.ttest_ind(pos_calls, neg_calls)
            t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
            print(f"    t-test (one-sided): t={t_stat:.2f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''}")

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


def cmd_focused(args):
    """Run focused experiment."""
    samples = args.samples or FOCUSED_SAMPLES
    workers = args.workers or FOCUSED_MAX_WORKERS

    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION: 1 run per condition")
        print("=" * 60)

        for cond in FOCUSED_CONDITIONS:
            print(f"\n--- {cond} ---")
            result = run_single_subprocess(
                FOCUSED_TASK_ID, FOCUSED_MODEL_KEY, 0,
                FOCUSED_MAX_TURNS, cond, FOCUSED_SUBPROCESS_TIMEOUT,
            )
            status = "OK" if not result.get("error") else f"ERR: {result['error'][:80]}"
            print(f"  {status} | calls={result['total_api_calls']} turns={result['turn_count']} "
                  f"reasoning={result.get('has_reasoning', False)}")
        print("\nValidation complete. Run with --run to start the full experiment.")

    elif args.run:
        total_runs = len(FOCUSED_CONDITIONS) * samples
        print(f"\n{'#' * 60}")
        print(f"# FOCUSED EXPERIMENT")
        print(f"# Model: {FOCUSED_MODEL_KEY} ({MODEL_CONFIGS[FOCUSED_MODEL_KEY]['model_id']})")
        print(f"# Task: {FOCUSED_TASK_ID}")
        print(f"# Conditions: {FOCUSED_CONDITIONS}")
        print(f"# Samples per condition: {samples}")
        print(f"# Total runs: {total_runs}")
        print(f"# Workers: {workers}")
        print(f"{'#' * 60}\n")

        jobs = []
        for cn in FOCUSED_CONDITIONS:
            for s in range(samples):
                jobs.append((FOCUSED_TASK_ID, FOCUSED_MODEL_KEY, s,
                             FOCUSED_MAX_TURNS, cn, FOCUSED_SUBPROCESS_TIMEOUT))

        def job_fn(task_id, model_key, sample, max_turns, constraint_name, timeout):
            return run_single_subprocess(task_id, model_key, sample, max_turns, constraint_name, timeout)

        def label_fn(job, result):
            _, _, s, _, cn, _ = job
            status = "OK" if not result.get("error") else f"ERR({result.get('error', '')[:30]})"
            return f"{cn} #{s}: calls={result.get('total_api_calls', 0)} turns={result.get('turn_count', 0)} {status}"

        results = run_parallel_jobs(jobs, job_fn, max_workers=workers, label_fn=label_fn)

        ts = timestamp_str()
        output_file = FOCUSED_LOGS_DIR / f"focused_{FOCUSED_MODEL_KEY}_{ts}.json"
        save_results({
            "metadata": {
                "experiment": "focused",
                "timestamp": ts,
                "model_key": FOCUSED_MODEL_KEY,
                "model_id": MODEL_CONFIGS[FOCUSED_MODEL_KEY]["model_id"],
                "task_id": FOCUSED_TASK_ID,
                "conditions": FOCUSED_CONDITIONS,
                "samples_per_condition": samples,
                "max_turns": FOCUSED_MAX_TURNS,
                "total_runs": total_runs,
            },
            "results": results,
        }, output_file)

        analyze_focused_results(results)

    elif args.analyze:
        if not FOCUSED_LOGS_DIR.exists():
            print("No results directory found.")
            return
        files = sorted(FOCUSED_LOGS_DIR.glob("focused_*.json"), reverse=True)
        if not files:
            print("No result files found.")
            return
        latest = files[0]
        print(f"Analyzing: {latest.name}")
        with open(latest) as f:
            data = json.load(f)
        print(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
        analyze_focused_results(data["results"])


# ──────────────────────────────────────────────────────────────────────
# SURVEY
# ──────────────────────────────────────────────────────────────────────

SURVEY_TASKS = [
    "21abae1_1", "29a7b7e_1", "31dc501_1", "425a494_1", "552869a_1",
    "59fae45_1", "5a83b05_1", "7847649_1", "a30375d_1", "afc4005_1",
    "dac78d9_1", "f3f60f0_1", "fd1f8fa_1",
]
SURVEY_MODELS = ["opus", "kimi"]
SURVEY_SAMPLES = 10
SURVEY_MAX_TURNS = 10
SURVEY_MAX_WORKERS = 6


def print_survey_summary(results: list[dict]):
    print(f"\n{'='*80}")
    print("TASK SURVEY SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Task':<14} {'Model':<8} {'Success':<10} {'Avg API':<10} {'Std API':<10} {'Avg Turns':<10}")
    print("-" * 62)

    task_model_stats = {}
    for r in results:
        key = (r["task_id"], r["model_key"])
        task_model_stats.setdefault(key, []).append(r)

    task_stats = {}
    for (task_id, model_key), runs in sorted(task_model_stats.items()):
        sr = sum(1 for r in runs if r["success"]) / len(runs)
        api_calls = [r["total_api_calls"] for r in runs]
        avg_api = np.mean(api_calls)
        std_api = np.std(api_calls)
        avg_turns = np.mean([r["turn_count"] for r in runs])
        print(f"{task_id:<14} {model_key:<8} {sr:>5.0%}      {avg_api:>5.1f}     {std_api:>5.1f}     {avg_turns:>5.1f}")

        task_stats.setdefault(task_id, {})[model_key] = {
            "success_rate": sr, "avg_api": avg_api, "std_api": std_api, "avg_turns": avg_turns,
        }

    print(f"\n{'='*50}")
    print("OVERALL PER MODEL")
    print(f"{'='*50}")
    for model_key in SURVEY_MODELS:
        model_runs = [r for r in results if r["model_key"] == model_key]
        sr = sum(1 for r in model_runs if r["success"]) / len(model_runs)
        avg_api = np.mean([r["total_api_calls"] for r in model_runs])
        print(f"{model_key:<8} success={sr:.0%}  avg_api={avg_api:.1f}")

    print(f"\n{'='*50}")
    print("GOOD TASKS (>=50% success on both models)")
    print(f"{'='*50}")
    for task_id, models_data in sorted(task_stats.items()):
        good_count = sum(1 for m in models_data.values() if m["success_rate"] >= 0.5)
        if good_count >= 2:
            parts = [f"{mk}: {md['success_rate']:.0%} ({md['avg_api']:.0f}±{md['std_api']:.0f} calls)"
                     for mk, md in models_data.items()]
            print(f"  {task_id}: {', '.join(parts)}")


def cmd_survey(args):
    """Run task survey."""
    max_workers = args.workers or SURVEY_MAX_WORKERS
    output_dir = APPWORLD_EXP_DIR / "logs" / "easy_test"

    jobs = []
    for task_id in SURVEY_TASKS:
        for model_key in SURVEY_MODELS:
            for sample in range(SURVEY_SAMPLES):
                jobs.append((task_id, model_key, sample, SURVEY_MAX_TURNS))

    total = len(jobs)
    print(f"Running {total} jobs ({len(SURVEY_TASKS)} tasks × {len(SURVEY_MODELS)} models × {SURVEY_SAMPLES} samples)")
    print(f"Parallelism: {max_workers} subprocess workers\n", flush=True)

    def job_fn(task_id, model_key, sample, max_turns):
        return run_single_subprocess(task_id, model_key, sample, max_turns)

    def label_fn(job, result):
        task_id, model_key, sample, _ = job
        status = "PASS" if result.get("success") else "FAIL"
        return f"{model_key} | {task_id} | s{sample+1} -> {status} | api={result.get('total_api_calls', 0)}"

    results = run_parallel_jobs(jobs, job_fn, max_workers=max_workers, label_fn=label_fn)

    ts = timestamp_str()
    out_file = output_dir / f"easy_test_{ts}.json"
    save_results({
        "results": results, "tasks": SURVEY_TASKS,
        "models": SURVEY_MODELS, "samples": SURVEY_SAMPLES,
    }, out_file)

    # Merge with previous results
    all_results = list(results)
    prior_file = output_dir / "easy_test_20260130_124621.json"
    if prior_file.exists():
        with open(prior_file) as f:
            prior = json.load(f)
        completed_tasks = set(SURVEY_TASKS)
        for r in prior["results"]:
            if r["task_id"] not in completed_tasks and (r["total_api_calls"] > 0 or r["turn_count"] > 0):
                all_results.append(r)
        print(f"Merged {len(all_results) - len(results)} results from prior run")

    print_survey_summary(all_results)

    # Auto-generate plots
    plot_module = load_local_module("plot")
    plot_module.plot_survey_results(all_results)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AppWorld experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- pilot ---
    p_pilot = subparsers.add_parser("pilot", help="Run pilot experiment")
    p_pilot.add_argument("--task-id", type=str, help="Specific task ID to run")
    p_pilot.add_argument("--all-simple-tasks", action="store_true", help="Run all simple tasks")
    p_pilot.add_argument("--model", type=str, default="kimi",
                         choices=list(MODEL_CONFIGS.keys()), help="Model to use")
    p_pilot.add_argument("--conditions", nargs="+", default=["baseline", "user_negative"],
                         help="Conditions to test (use 'all' for all conditions)")
    p_pilot.add_argument("--samples", type=int, default=1, help="Samples per condition")
    p_pilot.add_argument("--max-turns", type=int, default=10, help="Max turns per task")
    p_pilot.add_argument("--verbose", action="store_true", help="Verbose output")

    # --- phase1 ---
    p_phase1 = subparsers.add_parser("phase1", help="Phase 1: constraint effects across models")
    g1 = p_phase1.add_mutually_exclusive_group(required=True)
    g1.add_argument("--validate", action="store_true", help="Quick validation (1 baseline per model)")
    g1.add_argument("--run", action="store_true", help="Run full experiment")
    g1.add_argument("--plot", action="store_true", help="Plot results from saved logs")
    p_phase1.add_argument("--models", nargs="+", default=None,
                          choices=list(MODEL_CONFIGS.keys()), help="Models to run (default: all)")
    p_phase1.add_argument("--samples", type=int, default=None, help="Samples per condition")
    p_phase1.add_argument("--task-id", type=str, default=None, help="Task ID")
    p_phase1.add_argument("--max-turns", type=int, default=None, help="Max turns per run")

    # --- phase2 ---
    p_phase2 = subparsers.add_parser("phase2", help="Phase 2: multi-task constraint experiment")
    p_phase2.add_argument("--workers", type=int, default=None, help="Number of parallel workers")

    # --- focused ---
    p_focused = subparsers.add_parser("focused", help="Focused experiment on strongest constraint pairs")
    g2 = p_focused.add_mutually_exclusive_group(required=True)
    g2.add_argument("--validate", action="store_true", help="Quick validation (1 run per condition)")
    g2.add_argument("--run", action="store_true", help="Run full experiment")
    g2.add_argument("--analyze", action="store_true", help="Analyze latest saved results")
    p_focused.add_argument("--samples", type=int, default=None, help="Samples per condition")
    p_focused.add_argument("--workers", type=int, default=None, help="Parallel workers")

    # --- survey ---
    p_survey = subparsers.add_parser("survey", help="Survey tasks to find solvable ones")
    p_survey.add_argument("--workers", type=int, default=None, help="Number of parallel workers")

    args = parser.parse_args(argv)

    if args.command == "pilot":
        cmd_pilot(args)
    elif args.command == "phase1":
        cmd_phase1(args)
    elif args.command == "phase2":
        cmd_phase2(args)
    elif args.command == "focused":
        cmd_focused(args)
    elif args.command == "survey":
        cmd_survey(args)


if __name__ == "__main__":
    main()

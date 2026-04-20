#!/usr/bin/env python3
"""Baseline survey: run a model on AppWorld tasks to measure API call variance.

Tracks token costs and stops at budget limit. Checkpoints results incrementally.

Usage:
    # Sonnet on all test_normal tasks
    python appworld/run_baseline_survey.py --model sonnet --samples 5 --cost-limit 280

    # Opus on specific tasks
    python appworld/run_baseline_survey.py --model opus --tasks-file appworld/opus_survey_tasks.txt --samples 10 --cost-limit 250

    # Resume interrupted run
    python appworld/run_baseline_survey.py --model opus --resume

    # Just print report
    python appworld/run_baseline_survey.py --model opus --report-only
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

APPWORLD_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APPWORLD_EXP_DIR.parent
# Use dedicated appworld venv (pydantic v1) if available, else main venv
_APPWORLD_VENV = PROJECT_ROOT / ".venv_appworld" / "bin" / "python"
VENV_PYTHON = str(_APPWORLD_VENV if _APPWORLD_VENV.exists() else PROJECT_ROOT / ".venv" / "bin" / "python")
LOGS_DIR = APPWORLD_EXP_DIR / "logs" / "baseline_survey"

# Pricing per token (input, output, cache_write, cache_read)
MODEL_PRICING = {
    "sonnet": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
        "cache_read": 0.30 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
    },
    "opus": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
        "cache_read": 1.50 / 1_000_000,
        "cache_write": 18.75 / 1_000_000,
    },
}

MODEL_IDS = {
    "sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "opus": "anthropic/claude-opus-4-5-20251101",
}


def compute_cost(usage: dict | None, model_key: str = "sonnet") -> float:
    """Compute dollar cost from token usage dict."""
    if not usage:
        return 0.0
    pricing = MODEL_PRICING[model_key]
    return (
        usage.get("input_tokens", 0) * pricing["input"]
        + usage.get("output_tokens", 0) * pricing["output"]
        + usage.get("cache_read_input_tokens", 0) * pricing["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * pricing["cache_write"]
    )


def load_task_ids(tasks_file: str | None = None) -> list[str]:
    """Load task IDs from file or default to test_normal."""
    if tasks_file:
        return Path(tasks_file).read_text().strip().split("\n")
    task_file = APPWORLD_EXP_DIR / "data" / "datasets" / "test_normal.txt"
    return task_file.read_text().strip().split("\n")


def run_single(task_id: str, model_key: str, sample: int, max_turns: int, timeout: int = 300) -> dict:
    """Run a single job via subprocess."""
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        task_id, model_key, str(sample), str(max_turns),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ},
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return {
            "task_id": task_id, "model_key": model_key,
            "constraint_name": None, "sample": sample,
            "success": False,
            "error": f"No result. stderr: {proc.stderr[-500:] if proc.stderr else 'empty'}",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
            "usage": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id, "model_key": model_key,
            "constraint_name": None, "sample": sample,
            "success": False, "error": f"Timeout ({timeout}s)",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
            "usage": None,
        }
    except Exception as e:
        return {
            "task_id": task_id, "model_key": model_key,
            "constraint_name": None, "sample": sample,
            "success": False, "error": str(e),
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
            "usage": None,
        }


def checkpoint_path(model_key: str) -> Path:
    return LOGS_DIR / f"checkpoint_{model_key}.json"


def load_checkpoint(model_key: str) -> tuple[list[dict], set[str], float]:
    """Load checkpoint if exists. Returns (results, completed_task_sample_keys, total_cost)."""
    cp = checkpoint_path(model_key)
    if not cp.exists():
        return [], set(), 0.0
    with open(cp) as f:
        data = json.load(f)
    results = data.get("results", [])
    total_cost = data.get("total_cost", 0.0)
    completed = {f"{r['task_id']}_{r['sample']}" for r in results}
    return results, completed, total_cost


def save_checkpoint(results: list[dict], total_cost: float, metadata: dict, model_key: str):
    """Save checkpoint."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata,
        "total_cost": total_cost,
        "results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(checkpoint_path(model_key), "w") as f:
        json.dump(data, f, indent=2)


def save_final_results(results: list[dict], total_cost: float, metadata: dict, model_key: str):
    """Save final results."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = LOGS_DIR / f"baseline_survey_{model_key}_{ts}.json"
    data = {
        "metadata": {**metadata, "end_time": datetime.now().isoformat()},
        "total_cost": total_cost,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n=> Saved final results: {output_file}")
    return output_file


def print_report(results: list[dict], total_cost: float, model_key: str):
    """Print summary report of baseline survey."""
    import statistics

    model_display = {"sonnet": "Sonnet 4.5", "opus": "Opus 4.5"}.get(model_key, model_key)

    # Group by task
    tasks = defaultdict(list)
    for r in results:
        if r.get("error") is None and r.get("total_api_calls", 0) > 0:
            tasks[r["task_id"]].append(r)

    error_count = sum(1 for r in results if r.get("error"))
    tasks_with_data = len(tasks)

    print(f"\n{'='*70}")
    print(f"BASELINE SURVEY RESULTS — {model_display}")
    print(f"{'='*70}")
    print(f"Total runs: {len(results)} | Errors: {error_count} | Tasks with data: {tasks_with_data}")
    print(f"Total cost: ${total_cost:.2f}")
    if results:
        print(f"Avg cost/run: ${total_cost / len(results):.3f}")
    print()

    # Per-task stats sorted by CV
    task_stats = []
    for tid in sorted(tasks.keys()):
        calls = [r["total_api_calls"] for r in tasks[tid]]
        n = len(calls)
        if n < 2:
            continue
        mean = statistics.mean(calls)
        std = statistics.stdev(calls)
        cv = std / mean if mean > 0 else 999
        success_rate = sum(1 for r in tasks[tid] if r.get("success")) / n
        task_stats.append({
            "task_id": tid, "n": n, "mean": mean, "std": std, "cv": cv,
            "min": min(calls), "max": max(calls), "success_rate": success_rate,
        })

    # Sort by CV
    task_stats.sort(key=lambda x: x["cv"])

    print(f"{'Task':<14} {'n':>3} {'mean':>6} {'std':>5} {'CV':>5} {'min':>4} {'max':>4} {'pass%':>5}")
    print("-" * 55)
    for s in task_stats:
        print(f"{s['task_id']:<14} {s['n']:>3} {s['mean']:>6.1f} {s['std']:>5.1f} "
              f"{s['cv']:>5.2f} {s['min']:>4} {s['max']:>4} {s['success_rate']:>5.0%}")

    # Highlight best candidates
    sweet_spot = [s for s in task_stats if 0.03 <= s["cv"] <= 0.25 and s["mean"] >= 4]
    if sweet_spot:
        print(f"\n{'='*70}")
        print(f"SWEET SPOT TASKS (CV 0.03-0.25, mean >= 4 calls): {len(sweet_spot)}")
        print(f"{'='*70}")
        for s in sweet_spot:
            print(f"  {s['task_id']}: mean={s['mean']:.1f} std={s['std']:.1f} "
                  f"CV={s['cv']:.2f} pass={s['success_rate']:.0%}")

    # Very low variance (might have floor effect)
    low_var = [s for s in task_stats if s["cv"] < 0.03]
    if low_var:
        print(f"\nFloor effect tasks (CV < 0.03): {len(low_var)}")
        for s in low_var:
            print(f"  {s['task_id']}: mean={s['mean']:.1f} std={s['std']:.1f}")

    # High variance tasks
    high_var = [s for s in task_stats if s["cv"] > 0.30]
    if high_var:
        print(f"\nHigh variance tasks (CV > 0.30): {len(high_var)}")

    # Token usage summary
    total_usage = defaultdict(int)
    for r in results:
        usage = r.get("usage") or {}
        for k, v in usage.items():
            total_usage[k] += v
    if total_usage:
        print(f"\n{'='*70}")
        print("TOKEN USAGE SUMMARY")
        print(f"{'='*70}")
        print(f"  Input tokens:         {total_usage['input_tokens']:>12,}")
        print(f"  Output tokens:        {total_usage['output_tokens']:>12,}")
        print(f"  Cache read tokens:    {total_usage['cache_read_input_tokens']:>12,}")
        print(f"  Cache creation tokens:{total_usage['cache_creation_input_tokens']:>12,}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline survey on AppWorld tasks")
    parser.add_argument("--model", type=str, default="sonnet", choices=list(MODEL_PRICING.keys()),
                        help="Model to run")
    parser.add_argument("--tasks-file", type=str, default=None,
                        help="File with task IDs (one per line). Default: test_normal.txt")
    parser.add_argument("--samples", type=int, default=5, help="Samples per task")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--cost-limit", type=float, default=280.0, help="Budget limit in dollars")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per run")
    parser.add_argument("--timeout", type=int, default=300, help="Subprocess timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--report-only", action="store_true", help="Just print report from checkpoint")
    args = parser.parse_args()

    model_key = args.model

    # Load tasks
    task_ids = load_task_ids(args.tasks_file)

    metadata = {
        "experiment": "baseline_survey",
        "model_key": model_key,
        "model_id": MODEL_IDS.get(model_key, model_key),
        "samples_per_task": args.samples,
        "max_turns": args.max_turns,
        "cost_limit": args.cost_limit,
        "workers": args.workers,
        "total_tasks": len(task_ids),
        "start_time": datetime.now().isoformat(),
    }

    # Resume or start fresh
    if args.resume or args.report_only:
        results, completed_keys, total_cost = load_checkpoint(model_key)
        print(f"Resumed: {len(results)} results, ${total_cost:.2f} spent")
    else:
        results, completed_keys, total_cost = [], set(), 0.0

    if args.report_only:
        print_report(results, total_cost, model_key)
        return

    # Build job list (skip completed)
    jobs = []
    for task_id in task_ids:
        for sample in range(args.samples):
            key = f"{task_id}_{sample}"
            if key not in completed_keys:
                jobs.append((task_id, sample))

    total_jobs = len(jobs)
    model_display = {"sonnet": "Sonnet 4.5", "opus": "Opus 4.5"}.get(model_key, model_key)
    print(f"\n{'#'*60}")
    print(f"# BASELINE SURVEY")
    print(f"# Model: {model_display}")
    print(f"# Tasks: {len(task_ids)} | Samples/task: {args.samples}")
    print(f"# Jobs remaining: {total_jobs}")
    print(f"# Budget: ${args.cost_limit:.0f} | Spent so far: ${total_cost:.2f}")
    print(f"# Workers: {args.workers}")
    print(f"{'#'*60}\n")

    if total_jobs == 0:
        print("All jobs already completed!")
        print_report(results, total_cost, model_key)
        return

    # Thread-safe cost tracking
    cost_lock = threading.Lock()
    budget_exceeded = threading.Event()
    completed_count = [len(results)]  # mutable counter
    checkpoint_counter = [0]

    def run_job(task_id, sample):
        if budget_exceeded.is_set():
            return None
        return run_single(task_id, model_key, sample, args.max_turns, args.timeout)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {}
        for task_id, sample in jobs:
            if budget_exceeded.is_set():
                break
            f = executor.submit(run_job, task_id, sample)
            future_to_job[f] = (task_id, sample)

        for future in as_completed(future_to_job):
            task_id, sample = future_to_job[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "task_id": task_id, "model_key": model_key,
                    "constraint_name": None, "sample": sample,
                    "success": False, "error": str(e),
                    "turn_count": 0, "total_api_calls": 0,
                    "has_reasoning": False, "reasoning_len": 0,
                    "usage": None,
                }

            if result is None:
                continue  # Budget exceeded, job was skipped

            run_cost = compute_cost(result.get("usage"), model_key)

            with cost_lock:
                results.append(result)
                total_cost += run_cost
                completed_count[0] += 1
                checkpoint_counter[0] += 1
                count = completed_count[0]

                # Progress line
                status = "OK" if not result.get("error") else "ERR"
                elapsed = time.time() - start_time
                print(
                    f"[{count - len(completed_keys)}/{total_jobs}] "
                    f"{task_id} s{sample} | {status} | "
                    f"calls={result.get('total_api_calls', 0)} | "
                    f"run=${run_cost:.3f} | total=${total_cost:.2f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

                # Check budget
                if total_cost >= args.cost_limit:
                    print(f"\n*** BUDGET LIMIT REACHED: ${total_cost:.2f} >= ${args.cost_limit:.2f} ***")
                    budget_exceeded.set()

                # Checkpoint every 20 runs
                if checkpoint_counter[0] >= 20:
                    save_checkpoint(results, total_cost, metadata, model_key)
                    checkpoint_counter[0] = 0

    # Final save
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    save_checkpoint(results, total_cost, metadata, model_key)
    save_final_results(results, total_cost, metadata, model_key)
    print_report(results, total_cost, model_key)


if __name__ == "__main__":
    main()

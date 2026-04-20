#!/usr/bin/env python3
"""Constraint experiment: measure whether constraints in API docs shift agent behavior.

4 tasks × 24 constraints × n=5 = 480 runs (Opus only).
Constraints are embedded in tool descriptions (API Usage Policy note) rather than
a prominent "Important Note" system prompt section.

Saves full trajectories for downstream monitor analysis.

Usage:
    # Validate: 1 task × 2 constraints × 1 sample — prints system prompt + CoT
    python appworld/run_constraint_experiment.py --validate

    # Full run
    python appworld/run_constraint_experiment.py --workers 4 --cost-limit 600

    # Resume interrupted run
    python appworld/run_constraint_experiment.py --resume --workers 4 --cost-limit 600

    # Just print report
    python appworld/run_constraint_experiment.py --report-only
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

_APPWORLD_VENV = PROJECT_ROOT / ".venv_appworld" / "bin" / "python"
VENV_PYTHON = str(_APPWORLD_VENV if _APPWORLD_VENV.exists() else PROJECT_ROOT / ".venv" / "bin" / "python")

LOGS_DIR = APPWORLD_EXP_DIR / "logs" / "constraint_exp"

# --- Experiment config ---

TASK_IDS = ["cef9191_3", "270f1ff_1", "3aa1a22_3", "dac78d9_1"]
MODEL_KEY = "opus"
SAMPLES_PER_CELL = 5

# Pricing (Opus 4.5)
PRICING = {
    "input": 15.0 / 1_000_000,
    "output": 75.0 / 1_000_000,
    "cache_read": 1.50 / 1_000_000,
    "cache_write": 18.75 / 1_000_000,
}


def compute_cost(usage: dict | None) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) * PRICING["input"]
        + usage.get("output_tokens", 0) * PRICING["output"]
        + usage.get("cache_read_input_tokens", 0) * PRICING["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * PRICING["cache_write"]
    )


def load_constraint_names() -> list[str]:
    """Load all 24 constraint names from constraints.json."""
    json_path = APPWORLD_EXP_DIR / "constraints.json"
    with open(json_path) as f:
        data = json.load(f)
    names = []
    for entry in data["constraints"]:
        names.append(f"{entry['id']}_negative")
        names.append(f"{entry['id']}_positive")
    return names


def run_single(task_id: str, constraint_name: str, sample: int,
               max_turns: int = 10, timeout: int = 600) -> dict:
    """Run a single job via subprocess with trajectory saving."""
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        task_id, MODEL_KEY, str(sample), str(max_turns),
        constraint_name, "save_trajectory",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ},
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return _error_result(
            task_id, constraint_name, sample,
            f"No result. stderr: {proc.stderr[-500:] if proc.stderr else 'empty'}",
        )
    except subprocess.TimeoutExpired:
        return _error_result(task_id, constraint_name, sample, f"Timeout ({timeout}s)")
    except Exception as e:
        return _error_result(task_id, constraint_name, sample, str(e))


def _error_result(task_id, constraint_name, sample, error):
    return {
        "task_id": task_id, "model_key": MODEL_KEY,
        "constraint_name": constraint_name, "sample": sample,
        "success": False, "error": error,
        "turn_count": 0, "total_api_calls": 0,
        "has_reasoning": False, "reasoning_len": 0,
        "usage": None,
    }


# --- Checkpoint ---

def checkpoint_path() -> Path:
    return LOGS_DIR / "checkpoint.json"


def load_checkpoint() -> tuple[list[dict], set[str], float]:
    cp = checkpoint_path()
    if not cp.exists():
        return [], set(), 0.0
    with open(cp) as f:
        data = json.load(f)
    results = data.get("results", [])
    total_cost = data.get("total_cost", 0.0)
    completed = {f"{r['task_id']}_{r['constraint_name']}_{r['sample']}" for r in results}
    return results, completed, total_cost


def save_checkpoint(results, total_cost, metadata):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path(), "w") as f:
        json.dump({
            "metadata": metadata,
            "total_cost": total_cost,
            "results": results,
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


def save_final_results(results, total_cost, metadata):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = LOGS_DIR / f"constraint_exp_{ts}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {**metadata, "end_time": datetime.now().isoformat()},
            "total_cost": total_cost,
            "results": results,
        }, f, indent=2)
    print(f"\n=> Saved final results: {output_file}")
    return output_file


# --- Validation ---

def run_validate(max_turns: int):
    """Run 1 task × 2 constraints × 1 sample for manual review."""
    task_id = TASK_IDS[0]
    constraints = ["user_cost_negative", "user_cost_positive"]

    print(f"{'='*70}")
    print("VALIDATION MODE")
    print(f"Task: {task_id}")
    print(f"Constraints: {constraints}")
    print(f"{'='*70}\n")

    for cname in constraints:
        print(f"\n{'─'*60}")
        print(f"Running: {task_id} + {cname}")
        print(f"{'─'*60}")

        result = run_single(task_id, cname, sample=0, max_turns=max_turns, timeout=600)

        print(f"\nResult: success={result.get('success')}, "
              f"api_calls={result.get('total_api_calls')}, "
              f"turns={result.get('turn_count')}, "
              f"has_reasoning={result.get('has_reasoning')}, "
              f"reasoning_len={result.get('reasoning_len')}")

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            continue

        # Load and inspect trajectory file
        traj_path = result.get("trajectory_path")
        if not traj_path or not os.path.exists(traj_path):
            print(f"WARNING: No trajectory file at {traj_path}")
            continue

        with open(traj_path) as f:
            traj = json.load(f)

        # 1. Verify constraint appears in API docs section
        sys_prompt = traj.get("system_prompt", "")
        print(f"\n--- System Prompt ({len(sys_prompt)} chars) ---")

        # Check constraint is in API docs (as blockquote policy note)
        if "> **API Usage Policy**:" in sys_prompt:
            print("[OK] Constraint embedded as API Usage Policy in docs section")
        else:
            print("[FAIL] Constraint NOT found as API Usage Policy!")

        # Check constraint does NOT appear as Important Note
        if "## Important Note" in sys_prompt:
            print("[FAIL] Constraint still appears as '## Important Note'!")
        else:
            print("[OK] No '## Important Note' section found")

        # Show the relevant section
        api_section_start = sys_prompt.find("## Available APIs")
        if api_section_start >= 0:
            snippet = sys_prompt[api_section_start:api_section_start + 400]
            print(f"\n--- API Docs Section (first 400 chars) ---\n{snippet}\n---")

        # 2. Check CoT capture
        turns = traj.get("turns", [])
        if turns:
            t0 = turns[0]
            reasoning = t0.get("internal_reasoning")
            if reasoning:
                print(f"\n--- Turn 0 Internal Reasoning ({len(reasoning)} chars) ---")
                print(reasoning[:500])
                if len(reasoning) > 500:
                    print("... [truncated for display]")
            else:
                print("\n[WARNING] Turn 0 has no internal reasoning")

        # 3. Constraint metadata
        meta = traj.get("constraint_metadata")
        if meta:
            print(f"\nConstraint metadata: {json.dumps(meta, indent=2)}")
        else:
            print("\n[WARNING] No constraint metadata in trajectory")

        print(f"\nTrajectory file: {traj_path}")
        print(f"Trajectory has {len(turns)} turns, "
              f"keys: {list(traj.keys())}")

    print(f"\n{'='*70}")
    print("Validation complete. Review output above before full run.")
    print(f"{'='*70}")


# --- Report ---

def print_report(results, total_cost):
    import statistics

    print(f"\n{'='*70}")
    print("CONSTRAINT EXPERIMENT RESULTS — Opus 4.5")
    print(f"{'='*70}")
    print(f"Total runs: {len(results)} | "
          f"Errors: {sum(1 for r in results if r.get('error'))} | "
          f"Total cost: ${total_cost:.2f}")
    if results:
        print(f"Avg cost/run: ${total_cost / len(results):.3f}")

    # Group by (task, constraint)
    by_task = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            by_task[r["task_id"]][r["constraint_name"]].append(r)

    # Per-task summary
    for task_id in TASK_IDS:
        task_results = by_task.get(task_id, {})
        if not task_results:
            continue

        print(f"\n{'─'*60}")
        print(f"Task: {task_id}")
        print(f"{'─'*60}")
        print(f"  {'Constraint':<30} {'n':>3} {'mean':>6} {'std':>5} {'min':>4} {'max':>4}")

        rows = []
        for cname, rs in sorted(task_results.items()):
            calls = [r["total_api_calls"] for r in rs]
            n = len(calls)
            mean = statistics.mean(calls) if calls else 0
            std = statistics.stdev(calls) if n >= 2 else 0
            rows.append((cname, n, mean, std, min(calls), max(calls)))

        for cname, n, mean, std, mn, mx in rows:
            print(f"  {cname:<30} {n:>3} {mean:>6.1f} {std:>5.1f} {mn:>4} {mx:>4}")

    # Overall by direction
    print(f"\n{'='*70}")
    print("BY DIRECTION (aggregated)")
    print(f"{'='*70}")
    by_dir = defaultdict(list)
    for r in results:
        if not r.get("error") and r.get("constraint_name"):
            direction = "positive" if r["constraint_name"].endswith("_positive") else "negative"
            by_dir[direction].append(r["total_api_calls"])

    for direction in ["negative", "positive"]:
        calls = by_dir.get(direction, [])
        if calls:
            mean = statistics.mean(calls)
            std = statistics.stdev(calls) if len(calls) >= 2 else 0
            print(f"  {direction:<12}: n={len(calls):>4}, mean={mean:.1f}, std={std:.1f}")

    # By category
    print(f"\n{'='*70}")
    print("BY CATEGORY (aggregated)")
    print(f"{'='*70}")

    # Load constraints to get categories
    constraint_names = load_constraint_names()
    json_path = APPWORLD_EXP_DIR / "constraints.json"
    with open(json_path) as f:
        cdata = json.load(f)
    name_to_cat = {}
    for entry in cdata["constraints"]:
        name_to_cat[f"{entry['id']}_negative"] = entry["category"]
        name_to_cat[f"{entry['id']}_positive"] = entry["category"]

    by_cat = defaultdict(list)
    for r in results:
        if not r.get("error") and r.get("constraint_name"):
            cat = name_to_cat.get(r["constraint_name"], "unknown")
            by_cat[cat].append(r["total_api_calls"])

    for cat in ["user", "model", "values"]:
        calls = by_cat.get(cat, [])
        if calls:
            mean = statistics.mean(calls)
            std = statistics.stdev(calls) if len(calls) >= 2 else 0
            print(f"  {cat:<12}: n={len(calls):>4}, mean={mean:.1f}, std={std:.1f}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Run constraint experiment on AppWorld tasks")
    parser.add_argument("--validate", action="store_true",
                        help="Validation mode: 1 task × 2 constraints × 1 sample")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--cost-limit", type=float, default=600.0, help="Budget limit ($)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per run")
    parser.add_argument("--timeout", type=int, default=600, help="Subprocess timeout (s)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CELL,
                        help="Samples per (task, constraint) cell")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--report-only", action="store_true", help="Print report from checkpoint")
    args = parser.parse_args()

    if args.validate:
        run_validate(args.max_turns)
        return

    constraint_names = load_constraint_names()
    total_cells = len(TASK_IDS) * len(constraint_names) * args.samples

    metadata = {
        "experiment": "constraint_exp",
        "model_key": MODEL_KEY,
        "tasks": TASK_IDS,
        "n_constraints": len(constraint_names),
        "samples_per_cell": args.samples,
        "max_turns": args.max_turns,
        "cost_limit": args.cost_limit,
        "workers": args.workers,
        "total_jobs": total_cells,
        "start_time": datetime.now().isoformat(),
    }

    # Resume or start fresh
    if args.resume or args.report_only:
        results, completed_keys, total_cost = load_checkpoint()
        print(f"Resumed: {len(results)} results, ${total_cost:.2f} spent")
    else:
        results, completed_keys, total_cost = [], set(), 0.0

    if args.report_only:
        print_report(results, total_cost)
        return

    # Build job list
    jobs = []
    for task_id in TASK_IDS:
        for cname in constraint_names:
            for sample in range(args.samples):
                key = f"{task_id}_{cname}_{sample}"
                if key not in completed_keys:
                    jobs.append((task_id, cname, sample))

    total_jobs = len(jobs)
    print(f"\n{'#'*60}")
    print(f"# CONSTRAINT EXPERIMENT")
    print(f"# Model: Opus 4.5")
    print(f"# Tasks: {len(TASK_IDS)} | Constraints: {len(constraint_names)} | "
          f"Samples: {args.samples}")
    print(f"# Jobs remaining: {total_jobs} / {total_cells}")
    print(f"# Budget: ${args.cost_limit:.0f} | Spent: ${total_cost:.2f}")
    print(f"# Workers: {args.workers}")
    print(f"{'#'*60}\n")

    if total_jobs == 0:
        print("All jobs already completed!")
        print_report(results, total_cost)
        return

    # Thread-safe tracking
    cost_lock = threading.Lock()
    budget_exceeded = threading.Event()
    completed_count = [len(results)]
    checkpoint_counter = [0]

    def run_job(task_id, cname, sample):
        if budget_exceeded.is_set():
            return None
        return run_single(task_id, cname, sample, args.max_turns, args.timeout)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {}
        for task_id, cname, sample in jobs:
            if budget_exceeded.is_set():
                break
            f = executor.submit(run_job, task_id, cname, sample)
            future_to_job[f] = (task_id, cname, sample)

        for future in as_completed(future_to_job):
            task_id, cname, sample = future_to_job[future]
            try:
                result = future.result()
            except Exception as e:
                result = _error_result(task_id, cname, sample, str(e))

            if result is None:
                continue

            run_cost = compute_cost(result.get("usage"))

            with cost_lock:
                results.append(result)
                total_cost += run_cost
                completed_count[0] += 1
                checkpoint_counter[0] += 1
                count = completed_count[0]

                status = "OK" if not result.get("error") else "ERR"
                elapsed = time.time() - start_time
                print(
                    f"[{count - len(completed_keys)}/{total_jobs}] "
                    f"{task_id} {cname} s{sample} | {status} | "
                    f"calls={result.get('total_api_calls', 0)} | "
                    f"run=${run_cost:.3f} total=${total_cost:.2f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

                if total_cost >= args.cost_limit:
                    print(f"\n*** BUDGET LIMIT: ${total_cost:.2f} >= ${args.cost_limit:.2f} ***")
                    budget_exceeded.set()

                if checkpoint_counter[0] >= 20:
                    save_checkpoint(results, total_cost, metadata)
                    checkpoint_counter[0] = 0

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    save_checkpoint(results, total_cost, metadata)
    save_final_results(results, total_cost, metadata)
    print_report(results, total_cost)


if __name__ == "__main__":
    main()

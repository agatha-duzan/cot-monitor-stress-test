"""Shared utilities for AppWorld experiments.

Provides module loading, subprocess execution, parallel job running,
statistical analysis, and result saving used across run.py and plot.py.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Path constants
APPWORLD_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APPWORLD_EXP_DIR.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# Ensure project root is on sys.path and APPWORLD_ROOT is set
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)


def load_local_module(name: str):
    """Load a module from the appworld experiment folder by name."""
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_single_subprocess(
    task_id: str,
    model_key: str,
    sample: int,
    max_turns: int = 10,
    constraint_name: str | None = None,
    timeout: int = 300,
) -> dict:
    """Run a single job via subprocess, parsing RESULT_JSON from stdout."""
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        task_id, model_key, str(sample), str(max_turns),
    ]
    if constraint_name:
        cmd.append(constraint_name)

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
            "constraint_name": constraint_name, "sample": sample,
            "success": False,
            "error": f"No result. stderr: {proc.stderr[-500:] if proc.stderr else 'empty'}",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id, "model_key": model_key,
            "constraint_name": constraint_name, "sample": sample,
            "success": False, "error": f"Timeout ({timeout}s)",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except Exception as e:
        return {
            "task_id": task_id, "model_key": model_key,
            "constraint_name": constraint_name, "sample": sample,
            "success": False, "error": str(e),
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }


def run_parallel_jobs(
    jobs: list[tuple],
    job_fn,
    max_workers: int = 6,
    label_fn=None,
) -> list[dict]:
    """Run jobs in parallel using ThreadPoolExecutor.

    Args:
        jobs: List of argument tuples to pass to job_fn.
        job_fn: Callable that takes *job_args and returns a result dict.
        max_workers: Number of parallel workers.
        label_fn: Optional callable(job_tuple, result) -> str for progress logging.

    Returns:
        List of result dicts.
    """
    total = len(jobs)
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(job_fn, *job): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                result = {"success": False, "error": str(e),
                          "turn_count": 0, "total_api_calls": 0,
                          "has_reasoning": False, "reasoning_len": 0}
            results.append(result)

            if label_fn:
                label = label_fn(job, result)
            else:
                status = "PASS" if result.get("success") else "FAIL"
                label = f"{status} | api={result.get('total_api_calls', 0)}"
            print(f"[{completed}/{total}] {label}", flush=True)

    return results


def run_single_direct(
    task_id: str,
    model_id: str,
    constraint_name: str | None,
    max_turns: int = 10,
    verbose: bool = False,
    constraints_dict: dict | None = None,
) -> dict:
    """Run a single experiment in-process and return a results dict.

    Used by pilot and phase1 which run sequentially in-process.
    """
    agent_module = load_local_module("agent")
    if constraints_dict is None:
        config = load_local_module("config")
        constraints_dict = config.CONSTRAINTS

    try:
        trajectory = agent_module.run_appworld_agent(
            task_id=task_id,
            model_id=model_id,
            constraint_name=constraint_name,
            max_turns=max_turns,
            enable_thinking=True,
            verbose=verbose,
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "constraint_name": constraint_name or "baseline",
            "constraint_text": constraints_dict.get(constraint_name, {}).get("text") if constraint_name else None,
            "constraint_category": constraints_dict.get(constraint_name, {}).get("category") if constraint_name else None,
            "constraint_theme": constraints_dict.get(constraint_name, {}).get("theme") if constraint_name else None,
            "constraint_direction": constraints_dict.get(constraint_name, {}).get("direction") if constraint_name else None,
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


def compute_std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_condition_stats(results: list[dict], constraints_dict: dict | None = None) -> dict:
    """Compute per-condition stats and deltas vs baseline.

    Returns dict with keys 'by_condition' and 'behavioral_deltas'.
    Works for both pilot (analyze_results) and phase1 (analyze_model_results).
    """
    by_condition = {}
    for r in results:
        cond = r["constraint_name"]
        by_condition.setdefault(cond, []).append(r)

    stats = {}
    for cond, rs in by_condition.items():
        api_calls = [r["total_api_calls"] for r in rs]
        turns = [r["turn_count"] for r in rs]
        stats[cond] = {
            "n": len(rs),
            "total": len(rs),
            "success_count": sum(1 for r in rs if r["success"]),
            "success_rate": sum(1 for r in rs if r["success"]) / len(rs),
            "avg_api_calls": sum(api_calls) / len(api_calls),
            "std_api_calls": compute_std(api_calls),
            "min_api_calls": min(api_calls),
            "max_api_calls": max(api_calls),
            "avg_turns": sum(turns) / len(turns),
            "std_turns": compute_std(turns),
            "avg_internal_reasoning_length": sum(r.get("internal_reasoning_length", 0) for r in rs) / len(rs),
            "avg_reasoning_len": sum(r.get("internal_reasoning_length", 0) for r in rs) / len(rs),
            "has_reasoning_rate": sum(1 for r in rs if r.get("has_internal_reasoning")) / len(rs),
            "reasoning_rate": sum(1 for r in rs if r.get("has_internal_reasoning")) / len(rs),
        }

    deltas = compute_deltas(stats, constraints_dict)
    return {"by_condition": stats, "behavioral_deltas": deltas}


def compute_deltas(stats: dict, constraints_dict: dict | None = None) -> dict:
    """Compute behavioral deltas vs baseline from per-condition stats."""
    baseline_avg = stats.get("baseline", {}).get("avg_api_calls", 0)
    deltas = {}
    if baseline_avg > 0:
        for cond, s in stats.items():
            if cond != "baseline":
                delta = s["avg_api_calls"] - baseline_avg
                expected = "unknown"
                if constraints_dict and cond in constraints_dict:
                    expected = constraints_dict[cond].get("expected_effect", "unknown")
                deltas[cond] = {
                    "api_call_delta": delta,
                    "api_call_delta_pct": (delta / baseline_avg) * 100,
                    "expected_direction": expected,
                }
    return deltas


def save_results(data: dict, output_file: Path):
    """Save results dict to a JSON file, creating parent dirs if needed."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n=> Saved: {output_file}")


def timestamp_str() -> str:
    """Return a timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

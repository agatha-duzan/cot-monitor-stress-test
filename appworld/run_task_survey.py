#!/usr/bin/env python3
"""Survey easy AppWorld tasks to find ones the models can solve.

Runs opus and kimi on 19 level-1 test_normal tasks (baseline, no constraints)
with 10 samples each. Uses subprocess parallelism to avoid SQLite threading issues.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APPWORLD_EXP_DIR = Path(__file__).parent
PROJECT_ROOT = APPWORLD_EXP_DIR.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# 13 remaining level-1 test_normal tasks (first 6 already completed in prior run)
SURVEY_TASKS = [
    "21abae1_1", "29a7b7e_1", "31dc501_1", "425a494_1", "552869a_1",
    "59fae45_1", "5a83b05_1", "7847649_1", "a30375d_1", "afc4005_1",
    "dac78d9_1", "f3f60f0_1", "fd1f8fa_1",
]

MODELS = ["opus", "kimi"]
SAMPLES = 10
MAX_TURNS = 10
MAX_WORKERS = 6


def run_single_subprocess(task_id: str, model_key: str, sample: int) -> dict:
    """Spawn a subprocess for one job, parse JSON result."""
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        task_id, model_key, str(sample), str(MAX_TURNS),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ},
        )
        # Parse the RESULT_JSON line from stdout
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        # If no result line, return error
        return {
            "task_id": task_id, "model_key": model_key, "sample": sample,
            "success": False, "error": f"No result. stderr: {proc.stderr[-500:]}",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id, "model_key": model_key, "sample": sample,
            "success": False, "error": "Timeout (300s)",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except Exception as e:
        return {
            "task_id": task_id, "model_key": model_key, "sample": sample,
            "success": False, "error": str(e),
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }


def run_survey():
    output_dir = APPWORLD_EXP_DIR / "logs" / "easy_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all jobs
    jobs = []
    for task_id in SURVEY_TASKS:
        for model_key in MODELS:
            for sample in range(SAMPLES):
                jobs.append((task_id, model_key, sample))

    total = len(jobs)
    print(f"Running {total} jobs ({len(SURVEY_TASKS)} tasks × {len(MODELS)} models × {SAMPLES} samples)")
    print(f"Parallelism: {MAX_WORKERS} subprocess workers\n", flush=True)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(run_single_subprocess, t, m, s): (t, m, s)
            for t, m, s in jobs
        }
        for future in as_completed(future_to_job):
            task_id, model_key, sample = future_to_job[future]
            completed += 1
            entry = future.result()
            results.append(entry)
            status = "PASS" if entry["success"] else "FAIL"
            print(f"[{completed}/{total}] {model_key} | {task_id} | s{sample+1} -> {status} | api={entry['total_api_calls']}", flush=True)

    # Save raw results for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"easy_test_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({"results": results, "tasks": SURVEY_TASKS, "models": MODELS, "samples": SAMPLES}, f, indent=2)
    print(f"\nSaved: {out_file}")

    # Merge with previous results (keep only non-402 entries from old runs)
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

    print_summary(all_results)
    plot_results(all_results)


def print_summary(results: list[dict]):
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

    # Overall per model
    print(f"\n{'='*50}")
    print("OVERALL PER MODEL")
    print(f"{'='*50}")
    for model_key in MODELS:
        model_runs = [r for r in results if r["model_key"] == model_key]
        sr = sum(1 for r in model_runs if r["success"]) / len(model_runs)
        avg_api = np.mean([r["total_api_calls"] for r in model_runs])
        print(f"{model_key:<8} success={sr:.0%}  avg_api={avg_api:.1f}")

    # Good tasks
    print(f"\n{'='*50}")
    print("GOOD TASKS (>=50% success on both models)")
    print(f"{'='*50}")
    for task_id, models_data in sorted(task_stats.items()):
        good_count = sum(1 for m in models_data.values() if m["success_rate"] >= 0.5)
        if good_count >= 2:
            parts = [f"{mk}: {md['success_rate']:.0%} ({md['avg_api']:.0f}±{md['std_api']:.0f} calls)"
                     for mk, md in models_data.items()]
            print(f"  {task_id}: {', '.join(parts)}")


def plot_results(results: list[dict]):
    """Plot success rates and API call variance per task."""
    plot_dir = APPWORLD_EXP_DIR / "plots" / "easy_test"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
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

    for i, model_key in enumerate(MODELS):
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

    # --- Plot 2: API calls with variance (mean ± std) ---
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, model_key in enumerate(MODELS):
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

    # --- Plot 3: Combined — only tasks with >=50% success on at least one model ---
    good_tasks = []
    for task_id in tasks:
        for model_key in MODELS:
            runs = task_model_stats.get((task_id, model_key), [])
            sr = sum(1 for r in runs if r["success"]) / max(len(runs), 1)
            if sr >= 0.5:
                good_tasks.append(task_id)
                break

    if good_tasks:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(good_tasks) * 0.8), 8), sharex=True)
        xg = np.arange(len(good_tasks))

        for i, model_key in enumerate(MODELS):
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


if __name__ == "__main__":
    run_survey()

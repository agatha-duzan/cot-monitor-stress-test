#!/usr/bin/env python3
"""Phase 2: Constraint experiment on opus with top-5 tasks.

For each task × constraint (24) × 5 samples = 600 runs.
Baseline data reused from easy_test survey.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
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

# Top 5 tasks by opus success rate
TASKS = ["afc4005_1", "425a494_1", "7847649_1", "dac78d9_1", "1150ed6_1"]
MODEL = "opus"
SAMPLES = 5
MAX_TURNS = 10
MAX_WORKERS = 6

# Load constraint names
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("config", APPWORLD_EXP_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

CONSTRAINTS = config.CONSTRAINTS
CONSTRAINT_NAMES = sorted(CONSTRAINTS.keys())  # 24 constraints


def run_single_subprocess(task_id: str, constraint_name: str, sample: int) -> dict:
    cmd = [
        VENV_PYTHON, str(APPWORLD_EXP_DIR / "run_single_job.py"),
        task_id, MODEL, str(sample), str(MAX_TURNS), constraint_name,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ},
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return {
            "task_id": task_id, "model_key": MODEL, "constraint_name": constraint_name,
            "sample": sample, "success": False,
            "error": f"No result. stderr: {proc.stderr[-500:]}",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id, "model_key": MODEL, "constraint_name": constraint_name,
            "sample": sample, "success": False, "error": "Timeout (300s)",
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }
    except Exception as e:
        return {
            "task_id": task_id, "model_key": MODEL, "constraint_name": constraint_name,
            "sample": sample, "success": False, "error": str(e),
            "turn_count": 0, "total_api_calls": 0,
            "has_reasoning": False, "reasoning_len": 0,
        }


def load_baseline():
    """Load baseline results for opus on the 5 tasks from easy_test."""
    baseline = defaultdict(list)
    log_dir = APPWORLD_EXP_DIR / "logs" / "easy_test"
    for f in sorted(log_dir.glob("easy_test_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if r["model_key"] == "opus" and r["task_id"] in TASKS:
                if r["total_api_calls"] > 0 or r["turn_count"] > 0:
                    baseline[r["task_id"]].append(r)
    return baseline


def run_experiment():
    output_dir = APPWORLD_EXP_DIR / "logs" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task_id in TASKS:
        for cname in CONSTRAINT_NAMES:
            for sample in range(SAMPLES):
                jobs.append((task_id, cname, sample))

    total = len(jobs)
    print(f"Running {total} jobs ({len(TASKS)} tasks × {len(CONSTRAINT_NAMES)} constraints × {SAMPLES} samples)")
    print(f"Model: {MODEL} | Max turns: {MAX_TURNS} | Workers: {MAX_WORKERS}\n", flush=True)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(run_single_subprocess, t, c, s): (t, c, s)
            for t, c, s in jobs
        }
        for future in as_completed(future_to_job):
            task_id, cname, sample = future_to_job[future]
            completed += 1
            entry = future.result()
            results.append(entry)
            status = "PASS" if entry["success"] else "FAIL"
            print(f"[{completed}/{total}] {task_id} | {cname} | s{sample+1} -> {status} | api={entry['total_api_calls']}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"phase2_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({
            "results": results, "tasks": TASKS, "model": MODEL,
            "constraints": CONSTRAINT_NAMES, "samples": SAMPLES,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")

    baseline = load_baseline()
    print_summary(results, baseline)
    plot_results(results, baseline)


def print_summary(results, baseline):
    print(f"\n{'='*80}")
    print("PHASE 2 SUMMARY")
    print(f"{'='*80}")

    # Per task: baseline vs constraint stats
    for task_id in TASKS:
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

        for cname in CONSTRAINT_NAMES:
            runs = by_constraint.get(cname, [])
            if not runs:
                continue
            sr = sum(1 for r in runs if r["success"]) / len(runs)
            avg_api = np.mean([r["total_api_calls"] for r in runs])
            delta = ((avg_api - bl_api) / bl_api * 100) if bl_api > 0 else 0
            direction = CONSTRAINTS[cname]["direction"][:3]
            print(f"  {cname:<30} {direction:<5} {sr:>5.0%}   {avg_api:>6.1f}  {delta:>+6.1f}%")


def plot_results(results, baseline):
    plot_dir = APPWORLD_EXP_DIR / "plots" / "phase2"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Get constraint IDs (unique, sorted) and their neg/pos names
    constraint_ids = sorted(set(CONSTRAINTS[c]["constraint_id"] for c in CONSTRAINT_NAMES))

    # --- Plot 1: Per-task bar charts (neg=blue, pos=red, baseline=black line) ---
    for task_id in TASKS:
        bl_runs = baseline.get(task_id, [])
        bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 0

        task_results = [r for r in results if r["task_id"] == task_id]
        by_constraint = defaultdict(list)
        for r in task_results:
            by_constraint[r["constraint_name"]].append(r)

        fig, ax = plt.subplots(figsize=(16, 5))
        x = np.arange(len(constraint_ids))
        w = 0.35

        neg_means, neg_stds = [], []
        pos_means, pos_stds = [], []

        for cid in constraint_ids:
            neg_name = f"{cid}_negative"
            pos_name = f"{cid}_positive"

            neg_runs = by_constraint.get(neg_name, [])
            pos_runs = by_constraint.get(pos_name, [])

            neg_apis = [r["total_api_calls"] for r in neg_runs] if neg_runs else [0]
            pos_apis = [r["total_api_calls"] for r in pos_runs] if pos_runs else [0]

            neg_means.append(np.mean(neg_apis))
            neg_stds.append(np.std(neg_apis))
            pos_means.append(np.mean(pos_apis))
            pos_stds.append(np.std(pos_apis))

        ax.bar(x - w/2, neg_means, w, yerr=neg_stds, label="Negative", color="#3498db", alpha=0.85, capsize=2)
        ax.bar(x + w/2, pos_means, w, yerr=pos_stds, label="Positive", color="#e74c3c", alpha=0.85, capsize=2)
        ax.axhline(bl_api, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({bl_api:.1f})")

        ax.set_ylabel("API Calls (mean ± std)")
        ax.set_title(f"Opus — {task_id}: API Calls by Constraint")
        ax.set_xticks(x)
        ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"api_calls_{task_id}.png", dpi=150)
        print(f"Saved: {plot_dir}/api_calls_{task_id}.png")
        plt.close(fig)

    # --- Plot 2: Delta heatmap (tasks × constraints) ---
    fig, ax = plt.subplots(figsize=(18, 6))

    # Build matrix: rows=tasks, cols=constraint_names (neg then pos interleaved)
    col_labels = []
    for cid in constraint_ids:
        col_labels.append(f"{cid}_neg")
        col_labels.append(f"{cid}_pos")

    matrix = np.zeros((len(TASKS), len(col_labels)))
    for i, task_id in enumerate(TASKS):
        bl_runs = baseline.get(task_id, [])
        bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 1

        task_results_list = [r for r in results if r["task_id"] == task_id]
        by_constraint = defaultdict(list)
        for r in task_results_list:
            by_constraint[r["constraint_name"]].append(r)

        for j, cid in enumerate(constraint_ids):
            for k, direction in enumerate(["negative", "positive"]):
                cname = f"{cid}_{direction}"
                runs = by_constraint.get(cname, [])
                avg_api = np.mean([r["total_api_calls"] for r in runs]) if runs else bl_api
                delta_pct = (avg_api - bl_api) / bl_api * 100 if bl_api > 0 else 0
                matrix[i, j * 2 + k] = delta_pct

    vmax = max(abs(matrix.min()), abs(matrix.max()), 50)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([t.split("_")[0] for t in TASKS], fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_title("% Change in API Calls vs Baseline (Opus)")
    fig.colorbar(im, ax=ax, label="% change", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_heatmap.png", dpi=150)
    print(f"Saved: {plot_dir}/delta_heatmap.png")
    plt.close(fig)

    # --- Plot 3: Aggregated across tasks (avg delta per constraint) ---
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(constraint_ids))
    w = 0.35

    neg_deltas, pos_deltas = [], []
    for cid in constraint_ids:
        neg_d, pos_d = [], []
        for task_id in TASKS:
            bl_runs = baseline.get(task_id, [])
            bl_api = np.mean([r["total_api_calls"] for r in bl_runs]) if bl_runs else 1

            task_res = [r for r in results if r["task_id"] == task_id]
            by_c = defaultdict(list)
            for r in task_res:
                by_c[r["constraint_name"]].append(r)

            neg_runs = by_c.get(f"{cid}_negative", [])
            pos_runs = by_c.get(f"{cid}_positive", [])

            if neg_runs and bl_api > 0:
                neg_d.append((np.mean([r["total_api_calls"] for r in neg_runs]) - bl_api) / bl_api * 100)
            if pos_runs and bl_api > 0:
                pos_d.append((np.mean([r["total_api_calls"] for r in pos_runs]) - bl_api) / bl_api * 100)

        neg_deltas.append(np.mean(neg_d) if neg_d else 0)
        pos_deltas.append(np.mean(pos_d) if pos_d else 0)

    ax.bar(x - w/2, neg_deltas, w, label="Negative (expect fewer)", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, pos_deltas, w, label="Positive (expect more)", color="#e74c3c", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Avg % Change in API Calls vs Baseline")
    ax.set_title("Opus: Average Effect of Constraints Across 5 Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "avg_delta_by_constraint.png", dpi=150)
    print(f"Saved: {plot_dir}/avg_delta_by_constraint.png")
    plt.close(fig)

    # --- Plot 4: Success rate comparison ---
    fig, ax = plt.subplots(figsize=(16, 5))

    neg_sr, pos_sr, bl_sr_list = [], [], []
    for cid in constraint_ids:
        neg_all, pos_all = [], []
        for task_id in TASKS:
            task_res = [r for r in results if r["task_id"] == task_id]
            by_c = defaultdict(list)
            for r in task_res:
                by_c[r["constraint_name"]].append(r)

            neg_runs = by_c.get(f"{cid}_negative", [])
            pos_runs = by_c.get(f"{cid}_positive", [])
            neg_all.extend(neg_runs)
            pos_all.extend(pos_runs)

        neg_sr.append(sum(1 for r in neg_all if r["success"]) / max(len(neg_all), 1))
        pos_sr.append(sum(1 for r in pos_all if r["success"]) / max(len(pos_all), 1))

    # Overall baseline success
    all_bl = [r for bl in baseline.values() for r in bl]
    bl_sr_val = sum(1 for r in all_bl if r["success"]) / max(len(all_bl), 1)

    ax.bar(x - w/2, neg_sr, w, label="Negative", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, pos_sr, w, label="Positive", color="#e74c3c", alpha=0.85)
    ax.axhline(bl_sr_val, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({bl_sr_val:.0%})")
    ax.set_ylabel("Success Rate")
    ax.set_title("Opus: Success Rate by Constraint (across 5 tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "success_by_constraint.png", dpi=150)
    print(f"Saved: {plot_dir}/success_by_constraint.png")
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()

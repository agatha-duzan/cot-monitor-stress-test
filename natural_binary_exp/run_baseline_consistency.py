#!/usr/bin/env python3
"""Baseline consistency check: measure how often models flip on the same prompt.

Runs each baseline prompt multiple times per model to establish a "noise floor"
for the natural binary choice experiment. If models already flip 30% on identical
prompts, the 44% constraint-induced flip rate is less meaningful.

Usage:
    # Pilot: 1 model, 2 repeats
    python natural_binary_exp/run_baseline_consistency.py \
        --models haiku --repeats 2 --log-name consistency_pilot

    # Full run: all 6 models, 10 repeats
    python natural_binary_exp/run_baseline_consistency.py \
        --models haiku sonnet opus kimi glm grok --repeats 10 \
        --max-concurrent 10 --log-name baseline_consistency
"""

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from run_exp2 import (
    EXP_DIR,
    MODEL_CONFIGS,
    MODEL_ORDER,
    extract_result_from_eval,
    get_baselines,
    load_dataset,
    run_single_eval,
)


async def run_consistency_check(args):
    dataset = load_dataset()
    baselines = get_baselines(dataset)

    if args.scenarios:
        baselines = [b for b in baselines if b["scenario_id"] in args.scenarios]

    if args.models == ["all"]:
        models = [m for m in MODEL_ORDER if m in MODEL_CONFIGS]
    else:
        models = args.models

    base_log_dir = EXP_DIR / "logs" / args.log_name
    semaphore = asyncio.Semaphore(args.max_concurrent)

    total_runs = len(baselines) * len(models) * args.repeats
    print(f"\n{'='*70}")
    print("BASELINE CONSISTENCY CHECK")
    print(f"{'='*70}")
    print(f"Scenarios: {len(baselines)}")
    print(f"Models: {len(models)} ({models})")
    print(f"Repeats per (scenario, model): {args.repeats}")
    print(f"Total runs: {total_runs}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Log dir: {base_log_dir}")
    print()

    # ── Launch all runs concurrently ──
    # Key: (scenario_id, model_id) → list of choices
    results_map: dict[tuple[str, str], list[str]] = {}

    async def run_one(scenario_id, prompt_id, model_id, repeat_idx):
        config = MODEL_CONFIGS[model_id]
        log_dir = base_log_dir / scenario_id / model_id / f"rep_{repeat_idx}"
        eval_path = await run_single_eval(
            prompt_id, config, log_dir, semaphore, args.timeout
        )
        if eval_path:
            result = extract_result_from_eval(eval_path)
            if result and result.get("model_choice"):
                choice = result["model_choice"]
                print(
                    f"  [{config['display_name']}] {scenario_id} rep {repeat_idx} → {choice}",
                    flush=True,
                )
                return (scenario_id, model_id, choice)
            else:
                print(
                    f"  [{config['display_name']}] {scenario_id} rep {repeat_idx} → no choice",
                    flush=True,
                )
        else:
            print(
                f"  [{config['display_name']}] {scenario_id} rep {repeat_idx} → FAILED",
                flush=True,
            )
        return None

    tasks = []
    for baseline in baselines:
        sid = baseline["scenario_id"]
        pid = baseline["prompt_id"]
        for model_id in models:
            for rep in range(args.repeats):
                tasks.append(run_one(sid, pid, model_id, rep))

    print(f"Launching {len(tasks)} eval runs...\n")
    raw_results = await asyncio.gather(*tasks)

    # Collect choices
    for r in raw_results:
        if r is not None:
            sid, mid, choice = r
            results_map.setdefault((sid, mid), []).append(choice)

    # ── Compute consistency stats ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    per_pair = []  # per (scenario, model) stats
    for baseline in baselines:
        sid = baseline["scenario_id"]
        for model_id in models:
            choices = results_map.get((sid, model_id), [])
            if not choices:
                per_pair.append({
                    "scenario_id": sid,
                    "model_id": model_id,
                    "n_runs": 0,
                    "A_count": 0,
                    "B_count": 0,
                    "other_count": 0,
                    "consistency_rate": None,
                    "majority_choice": None,
                })
                continue

            counter = Counter(choices)
            a_count = counter.get("A", 0)
            b_count = counter.get("B", 0)
            other_count = len(choices) - a_count - b_count
            majority_count = max(a_count, b_count)
            consistency = majority_count / len(choices)
            majority_choice = "A" if a_count >= b_count else "B"

            per_pair.append({
                "scenario_id": sid,
                "model_id": model_id,
                "n_runs": len(choices),
                "A_count": a_count,
                "B_count": b_count,
                "other_count": other_count,
                "consistency_rate": consistency,
                "majority_choice": majority_choice,
            })

    # ── Console table ──
    # Header
    scenario_ids = [b["scenario_id"] for b in baselines]
    col_w = 14
    header = f"{'Scenario':<22}" + "".join(
        f"{MODEL_CONFIGS[m]['display_name']:^{col_w}}" for m in models
    )
    print(header)
    print("-" * len(header))

    for sid in scenario_ids:
        row = f"{sid:<22}"
        for model_id in models:
            entry = next(
                (p for p in per_pair if p["scenario_id"] == sid and p["model_id"] == model_id),
                None,
            )
            if entry and entry["consistency_rate"] is not None:
                cell = f"{entry['A_count']}A/{entry['B_count']}B {entry['consistency_rate']:.0%}"
            else:
                cell = "—"
            row += f"{cell:^{col_w}}"
        print(row)

    # ── Per-model averages ──
    print(f"\n{'Per-model average consistency':}")
    print("-" * 40)
    model_averages = {}
    for model_id in models:
        rates = [
            p["consistency_rate"]
            for p in per_pair
            if p["model_id"] == model_id and p["consistency_rate"] is not None
        ]
        if rates:
            avg = sum(rates) / len(rates)
            model_averages[model_id] = avg
            print(f"  {MODEL_CONFIGS[model_id]['display_name']:<20} {avg:.1%}  (n={len(rates)} scenarios)")
        else:
            model_averages[model_id] = None
            print(f"  {MODEL_CONFIGS[model_id]['display_name']:<20} —")

    # Overall
    all_rates = [p["consistency_rate"] for p in per_pair if p["consistency_rate"] is not None]
    overall = sum(all_rates) / len(all_rates) if all_rates else None
    print(f"\n  {'Overall':<20} {overall:.1%}" if overall else "\n  Overall: —")
    if overall:
        print(f"  (Noise floor = {1 - overall:.1%} flip rate on identical prompts)")

    # ── Save results ──
    base_log_dir.mkdir(parents=True, exist_ok=True)
    results_path = base_log_dir / "consistency_results.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "baseline_consistency_check",
        "config": {
            "scenarios": scenario_ids,
            "models": models,
            "repeats": args.repeats,
            "max_concurrent": args.max_concurrent,
        },
        "summary": {
            "total_runs": sum(p["n_runs"] for p in per_pair),
            "overall_consistency": overall,
            "noise_floor_flip_rate": (1 - overall) if overall else None,
            "model_averages": model_averages,
        },
        "per_scenario_model": per_pair,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline consistency check")
    parser.add_argument("--models", nargs="+", default=["haiku"])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--log-name", type=str, required=True)

    args = parser.parse_args()

    if args.models == ["all"]:
        pass  # handled in run_consistency_check
    asyncio.run(run_consistency_check(args))


if __name__ == "__main__":
    main()

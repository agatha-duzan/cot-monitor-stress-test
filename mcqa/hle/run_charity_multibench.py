#!/usr/bin/env python3
"""Runner for charity motivation experiment across WMDP-bio and GPQA.

Usage:
    python mcqa/hle/run_charity_multibench.py --benchmarks wmdp_bio gpqa --log-name charity_multi_v1
    python mcqa/hle/run_charity_multibench.py --benchmarks wmdp_bio --models haiku sonnet --log-name test
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HLE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = HLE_DIR.parent.parent

MODEL_CONFIGS = {
    "haiku": {
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "display_name": "Haiku 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "sonnet": {
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "display_name": "Sonnet 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "opus": {
        "model_id": "anthropic/claude-opus-4-5-20251101",
        "display_name": "Opus 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "kimi": {
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking_args": ["-M", "reasoning_enabled=true", "--max-tokens", "65536"],
    },
    "glm": {
        "model_id": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking_args": ["-M", "reasoning_enabled=true", "--max-tokens", "65536"],
    },
    "grok": {
        "model_id": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
}

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
CONDITIONS = ["baseline", "neutral", "good_entity", "bad_entity"]

BENCHMARK_TASKS = {
    "wmdp_bio": "charity_wmdp_bio",
    "gpqa": "charity_gpqa",
}


async def run_single(
    benchmark: str,
    condition: str,
    model_config: dict,
    log_dir: Path,
    limit: int = 400,
    timeout: int = 3600,
    max_connections: int = 10,
) -> Path | None:
    """Run a single evaluation via inspect eval subprocess."""
    log_dir.mkdir(parents=True, exist_ok=True)

    task_name = BENCHMARK_TASKS[benchmark]

    cmd = [
        "/home/agatha/Anaconda3/bin/inspect",
        "eval",
        f"mcqa/hle/charity_multibench_task.py@{task_name}",
        "--model", model_config["model_id"],
        "--limit", str(limit),
        "--log-dir", str(log_dir),
        "--max-connections", str(max_connections),
        "--max-retries", "3",
        "--no-fail-on-error",
        "-T", f"condition={condition}",
    ]

    cmd.extend(model_config.get("thinking_args", []))

    display = f"{model_config['display_name']} [{benchmark}/{condition}]"
    print(f"  Starting {display}...", flush=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_DIR),
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        print(f"  TIMEOUT: {display}", flush=True)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode()[:500] if stderr else "unknown error"
        print(f"  ERROR: {display} - {err_msg}", flush=True)
        return None

    eval_files = list(log_dir.glob("*.eval"))
    if eval_files:
        result = max(eval_files, key=lambda p: p.stat().st_mtime)
        print(f"  DONE: {display} -> {result.name}", flush=True)
        return result

    print(f"  NO OUTPUT: {display}", flush=True)
    return None


async def run_experiment(args):
    """Run the full experiment."""
    models = MODEL_ORDER if args.models == ["all"] else args.models
    conditions = args.conditions or CONDITIONS
    benchmarks = args.benchmarks

    base_log_dir = HLE_DIR / "logs" / args.log_name

    print(f"\n{'='*70}")
    print(f"CHARITY MULTIBENCH EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Models: {', '.join(models)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Limit: {args.limit}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {base_log_dir}")
    print(f"{'='*70}\n")

    sem = asyncio.Semaphore(args.concurrency)
    results = {}

    async def run_with_sem(benchmark, condition, model_id):
        async with sem:
            config = MODEL_CONFIGS[model_id]
            log_dir = base_log_dir / benchmark / condition / model_id / "rep1"
            return (
                benchmark, condition, model_id,
                await run_single(
                    benchmark=benchmark,
                    condition=condition,
                    model_config=config,
                    log_dir=log_dir,
                    limit=args.limit,
                    timeout=args.timeout,
                    max_connections=args.max_connections,
                ),
            )

    tasks = []
    for benchmark in benchmarks:
        for condition in conditions:
            for model_id in models:
                tasks.append(run_with_sem(benchmark, condition, model_id))

    completed = await asyncio.gather(*tasks)

    for benchmark, condition, model_id, eval_path in completed:
        key = f"{benchmark}/{condition}"
        if key not in results:
            results[key] = {}
        if eval_path:
            results[key][model_id] = str(eval_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "benchmarks": benchmarks,
            "models": models,
            "conditions": conditions,
            "limit": args.limit,
        },
        "eval_files": results,
    }

    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Benchmark':<12} {'Condition':<15} {'Status'}")
    print("-" * 60)
    for benchmark in benchmarks:
        for condition in conditions:
            for model_id in models:
                key = f"{benchmark}/{condition}"
                status = "OK" if results.get(key, {}).get(model_id) else "FAILED"
                display = MODEL_CONFIGS[model_id]["display_name"]
                print(f"{display:<20} {benchmark:<12} {condition:<15} {status}")

    total = len(benchmarks) * len(conditions) * len(models)
    ok = sum(1 for b in benchmarks for c in conditions for m in models
             if results.get(f"{b}/{c}", {}).get(m))
    print(f"\n{ok}/{total} evaluations completed successfully")
    print(f"Results saved to: {base_log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run charity motivation experiment across benchmarks"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["wmdp_bio", "gpqa"],
        choices=["wmdp_bio", "gpqa"],
        help="Benchmarks to run (default: both)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        choices=CONDITIONS,
        help="Conditions to run (default: all four)",
    )
    parser.add_argument("--limit", type=int, default=400,
                        help="Max samples per evaluation (default: 400 = 50 questions × 8 pairs)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per eval in seconds (default: 3600)")
    parser.add_argument("--concurrency", type=int, default=6,
                        help="Max concurrent evaluations (default: 6)")
    parser.add_argument("--max-connections", type=int, default=10,
                        help="Max concurrent API connections per eval (default: 10)")
    parser.add_argument("--log-name", type=str, required=True,
                        help="Name for log subdirectory")

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

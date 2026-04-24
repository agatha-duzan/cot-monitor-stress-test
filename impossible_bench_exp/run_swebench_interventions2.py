#!/usr/bin/env python3
"""
SWE-bench new interventions: holistic_codebase and spec_is_truth.
6 models × 2 interventions × conflicting only = 12 evals.
Uses same 10 samples (0-9) as original interventions for comparability.

Usage:
    sg docker -c "bash -c 'source ../keys.sh && /home/agatha/Anaconda3/bin/python impossible_bench_exp/run_swebench_interventions2.py'"
"""

import asyncio
import sys
from pathlib import Path

INSPECT_BIN = "/home/agatha/Anaconda3/bin/inspect"

LOG_DIR = Path("impossible_bench_exp/logs/swebench_interventions")

MODELS = [
    {
        "name": "haiku",
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "thinking_args": ["--reasoning-tokens", "4096"],
        "provider": "anthropic",
    },
    {
        "name": "sonnet",
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "thinking_args": ["--reasoning-tokens", "4096"],
        "provider": "anthropic",
    },
    {
        "name": "opus",
        "model_id": "anthropic/claude-opus-4-20250514",
        "thinking_args": ["--reasoning-tokens", "4096"],
        "provider": "anthropic",
    },
    {
        "name": "kimi",
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "provider": "openrouter",
    },
    {
        "name": "glm",
        "model_id": "openrouter/z-ai/glm-4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "provider": "openrouter",
    },
    {
        "name": "grok",
        "model_id": "openrouter/x-ai/grok-3-mini",
        "thinking_args": ["--reasoning-tokens", "4096"],
        "provider": "openrouter",
    },
]

INTERVENTIONS = ["holistic_codebase", "spec_is_truth"]

MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 100

ANTHROPIC_SEMAPHORE = asyncio.Semaphore(2)
OPENROUTER_SEMAPHORE = asyncio.Semaphore(2)


def get_semaphore(provider: str) -> asyncio.Semaphore:
    if provider == "anthropic":
        return ANTHROPIC_SEMAPHORE
    return OPENROUTER_SEMAPHORE


async def run_eval(model: dict, intervention: str) -> dict:
    sem = get_semaphore(model["provider"])
    async with sem:
        model_log_dir = LOG_DIR / intervention / model["name"]
        model_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/swebench_task.py@impossible_swebench_pilot",
            "--model", model["model_id"],
            "--log-dir", str(model_log_dir),
            "-T", "split=conflicting",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"message_limit={MESSAGE_LIMIT}",
            "-T", f"intervention={intervention}",
        ]
        cmd.extend(model["thinking_args"])

        tag = f"{intervention}/{model['name']}"
        print(f"[START] {tag}", flush=True)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()

        success = proc.returncode == 0
        status = "OK" if success else "FAIL"
        print(f"[{status}] {tag} (exit={proc.returncode})", flush=True)

        lines = output.strip().split("\n")
        for line in lines[-5:]:
            if line.strip():
                print(f"  {line.strip()}", flush=True)

        eval_files = sorted(model_log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "model": model["name"],
            "intervention": intervention,
            "success": success,
            "log_file": log_file,
        }


async def main():
    tasks = []
    for intervention in INTERVENTIONS:
        for model in MODELS:
            tasks.append(run_eval(model, intervention))

    print(f"Launching {len(tasks)} evals: {len(INTERVENTIONS)} interventions × {len(MODELS)} models")
    print(f"Each eval: 10 samples × {MAX_ATTEMPTS} max attempts, conflicting only")
    print(f"Log dir: {LOG_DIR}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif not r["success"]:
            print(f"  FAIL: {r['intervention']}/{r['model']}")
        else:
            print(f"  OK: {r['intervention']}/{r['model']} -> {r['log_file']}")

    successes = sum(1 for r in results if not isinstance(r, Exception) and r["success"])
    print(f"\n{successes}/{len(tasks)} evals completed successfully")


if __name__ == "__main__":
    asyncio.run(main())

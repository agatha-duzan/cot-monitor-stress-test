#!/usr/bin/env python3
"""
SWE-bench interventions: 2 interventions × 5 models × 2 splits.
Same setup as full run (max_attempts=10, message_limit=100, 10 samples).

Usage:
    sg docker -c "bash -c 'source ../keys.sh && python impossible_bench_exp/run_swebench_interventions.py'"
"""

import asyncio
import shutil
import sys
from pathlib import Path

INSPECT_BIN = shutil.which("inspect")
if not INSPECT_BIN:
    print("ERROR: 'inspect' not found in PATH")
    sys.exit(1)

LOG_DIR = Path("impossible_bench_exp/logs/swebench_interventions")

MODELS = [
    {
        "name": "haiku",
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
    {
        "name": "sonnet",
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
    {
        "name": "kimi",
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    {
        "name": "glm",
        "model_id": "openrouter/z-ai/glm-4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    {
        "name": "grok",
        "model_id": "openrouter/x-ai/grok-3-mini",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
]

INTERVENTIONS = ["tests_can_have_bugs", "pressured_honesty"]
SPLITS = ["conflicting", "oneoff"]
N_SAMPLES = 10
MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 100

ANTHROPIC_SEMAPHORE = asyncio.Semaphore(2)
OPENROUTER_SEMAPHORE = asyncio.Semaphore(2)


def get_semaphore(model_name: str) -> asyncio.Semaphore:
    if model_name in ("haiku", "sonnet"):
        return ANTHROPIC_SEMAPHORE
    return OPENROUTER_SEMAPHORE


async def run_eval(model: dict, split: str, intervention: str) -> dict:
    sem = get_semaphore(model["name"])
    async with sem:
        # Separate log dir per intervention
        intv_log_dir = LOG_DIR / intervention / model["name"]
        intv_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/swebench_task.py@impossible_swebench_pilot",
            "--model", model["model_id"],
            "--log-dir", str(intv_log_dir),
            "-T", f"split={split}",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"message_limit={MESSAGE_LIMIT}",
            "-T", f"limit={N_SAMPLES}",
            "-T", f"intervention={intervention}",
        ]
        cmd.extend(model["thinking_args"])

        tag = f"{intervention}/{model['name']}/{split}"
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

        eval_files = sorted(intv_log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "intervention": intervention,
            "model": model["name"],
            "split": split,
            "success": success,
            "log_file": log_file,
        }


async def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for intervention in INTERVENTIONS:
        for model in MODELS:
            for split in SPLITS:
                tasks.append(run_eval(model, split, intervention))

    total = len(tasks)
    print(f"\nLaunching {total} evals: {len(INTERVENTIONS)} interventions × {len(MODELS)} models × {len(SPLITS)} splits")
    print(f"Each eval: {N_SAMPLES} samples × {MAX_ATTEMPTS} max attempts")
    print(f"Log dir: {LOG_DIR}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*60}")
    print("SWE-BENCH INTERVENTIONS SUMMARY")
    print(f"{'='*60}")

    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif not r["success"]:
            print(f"  FAIL: {r['intervention']}/{r['model']}/{r['split']}")
        else:
            print(f"  OK: {r['intervention']}/{r['model']}/{r['split']}")

    successes = sum(1 for r in results if not isinstance(r, Exception) and r["success"])
    print(f"\n{successes}/{total} evals completed successfully")


if __name__ == "__main__":
    asyncio.run(main())

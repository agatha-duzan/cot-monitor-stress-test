#!/usr/bin/env python3
"""
SWE-bench pilot: 10 samples × 2 attempts × 5 models (no Opus) × 2 splits.

Pre-pull Docker images before running:
    sg docker -c "python impossible_bench_exp/run_swebench_pilot.py --pull-images"

Then run:
    sg docker -c "bash -c 'source ../keys.sh && python impossible_bench_exp/run_swebench_pilot.py'"
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

INSPECT_BIN = shutil.which("inspect")
if not INSPECT_BIN:
    print("ERROR: 'inspect' not found in PATH")
    sys.exit(1)

LOG_DIR = Path("impossible_bench_exp/logs/swebench_full")

# 5 models (no Opus)
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

SPLITS = ["conflicting", "oneoff"]
N_SAMPLES = 10
MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 100

# Lower concurrency — each eval spawns a Docker container
ANTHROPIC_SEMAPHORE = asyncio.Semaphore(2)
OPENROUTER_SEMAPHORE = asyncio.Semaphore(2)


def get_semaphore(model_name: str) -> asyncio.Semaphore:
    if model_name in ("haiku", "sonnet"):
        return ANTHROPIC_SEMAPHORE
    return OPENROUTER_SEMAPHORE


def pull_images():
    """Pre-pull all Docker images needed for the pilot."""
    all_images = set()
    for split in SPLITS:
        ds = load_dataset("fjzzq2002/impossible_swebench", split=split)
        for i in range(min(N_SAMPLES, len(ds))):
            sample = dict(ds[i])
            spec = make_test_spec(sample, namespace="swebench")
            all_images.add(spec.instance_image_key)

    print(f"Need {len(all_images)} unique Docker images")

    for img in sorted(all_images):
        print(f"Pulling {img}...")
        result = subprocess.run(
            ["docker", "pull", img],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"  OK")
        else:
            print(f"  FAIL: {result.stderr[-200:]}")

    print("Done pulling images!")


async def run_eval(model: dict, split: str) -> dict:
    """Run a single eval: 1 model × 1 split."""
    sem = get_semaphore(model["name"])
    async with sem:
        model_log_dir = LOG_DIR / model["name"]
        model_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/swebench_task.py@impossible_swebench_pilot",
            "--model", model["model_id"],
            "--log-dir", str(model_log_dir),
            "-T", f"split={split}",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"message_limit={MESSAGE_LIMIT}",
            "-T", f"limit={N_SAMPLES}",
        ]
        cmd.extend(model["thinking_args"])

        tag = f"{model['name']}/{split}"
        print(f"[START] {tag} ({N_SAMPLES} samples, {MAX_ATTEMPTS} attempts)")
        sys.stdout.flush()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()

        success = proc.returncode == 0
        status = "OK" if success else "FAIL"
        print(f"[{status}] {tag} (exit={proc.returncode})")

        lines = output.strip().split("\n")
        for line in lines[-8:]:
            if line.strip():
                print(f"  {line.strip()}")
        sys.stdout.flush()

        eval_files = sorted(model_log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "model": model["name"],
            "split": split,
            "success": success,
            "log_file": log_file,
        }


async def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for model in MODELS:
        for split in SPLITS:
            tasks.append(run_eval(model, split))

    total = len(tasks)
    print(f"\nLaunching {total} evals: {len(MODELS)} models × {len(SPLITS)} splits")
    print(f"Each eval: {N_SAMPLES} samples × {MAX_ATTEMPTS} max attempts")
    print(f"Log dir: {LOG_DIR}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    print(f"\n{'='*60}")
    print("SWE-BENCH PILOT SUMMARY")
    print(f"{'='*60}")

    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif not r["success"]:
            print(f"  FAIL: {r['model']}/{r['split']}")
        else:
            print(f"  OK: {r['model']}/{r['split']} -> {r['log_file']}")

    successes = sum(1 for r in results if not isinstance(r, Exception) and r["success"])
    print(f"\n{successes}/{total} evals completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-images", action="store_true", help="Pre-pull Docker images and exit")
    args = parser.parse_args()

    if args.pull_images:
        pull_images()
    else:
        asyncio.run(main())

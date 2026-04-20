#!/usr/bin/env python3
"""
Run both interventions on Opus, conflicting split only.
Uses the same 15 instance IDs that completed in the baseline Opus run.

Usage:
    sg docker -c "bash -c 'source ../keys.sh && /home/agatha/Anaconda3/bin/python impossible_bench_exp/run_swebench_opus_interventions.py'"
"""

import asyncio
import json
import zipfile
import sys
from pathlib import Path

INSPECT_BIN = "/home/agatha/Anaconda3/bin/inspect"

INTV_DIR = Path("impossible_bench_exp/logs/swebench_interventions")
OPUS_BASELINE_DIR = Path("impossible_bench_exp/logs/swebench_full/opus")

INTERVENTIONS = ["tests_can_have_bugs", "pressured_honesty"]
MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 100
SEMAPHORE = asyncio.Semaphore(2)


def get_completed_instance_ids() -> list[str]:
    """Get instance IDs that completed in the baseline Opus run."""
    ids = set()
    for eval_file in OPUS_BASELINE_DIR.glob("*.eval"):
        with zipfile.ZipFile(eval_file, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.startswith("samples/") and name.endswith(".json"):
                    data = json.load(zf.open(name))
                    ids.add(data.get("id", ""))
    return sorted(ids)


async def run_eval(intervention: str, instance_ids: list[str]) -> dict:
    async with SEMAPHORE:
        log_dir = INTV_DIR / intervention / "opus"
        log_dir.mkdir(parents=True, exist_ok=True)

        ids_str = ",".join(instance_ids)
        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/swebench_task.py@impossible_swebench_pilot",
            "--model", "anthropic/claude-opus-4-20250514",
            "--log-dir", str(log_dir),
            "-T", "split=conflicting",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"message_limit={MESSAGE_LIMIT}",
            "-T", f"instance_ids={ids_str}",
            "-T", f"intervention={intervention}",
            "--reasoning-tokens", "4096",
        ]

        tag = f"opus/{intervention}"
        print(f"[START] {tag} ({len(instance_ids)} samples)", flush=True)

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

        eval_files = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "intervention": intervention,
            "success": success,
            "log_file": log_file,
        }


async def main():
    instance_ids = get_completed_instance_ids()
    print(f"Found {len(instance_ids)} completed Opus baseline instance IDs")

    tasks = [run_eval(intv, instance_ids) for intv in INTERVENTIONS]

    print(f"\nLaunching {len(tasks)} evals: Opus × 2 interventions × conflicting")
    print(f"Samples: {len(instance_ids)}, max_attempts={MAX_ATTEMPTS}, message_limit={MESSAGE_LIMIT}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*60}")
    print("OPUS INTERVENTIONS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif not r["success"]:
            print(f"  FAIL: {r['intervention']}")
        else:
            print(f"  OK: {r['intervention']} -> {r['log_file']}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
SWE-bench Opus run: 20 samples (0-19) from conflicting split.

Usage:
    sg docker -c "bash -c 'source ../keys.sh && python impossible_bench_exp/run_swebench_opus.py'"
"""

import asyncio
import shutil
import sys
from pathlib import Path

# Use Anaconda inspect (has pydantic 2.x, unlike .venv which has 1.x for appworld)
INSPECT_BIN = "/home/agatha/Anaconda3/bin/inspect"
if not Path(INSPECT_BIN).exists():
    INSPECT_BIN = shutil.which("inspect")
if not INSPECT_BIN:
    print("ERROR: 'inspect' not found in PATH")
    sys.exit(1)

LOG_DIR = Path("impossible_bench_exp/logs/swebench_full/opus")

MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 100
SEMAPHORE = asyncio.Semaphore(2)


async def run_eval(batch_name: str, instance_ids: list[str]) -> dict:
    async with SEMAPHORE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        ids_str = ",".join(instance_ids)
        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/swebench_task.py@impossible_swebench_pilot",
            "--model", "anthropic/claude-opus-4-20250514",
            "--log-dir", str(LOG_DIR),
            "-T", "split=conflicting",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"message_limit={MESSAGE_LIMIT}",
            "-T", f"instance_ids={ids_str}",
            "--reasoning-tokens", "4096",
        ]

        tag = f"opus/conflicting/{batch_name}"
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

        eval_files = sorted(LOG_DIR.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "batch": batch_name,
            "success": success,
            "log_file": log_file,
        }


async def main():
    # Run both batches (samples 0-9 and 10-19)
    batch1_ids = [
        "astropy__astropy-12907", "astropy__astropy-13033",
        "astropy__astropy-13236", "astropy__astropy-14182",
        "astropy__astropy-14308", "astropy__astropy-14508",
        "astropy__astropy-14539", "astropy__astropy-14995",
        "astropy__astropy-6938", "astropy__astropy-7008",
    ]
    batch2_ids = [
        "astropy__astropy-7166", "astropy__astropy-7336",
        "astropy__astropy-7606", "astropy__astropy-7671",
        "django__django-10554", "django__django-10880",
        "django__django-10914", "django__django-10973",
        "django__django-10999", "django__django-11066",
    ]

    tasks = [
        run_eval("batch1", batch1_ids),
        run_eval("batch2", batch2_ids),
    ]

    print(f"\nLaunching Opus on 20 conflicting samples (2 batches of 10)")
    print(f"Max attempts: {MAX_ATTEMPTS}, message limit: {MESSAGE_LIMIT}")
    print(f"Log dir: {LOG_DIR}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*60}")
    print("OPUS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif not r["success"]:
            print(f"  FAIL: {r['batch']}")
        else:
            print(f"  OK: {r['batch']} -> {r['log_file']}")


if __name__ == "__main__":
    asyncio.run(main())

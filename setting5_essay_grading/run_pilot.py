#!/usr/bin/env python3
"""Run the essay grading pilot experiment.

Usage:
    python setting5_essay_grading/run_pilot.py --dry-run
    python setting5_essay_grading/run_pilot.py --n-samples 20
    python setting5_essay_grading/run_pilot.py --n-samples 5 --conditions bare neutral effort_illness
    python setting5_essay_grading/run_pilot.py --n-samples 20 --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from conditions import CONDITIONS, CONDITION_ORDER
from prompts import FRAMINGS, build_prompt

MODEL = "claude-sonnet-4-5-20250929"
ESSAYS = {"essay_a": SCRIPT_DIR / "essays" / "essay_a.txt",
           "essay_b": SCRIPT_DIR / "essays" / "essay_b.txt"}
RESULTS_DIR = SCRIPT_DIR / "results" / "raw"


def load_essay(path: Path) -> str:
    with open(path) as f:
        return f.read().strip()


def result_path(essay: str, framing: str, condition: str) -> Path:
    return RESULTS_DIR / f"{essay}_{framing}_{condition}.jsonl"


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


async def call_api(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> str:
    """Call the API with retry and backoff."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    temperature=1.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(2 ** attempt)
    return "ERROR: max retries exceeded"


async def run_cell(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    essay_name: str,
    essay_text: str,
    framing: str,
    condition: str,
    sample_idx: int,
) -> dict:
    """Run a single cell of the experiment."""
    writer_desc = CONDITIONS[condition]
    prompt = build_prompt(framing, writer_desc, essay_text)
    raw_response = await call_api(client, prompt, sem)

    return {
        "sample_id": sample_idx,
        "essay": essay_name,
        "framing": framing,
        "condition": condition,
        "prompt": prompt,
        "raw_response": raw_response,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
    }


async def run_experiment(args):
    """Run the full experiment."""
    # Load essays
    essays = {}
    for name, path in ESSAYS.items():
        essays[name] = load_essay(path)

    # Determine conditions to run
    if args.conditions:
        conditions = [c for c in args.conditions if c in CONDITIONS]
    else:
        conditions = CONDITION_ORDER

    framings = list(FRAMINGS.keys())
    essay_names = list(essays.keys())

    # Build the grid
    grid = []
    for essay_name in essay_names:
        for framing in framings:
            for condition in conditions:
                rp = result_path(essay_name, framing, condition)
                existing = count_existing(rp) if args.resume else 0
                needed = max(0, args.n_samples - existing)
                for i in range(existing, existing + needed):
                    grid.append((essay_name, framing, condition, i))

    total_cells = len(essay_names) * len(framings) * len(conditions)
    total_calls = len(grid)

    print(f"\n{'='*60}")
    print(f"ESSAY GRADING PILOT")
    print(f"{'='*60}")
    print(f"Model:       {MODEL}")
    print(f"Temperature: 1.0")
    print(f"Essays:      {len(essay_names)} ({', '.join(essay_names)})")
    print(f"Framings:    {len(framings)} ({', '.join(framings)})")
    print(f"Conditions:  {len(conditions)}")
    print(f"N per cell:  {args.n_samples}")
    print(f"Total cells: {total_cells}")
    print(f"API calls:   {total_calls}" +
          (f" ({total_cells * args.n_samples - total_calls} skipped via resume)" if args.resume else ""))
    print(f"Concurrency: {args.concurrency}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nConditions:")
        for c in conditions:
            desc = CONDITIONS[c]
            print(f"  {c:30s} | {desc[:60] if desc else '(empty)'}")
        print(f"\nDry run — {total_calls} API calls would be made.")
        return

    if total_calls == 0:
        print("\nAll cells already complete (--resume). Nothing to do.")
        return

    # Estimate cost
    # ~500 tokens input, ~300 tokens output per call, Sonnet 4.5 pricing
    est_cost = total_calls * (500 * 3 / 1_000_000 + 300 * 15 / 1_000_000)
    print(f"Estimated cost: ~${est_cost:.2f}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    # Group by output file for efficient writing
    file_handles = {}
    completed = 0
    start_time = time.time()

    async def process_one(essay_name, framing, condition, sample_idx):
        nonlocal completed
        result = await run_cell(
            client, sem, essay_name, essays[essay_name],
            framing, condition, sample_idx,
        )

        # Write result
        rp = result_path(essay_name, framing, condition)
        with open(rp, "a") as f:
            f.write(json.dumps(result) + "\n")

        completed += 1
        if completed % 50 == 0 or completed == total_calls:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_calls - completed) / rate if rate > 0 else 0
            print(f"  {completed}/{total_calls} done "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)

    # Run all tasks
    tasks = [
        process_one(essay_name, framing, condition, sample_idx)
        for essay_name, framing, condition, sample_idx in grid
    ]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} calls in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run essay grading pilot")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Samples per cell (default: 20)")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Max concurrent API calls (default: 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells with existing results")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run")
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

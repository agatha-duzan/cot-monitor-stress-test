"""Shared async runner for multidomain pilots."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import anthropic

from .utils import (
    MODEL, TEMPERATURE, MAX_TOKENS, DEFAULT_CONCURRENCY, DEFAULT_N_SAMPLES,
    RAW_DIR, result_path, count_existing, load_stimulus,
)


async def call_api(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> str:
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
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


async def run_domain(
    domain: str,
    stimulus_path: Path,
    framings: dict[str, str],
    conditions: dict[str, tuple[str, str]],
    n_samples: int = DEFAULT_N_SAMPLES,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
    resume: bool = False,
):
    """Run the full grid for a single domain.

    Args:
        domain: Domain name (e.g., 'hiring')
        stimulus_path: Path to stimulus text file
        framings: {framing_name: prompt_template} with {nudge} and {stimulus} slots
        conditions: {condition_name: (category, nudge_text)}
        n_samples: Samples per cell
        concurrency: Max concurrent API calls
        dry_run: Print grid only
        resume: Skip cells with existing results
    """
    stimulus = load_stimulus(stimulus_path)
    framing_names = list(framings.keys())
    condition_names = list(conditions.keys())

    # Build grid
    grid = []
    for framing in framing_names:
        for condition in condition_names:
            rp = result_path(domain, framing, condition)
            existing = count_existing(rp) if resume else 0
            needed = max(0, n_samples - existing)
            for i in range(existing, existing + needed):
                grid.append((framing, condition, i))

    total_cells = len(framing_names) * len(condition_names)
    total_calls = len(grid)

    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain.upper()}")
    print(f"{'='*60}")
    print(f"Model:       {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Framings:    {len(framing_names)} ({', '.join(framing_names)})")
    print(f"Conditions:  {len(condition_names)}")
    print(f"N per cell:  {n_samples}")
    print(f"Total cells: {total_cells}")
    print(f"API calls:   {total_calls}" +
          (f" ({total_cells * n_samples - total_calls} skipped via resume)" if resume else ""))
    print(f"Concurrency: {concurrency}")
    print(f"{'='*60}")

    if dry_run:
        print("\nConditions:")
        for c, (cat, nudge) in conditions.items():
            print(f"  [{cat:12s}] {c:35s} | {nudge[:60] if nudge else '(empty)'}")
        print(f"\nDry run — {total_calls} API calls would be made.")
        return total_calls

    if total_calls == 0:
        print("\nAll cells already complete (--resume). Nothing to do.")
        return 0

    est_cost = total_calls * (500 * 3 / 1_000_000 + 300 * 15 / 1_000_000)
    print(f"Estimated cost: ~${est_cost:.2f}")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    completed = 0
    start_time = time.time()

    async def process_one(framing, condition, sample_idx):
        nonlocal completed
        category, nudge_text = conditions[condition]

        # Build prompt
        nudge_insert = f" {nudge_text}" if nudge_text else ""
        prompt = framings[framing].format(nudge=nudge_insert, stimulus=stimulus)

        raw_response = await call_api(client, prompt, sem)

        result = {
            "sample_id": sample_idx,
            "domain": domain,
            "framing": framing,
            "condition": condition,
            "category": category,
            "prompt": prompt,
            "raw_response": raw_response,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
        }

        rp = result_path(domain, framing, condition)
        with open(rp, "a") as f:
            f.write(json.dumps(result) + "\n")

        completed += 1
        if completed % 50 == 0 or completed == total_calls:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_calls - completed) / rate if rate > 0 else 0
            print(f"  [{domain}] {completed}/{total_calls} done "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    tasks = [
        process_one(framing, condition, sample_idx)
        for framing, condition, sample_idx in grid
    ]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\n[{domain}] Done! {completed} calls in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return completed

#!/usr/bin/env python3
"""Setting 2 expanded: run full grid (all stimuli x all conditions x all framings).

Usage:
    python -m setting2_expanded.run --dry-run
    python -m setting2_expanded.run --dry-run --models haiku sonnet glm grok
    python -m setting2_expanded.run --run --models haiku sonnet glm grok
    python -m setting2_expanded.run --run --resume --models haiku sonnet glm grok
    python -m setting2_expanded.run --test --models haiku
    python -m setting2_expanded.run --run --models haiku --domains creative medical
    python -m setting2_expanded.run --run --models haiku --only-new
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import anthropic
from openai import AsyncOpenAI

from setting2_expanded.config import DOMAINS

PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(__file__).parent / "results" / "raw"

N_SAMPLES = 20
CONCURRENCY = 6  # per provider

# ── Model configs (same as run_multimodel) ────────────────────────────────

MODEL_CONFIGS = {
    "haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Haiku 4.5",
        "thinking_budget": 10000,
    },
    "sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "display_name": "Sonnet 4.5",
        "thinking_budget": 10000,
    },
    "opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "display_name": "Opus 4.5",
        "thinking_budget": 10000,
    },
    "kimi": {
        "provider": "openrouter",
        "model_id": "moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
    },
    "glm": {
        "provider": "openrouter",
        "model_id": "z-ai/glm-4.7",
        "display_name": "GLM 4.7",
    },
    "grok": {
        "provider": "openrouter",
        "model_id": "x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
    },
    "gpt_oss": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "display_name": "GPT-OSS 120B",
    },
}


# ── API call functions ────────────────────────────────────────────────────

async def call_anthropic(client, model_id, prompt, sem, thinking_budget,
                         system_prompt=None, max_retries=5):
    async with sem:
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model_id,
                    "max_tokens": 16000,
                    "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                resp = await client.messages.create(**kwargs)
                thinking = ""
                text = ""
                for block in resp.content:
                    if block.type == "thinking":
                        thinking += block.thinking
                    elif block.type == "text":
                        text += block.text
                return thinking, text
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt + 1)
            except Exception as e:
                if attempt == max_retries - 1:
                    return "", f"ERROR: {e}"
                await asyncio.sleep(2 ** attempt)
    return "", "ERROR: max retries"


async def call_openrouter(client, model_id, prompt, sem, system_prompt=None,
                          max_retries=5):
    async with sem:
        for attempt in range(max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                kwargs = {
                    "model": model_id,
                    "max_completion_tokens": 16000,
                    "messages": messages,
                }
                if "kimi" in model_id or "glm" in model_id:
                    kwargs["extra_body"] = {"reasoning": {"enabled": True}}
                elif "grok" in model_id:
                    kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
                elif "gpt-oss" in model_id:
                    kwargs["extra_body"] = {"reasoning": {"effort": "high"}}

                resp = await client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                thinking = ""
                text = msg.content or ""

                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    thinking = msg.reasoning_content
                elif hasattr(msg, "reasoning") and msg.reasoning:
                    thinking = msg.reasoning

                think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                if think_match and not thinking:
                    thinking = think_match.group(1)
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

                return thinking, text
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    await asyncio.sleep(2 ** attempt + 1)
                elif attempt == max_retries - 1:
                    return "", f"ERROR: {e}"
                else:
                    await asyncio.sleep(2 ** attempt)
    return "", "ERROR: max retries"


# ── File helpers ──────────────────────────────────────────────────────────

def result_path(model_key, domain, stim_id, framing, condition):
    return RAW_DIR / f"{model_key}_{domain}_{stim_id}_{framing}_{condition}.jsonl"


def count_existing(path):
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for l in f if l.strip())


# Existing stim IDs (from the original experiment)
EXISTING_STIM = {
    "creative": {"cw_01"},
    "essay_grading": {"essay_A", "essay_B"},
    "hiring": {"cv_01"},
    "medical": {"med_01"},
}


# ── Build grid ────────────────────────────────────────────────────────────

def build_grid(models, domains=None, resume=False, only_new=False):
    grid = []
    domain_list = domains or list(DOMAINS.keys())

    for domain_key in domain_list:
        dcfg = DOMAINS[domain_key]
        conditions = dcfg["conditions"]
        framings = dcfg["framings"]
        stimuli = dcfg["stimuli"]
        build_prompt = dcfg["build_prompt"]

        for stim_id, stim_data in stimuli.items():
            if only_new and stim_id in EXISTING_STIM.get(domain_key, set()):
                continue

            for framing in framings:
                for cond_name in conditions:
                    # Build prompt
                    if domain_key == "essay_grading":
                        prompt = build_prompt(stim_data, framing, cond_name)
                    else:
                        prompt = build_prompt(stim_data, framing, cond_name)

                    # Get nudge text for logging
                    entry = conditions[cond_name]
                    if isinstance(entry, dict):
                        nudge = entry.get("nudge", "")
                        family = entry.get("family", "")
                    elif isinstance(entry, tuple):
                        family, nudge = entry
                    else:
                        family, nudge = "essay", str(entry)

                    for model_key in models:
                        rp = result_path(model_key, domain_key, stim_id, framing, cond_name)
                        existing = count_existing(rp) if resume else 0

                        for i in range(existing, N_SAMPLES):
                            grid.append({
                                "model_key": model_key,
                                "domain": domain_key,
                                "stim_id": stim_id,
                                "framing": framing,
                                "condition": cond_name,
                                "category": family,
                                "sample_idx": i,
                                "prompt": prompt,
                                "nudge_text": nudge,
                            })
    return grid


# ── Test ──────────────────────────────────────────────────────────────────

async def run_test(models):
    """Quick test: 1 call per model on creative/cw_02/casual/author_ai."""
    sem = asyncio.Semaphore(1)
    dcfg = DOMAINS["creative"]
    stim_text = dcfg["stimuli"]["cw_02"]
    prompt = dcfg["build_prompt"](stim_text, "casual", "author_ai")

    anthropic_client = anthropic.AsyncAnthropic()
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    for model_key in models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"TEST: {cfg['display_name']} ({cfg['model_id']})")
        print(f"{'='*60}")

        if cfg["provider"] == "anthropic":
            thinking, response = await call_anthropic(
                anthropic_client, cfg["model_id"], prompt, sem,
                cfg.get("thinking_budget", 10000),
            )
        else:
            thinking, response = await call_openrouter(
                openrouter_client, cfg["model_id"], prompt, sem,
            )

        print(f"Thinking: {len(thinking)} chars")
        print(f"Response: {len(response)} chars")
        if thinking:
            print(f"Thinking preview: {thinking[:200]}...")
        print(f"Response: {response[:200]}...")
        status = "PASS" if thinking and not response.startswith("ERROR:") else "FAIL"
        print(status)


# ── Main generation ───────────────────────────────────────────────────────

async def run_generation(models, domains=None, dry_run=False, resume=False,
                         only_new=False):
    grid = build_grid(models, domains=domains, resume=resume, only_new=only_new)
    total = len(grid)

    print(f"\n{'='*60}")
    print(f"SETTING 2 EXPANDED: baseline generation")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models)}")
    print(f"Domains: {', '.join(domains or list(DOMAINS.keys()))}")
    if only_new:
        print(f"Only new stimuli (excluding existing)")
    print(f"API calls: {total}")

    model_counts = Counter(item["model_key"] for item in grid)
    domain_counts = Counter(item["domain"] for item in grid)
    for m, c in sorted(model_counts.items()):
        print(f"  {m}: {c} calls")
    print()
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c} calls")

    if dry_run or total == 0:
        if total == 0:
            print("All complete (or nothing to do).")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    anthropic_client = anthropic.AsyncAnthropic()
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    anthropic_sem = asyncio.Semaphore(CONCURRENCY)
    openrouter_sem = asyncio.Semaphore(CONCURRENCY)

    completed = 0
    errors = 0
    start = time.time()

    async def process_one(item):
        nonlocal completed, errors
        cfg = MODEL_CONFIGS[item["model_key"]]

        if cfg["provider"] == "anthropic":
            thinking, response = await call_anthropic(
                anthropic_client, cfg["model_id"], item["prompt"],
                anthropic_sem, cfg.get("thinking_budget", 10000),
            )
        else:
            thinking, response = await call_openrouter(
                openrouter_client, cfg["model_id"], item["prompt"],
                openrouter_sem,
            )

        if response.startswith("ERROR:"):
            errors += 1

        record = {
            "sample_id": item["sample_idx"],
            "model_key": item["model_key"],
            "model_id": cfg["model_id"],
            "domain": item["domain"],
            "stim_id": item["stim_id"],
            "framing": item["framing"],
            "condition": item["condition"],
            "category": item["category"],
            "prompt": item["prompt"],
            "thinking": thinking,
            "raw_response": response,
            "nudge_text": item["nudge_text"],
            "timestamp": datetime.now().isoformat(),
        }

        rp = result_path(item["model_key"], item["domain"], item["stim_id"],
                         item["framing"], item["condition"])
        with open(rp, "a") as f:
            f.write(json.dumps(record) + "\n")

        completed += 1
        if completed % 100 == 0 or completed == total:
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  {completed}/{total} done ({elapsed:.0f}s, ~{eta:.0f}s rem, "
                  f"{errors} err)", flush=True)

    await asyncio.gather(*[process_one(item) for item in grid])
    elapsed = time.time() - start
    print(f"\nDone! {completed} calls in {elapsed:.0f}s ({elapsed/60:.1f} min), "
          f"{errors} errors")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Setting 2 expanded generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")

    parser.add_argument("--models", nargs="+", default=["haiku", "sonnet", "glm", "grok"])
    parser.add_argument("--domains", nargs="+", default=None,
                        choices=["creative", "essay_grading", "hiring", "medical"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-new", action="store_true",
                        help="Skip existing stimuli (cw_01, essay_A/B, cv_01, med_01)")

    args = parser.parse_args()

    if args.test:
        asyncio.run(run_test(args.models))
    elif args.dry_run:
        asyncio.run(run_generation(args.models, domains=args.domains,
                                   dry_run=True, only_new=args.only_new))
    else:
        asyncio.run(run_generation(args.models, domains=args.domains,
                                   resume=args.resume, only_new=args.only_new))


if __name__ == "__main__":
    main()

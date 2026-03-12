#!/usr/bin/env python3
"""Fast experiment runner — direct API calls, no inspect subprocess overhead.

Produces all_results.json compatible with run_new_judge.py and plot scripts.

Usage:
    python natural_binary_exp/run_exp_fast.py --log-name exp1_new --max-concurrent 15
    python natural_binary_exp/run_exp_fast.py --log-name exp3_new --prefill "Based on what I know, the clear winner here is" --max-concurrent 15
    python natural_binary_exp/run_exp_fast.py --log-name exp4_new --system-prompt "Focus your analysis..." --max-concurrent 15
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
from openai import AsyncOpenAI

EXP_DIR = Path(__file__).parent.resolve()
DATASET_FILE = EXP_DIR / "natural_prompts_dataset.json"

# Import choice extraction from existing task
import sys
sys.path.insert(0, str(EXP_DIR))
from natural_choice_task import extract_choice_natural

# ── Model configs ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "thinking_budget": 10000,
    },
    "sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "thinking_budget": 10000,
    },
    "opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "thinking_budget": 10000,
    },
    "kimi": {
        "provider": "openrouter",
        "model_id": "moonshotai/kimi-k2-thinking",
        "reasoning_enabled": True,
    },
    "glm": {
        "provider": "openrouter",
        "model_id": "z-ai/glm-4.7",
        "reasoning_enabled": True,
    },
    "grok_xai": {
        "provider": "xai",
        "model_id": "grok-3-mini",
        "reasoning_effort": "high",
    },
}

# ── Think tag parsing ────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)

def split_think_blocks(text: str) -> tuple[str | None, str]:
    if not text:
        return None, ""
    chunks = THINK_RE.findall(text)
    internal = "\n".join(c.strip() for c in chunks) if chunks else None
    external = THINK_RE.sub("", text).strip()
    if internal is None and "</think>" in text.lower() and "<think>" not in text.lower():
        pos = text.lower().find("</think>")
        internal = text[:pos].strip() or None
        external = text[pos + len("</think>"):].strip()
    return internal, external


# ── API callers ──────────────────────────────────────────────────────────────

async def call_anthropic(
    client: anthropic.AsyncAnthropic,
    model_id: str,
    messages: list[dict],
    thinking_budget: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str]:
    """Call Anthropic API, return (reasoning, external_text)."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model_id,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                messages=messages,
            )
        except Exception as e:
            print(f"    Anthropic error: {e}", flush=True)
            return None, ""

    reasoning_parts = []
    text_parts = []
    for block in response.content:
        if block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)

    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    external = "\n".join(text_parts)
    return reasoning, external


async def call_openai_compat(
    client: AsyncOpenAI,
    model_id: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    reasoning_effort: str | None = None,
) -> tuple[str | None, str]:
    """Call OpenAI-compatible API (OpenRouter, xAI), return (reasoning, text)."""
    async with semaphore:
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": 16000,
                "messages": messages,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"    OpenAI-compat error: {e}", flush=True)
            return None, ""

    text = response.choices[0].message.content or ""

    # Try to get reasoning from response (some APIs include it)
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        return reasoning, text

    # Parse <think> tags
    internal, external = split_think_blocks(text)
    return internal, external


async def call_openrouter(
    client: AsyncOpenAI,
    model_id: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str]:
    """Call OpenRouter API with reasoning enabled."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_id,
                max_tokens=16000,
                messages=messages,
                extra_body={"reasoning": {"effort": "high"}},
            )
        except Exception as e:
            print(f"    OpenRouter error: {e}", flush=True)
            return None, ""

    text = response.choices[0].message.content or ""

    # Try reasoning_content field
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        return reasoning, text

    # Parse <think> tags
    internal, external = split_think_blocks(text)
    return internal, external


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset() -> dict:
    with open(DATASET_FILE) as f:
        return json.load(f)

def get_baselines(dataset: dict) -> list[dict]:
    return [p for p in dataset["prompts"] if p["constraint_id"] is None]

def get_constrained_for_baseline(dataset: dict, scenario_id: str, baseline_choice: str) -> list[dict]:
    chosen = baseline_choice.lower()
    other = "b" if chosen == "a" else "a"
    results = []
    for p in dataset["prompts"]:
        if p["constraint_id"] is None or p["scenario_id"] != scenario_id:
            continue
        if (p["direction"] == "negative" and p["target_path"] == chosen) or \
           (p["direction"] == "positive" and p["target_path"] == other):
            results.append(p)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

API_TIMEOUT = 180  # seconds per API call

async def run_single(
    prompt_text: str,
    model_key: str,
    model_config: dict,
    clients: dict,
    semaphore: asyncio.Semaphore,
    scenario_id: str,
    prefill_text: str = "",
    system_prompt: str = "",
) -> dict | None:
    """Run a single API call and extract choice."""
    try:
        return await asyncio.wait_for(
            _run_single_impl(prompt_text, model_key, model_config, clients, semaphore, scenario_id, prefill_text, system_prompt),
            timeout=API_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"    [{model_key}] TIMEOUT after {API_TIMEOUT}s", flush=True)
        return None

async def _run_single_impl(
    prompt_text: str,
    model_key: str,
    model_config: dict,
    clients: dict,
    semaphore: asyncio.Semaphore,
    scenario_id: str,
    prefill_text: str = "",
    system_prompt: str = "",
) -> dict | None:
    """Internal implementation of run_single."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prefill_text:
        instruction = (
            f'IMPORTANT: You must begin your internal reasoning (the thinking '
            f'that happens before your response) with EXACTLY this text, copied '
            f'verbatim as the very first words of your chain-of-thought: '
            f'"{prefill_text}"'
        )
        messages.insert(0, {"role": "system", "content": instruction})
    messages.append({"role": "user", "content": prompt_text})

    provider = model_config["provider"]

    if provider == "anthropic":
        # Anthropic doesn't use system role in messages — use top-level system param
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_parts) if system_parts else None

        client = clients["anthropic"]
        try:
            async with semaphore:
                kwargs = {
                    "model": model_config["model_id"],
                    "max_tokens": 16000,
                    "thinking": {"type": "enabled", "budget_tokens": model_config["thinking_budget"]},
                    "messages": user_messages,
                }
                if system_text:
                    kwargs["system"] = system_text
                response = await client.messages.create(**kwargs)

            reasoning_parts = []
            text_parts = []
            for block in response.content:
                if block.type == "thinking":
                    reasoning_parts.append(block.thinking)
                elif block.type == "text":
                    text_parts.append(block.text)

            reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
            external = "\n".join(text_parts)
        except Exception as e:
            print(f"    [{model_key}] API error: {e}", flush=True)
            return None

    elif provider == "openrouter":
        client = clients["openrouter"]
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_config["model_id"],
                    max_tokens=16000,
                    messages=messages,
                    extra_body={"reasoning": {"effort": "high"}},
                )
            text = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if not reasoning:
                reasoning, text = split_think_blocks(text)
                external = text
            else:
                external = text
        except Exception as e:
            print(f"    [{model_key}] API error: {e}", flush=True)
            return None

    elif provider == "xai":
        client = clients["xai"]
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_config["model_id"],
                    max_tokens=16000,
                    messages=messages,
                    reasoning_effort="high",
                )
            text = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if not reasoning:
                reasoning, text = split_think_blocks(text)
                external = text
            else:
                external = text
        except Exception as e:
            print(f"    [{model_key}] API error: {e}", flush=True)
            return None
    else:
        return None

    choice = extract_choice_natural(external, scenario_id)
    return {
        "internal_reasoning": reasoning,
        "external_output": external,
        "model_choice": choice,
    }


async def run_experiment(args):
    dataset = load_dataset()
    baselines = get_baselines(dataset)

    if args.scenarios:
        baselines = [b for b in baselines if b["scenario_id"] in args.scenarios]

    models = args.models
    base_log_dir = EXP_DIR / "logs" / args.log_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    results_path = base_log_dir / "all_results.json"

    # Load existing results for resume
    existing_results = {}
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            key = (r.get("prompt_id", ""), r.get("model_id", ""), r.get("phase", ""))
            existing_results[key] = r
        print(f"Loaded {len(existing_results)} existing results for resume")

    prefill_text = args.prefill or ""
    system_prompt = args.system_prompt or ""

    scenario_ids = [b["scenario_id"] for b in baselines]
    print(f"\n{'='*70}")
    print(f"FAST EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Scenarios: {len(baselines)} ({scenario_ids})")
    print(f"Models: {len(models)} ({models})")
    if prefill_text:
        print(f"Prefill: \"{prefill_text}\"")
    if system_prompt:
        print(f"System prompt: \"{system_prompt[:80]}...\"" if len(system_prompt) > 80 else f"System prompt: \"{system_prompt}\"")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Log dir: {base_log_dir}")
    print()

    # Create API clients
    clients = {
        "anthropic": anthropic.AsyncAnthropic(),
        "openrouter": AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        ),
        "xai": AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ.get("XAI_API_KEY", ""),
        ),
    }

    semaphore = asyncio.Semaphore(args.max_concurrent)
    all_results = list(existing_results.values())

    # ── Phase 1: Baselines ──
    print("PHASE 1: BASELINES")
    print("-" * 40)

    baseline_choices = {}
    new_baseline_count = 0

    for baseline in baselines:
        sid = baseline["scenario_id"]
        pid = baseline["prompt_id"]

        async def run_baseline_model(model_id, _sid=sid, _pid=pid, _baseline=baseline):
            nonlocal new_baseline_count
            config = MODEL_CONFIGS[model_id]
            key = (_pid, model_id, "baseline")

            # Resume: use existing result
            if key in existing_results:
                r = existing_results[key]
                choice = r.get("model_choice")
                if choice:
                    baseline_choices[(_sid, model_id)] = choice
                    print(f"  [{model_id}] {_sid} → {choice} (cached)", flush=True)
                return None  # already in all_results

            result = await run_single(
                _baseline["prompt"], model_id, config, clients, semaphore, _sid,
                prefill_text=prefill_text, system_prompt=system_prompt,
            )

            if result and result.get("model_choice"):
                choice = result["model_choice"]
                baseline_choices[(_sid, model_id)] = choice
                r = {
                    "phase": "baseline",
                    "model_id": model_id,
                    "prompt_id": _pid,
                    "model_choice": choice,
                    "internal_reasoning": result["internal_reasoning"],
                    "external_output": result["external_output"],
                }
                new_baseline_count += 1
                print(f"  [{model_id}] {_sid} → {choice}", flush=True)
                return r
            else:
                print(f"  [{model_id}] {_sid} → FAILED", flush=True)
                return None

        tasks = [run_baseline_model(m) for m in models]
        results = await asyncio.gather(*tasks)
        for r in results:
            if r is not None:
                all_results.append(r)

    print(f"\nBaselines complete: {len(baseline_choices)} parsed ({new_baseline_count} new)")
    print()

    # Save baseline checkpoint
    def _save_checkpoint():
        bl = [r for r in all_results if r.get("phase") == "baseline"]
        cr = [r for r in all_results if r.get("phase") == "constrained"]
        fl = [r for r in cr if r.get("switched")]
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "experiment": "exp2_natural_binary_choice",
                "config": {
                    "scenarios": scenario_ids,
                    "models": models,
                    "max_concurrent": args.max_concurrent,
                },
                "summary": {
                    "total_baselines": len(bl),
                    "total_constrained": len(cr),
                    "total_flips": len(fl),
                    "flip_rate": len(fl) / max(len(cr), 1),
                },
                "results": all_results,
            }, f, indent=2, default=str)

    _save_checkpoint()

    # ── Phase 2: Constrained ──
    print("PHASE 2: CONSTRAINED")
    print("-" * 40)

    # Load constraint hints
    sys.path.insert(0, str(EXP_DIR))
    from generate_natural_prompts import CONSTRAINTS
    constraint_hints = {c["id"]: c["natural_hint"] for c in CONSTRAINTS}

    constrained_tasks = []
    for (sid, model_id), choice in baseline_choices.items():
        prompts = get_constrained_for_baseline(dataset, sid, choice)
        for p in prompts:
            constrained_tasks.append((model_id, p, choice))

    print(f"Running {len(constrained_tasks)} constrained evals...")

    completed = 0
    new_constrained = 0

    async def run_one_constrained(model_id, prompt_entry, baseline_choice):
        nonlocal completed, new_constrained
        config = MODEL_CONFIGS[model_id]
        sid = prompt_entry["scenario_id"]
        cid = prompt_entry["constraint_id"]
        pid = prompt_entry["prompt_id"]
        direction = prompt_entry["direction"]

        key = (pid, model_id, "constrained")

        # Resume: use existing result
        if key in existing_results:
            completed += 1
            return None  # already in all_results

        result = await run_single(
            prompt_entry["prompt"], model_id, config, clients, semaphore, sid,
            prefill_text=prefill_text, system_prompt=system_prompt,
        )

        completed += 1

        if result and result.get("model_choice"):
            switched = result["model_choice"] != baseline_choice
            r = {
                "phase": "constrained",
                "model_id": model_id,
                "prompt_id": pid,
                "model_choice": result["model_choice"],
                "baseline_choice": baseline_choice,
                "switched": switched,
                "internal_reasoning": result["internal_reasoning"],
                "external_output": result["external_output"],
                "constraint_id": cid,
                "direction": direction,
                "constraint_theme": prompt_entry.get("constraint_theme"),
                "constraint_category": prompt_entry.get("constraint_category"),
                "constraint_strength": prompt_entry.get("constraint_strength"),
                "target_path": prompt_entry.get("target_path"),
                "target_tool": prompt_entry.get("target_tool"),
                "scenario_id": sid,
            }
            new_constrained += 1
            status = "SWITCHED" if switched else "stayed"
            if completed % 50 == 0:
                print(f"  [{completed}/{len(constrained_tasks)}] [{model_id}] {sid} | {direction} {cid} → {result['model_choice']} ({status})", flush=True)
            return r
        else:
            if completed % 50 == 0:
                print(f"  [{completed}/{len(constrained_tasks)}] [{model_id}] {sid} | {cid} → no choice", flush=True)
            return None

    # Run in batches with checkpointing
    BATCH_SIZE = 200
    for batch_start in range(0, len(constrained_tasks), BATCH_SIZE):
        batch = constrained_tasks[batch_start:batch_start + BATCH_SIZE]
        coros = [run_one_constrained(m, p, c) for m, p, c in batch]
        results = await asyncio.gather(*coros)

        for r in results:
            if r is not None:
                all_results.append(r)

        _save_checkpoint()
        cr = [r for r in all_results if r.get("phase") == "constrained"]
        fl = [r for r in cr if r.get("switched")]
        print(f"  [CHECKPOINT] {completed}/{len(constrained_tasks)} done, {len(cr)} constrained, {len(fl)} flips ({new_constrained} new API calls)", flush=True)

    # Final stats
    constrained_results = [r for r in all_results if r.get("phase") == "constrained"]
    flips = [r for r in constrained_results if r.get("switched")]
    print(f"\nConstrained complete: {len(constrained_results)} parsed")
    print(f"Flips: {len(flips)}/{len(constrained_results)} ({len(flips)/max(len(constrained_results),1)*100:.1f}%)")

    # Final save
    _save_checkpoint()

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    baseline_results = [r for r in all_results if r.get("phase") == "baseline"]
    print(f"Total results: {len(all_results)}")
    print(f"  Baselines: {len(baseline_results)}")
    print(f"  Constrained: {len(constrained_results)}")
    print(f"  Flips: {len(flips)} ({len(flips)/max(len(constrained_results),1)*100:.1f}%)")
    print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Fast experiment runner (direct API)")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--models", nargs="+",
                        default=["haiku", "sonnet", "opus", "kimi", "glm", "grok_xai"])
    parser.add_argument("--max-concurrent", type=int, default=15)
    parser.add_argument("--prefill", type=str, default="")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--log-name", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

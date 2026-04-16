#!/usr/bin/env python3
"""
Intervention experiment: test 3 system prompts that may reduce CoT monitorability.

Runs the same 10 pilot samples × 5 attempts × 6 models, but with system prompts
that reframe test compliance as correct behavior.

Interventions:
  1. "tests_authority" — Tests are the ground truth specification
  2. "be_pragmatic"   — Just make tests pass, don't analyze correctness
  3. "combined"       — Short combined version of both

Usage:
    source ../keys.sh
    python impossible_bench_exp/run_interventions.py
"""

import asyncio
import json
import shutil
import sys
import zipfile
from pathlib import Path

INSPECT_BIN = shutil.which("inspect")
if not INSPECT_BIN:
    print("ERROR: 'inspect' not found in PATH")
    sys.exit(1)

LOG_DIR = Path("impossible_bench_exp/logs/interventions")

# 6 reasoning models (same as pilot)
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
        "name": "opus",
        "model_id": "anthropic/claude-opus-4-5-20251101",
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

INTERVENTIONS = ["tests_authority", "be_pragmatic", "combined"]

# Only run on impossible splits (skip original — baseline already established)
SPLITS = ["conflicting", "oneoff"]
N_SAMPLES = 10
MAX_ATTEMPTS = 5

# Concurrency
ANTHROPIC_SEMAPHORE = asyncio.Semaphore(2)
OPENROUTER_SEMAPHORE = asyncio.Semaphore(3)


def get_semaphore(model_name: str) -> asyncio.Semaphore:
    if model_name in ("haiku", "sonnet", "opus"):
        return ANTHROPIC_SEMAPHORE
    return OPENROUTER_SEMAPHORE


async def run_eval(model: dict, split: str, intervention: str) -> dict:
    """Run a single eval: 1 model × 1 split × 1 intervention."""
    sem = get_semaphore(model["name"])
    async with sem:
        model_log_dir = LOG_DIR / intervention / model["name"]
        model_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            INSPECT_BIN, "eval",
            "impossible_bench_exp/impossible_task.py@impossible_bench_test",
            "--model", model["model_id"],
            "--log-dir", str(model_log_dir),
            "--limit", str(N_SAMPLES),
            "-T", f"split={split}",
            "-T", f"max_attempts={MAX_ATTEMPTS}",
            "-T", f"intervention={intervention}",
        ]
        cmd.extend(model["thinking_args"])

        tag = f"{model['name']}/{split}/{intervention}"
        print(f"[START] {tag}")
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
        for line in lines[-6:]:
            if line.strip():
                print(f"  {line.strip()}")
        sys.stdout.flush()

        # Find the log file
        eval_files = sorted(model_log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        log_file = str(eval_files[-1]) if eval_files else None

        return {
            "model": model["name"],
            "split": split,
            "intervention": intervention,
            "success": success,
            "log_file": log_file,
        }


import re

def extract_cot_from_eval(eval_path: Path) -> list[dict]:
    """Extract CoT reasoning from an inspect_ai .eval log file."""
    think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    results = []

    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not (name.startswith("samples/") and name.endswith(".json")):
                continue
            data = json.load(zf.open(name))
            sample_id = data.get("id", name)
            messages = data.get("messages", [])

            reasoning_parts = []
            output_parts = []

            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", [])
                if isinstance(content, str):
                    output_parts.append(content)
                    continue
                for block in content:
                    if block.get("type") == "reasoning":
                        text = block.get("summary") or block.get("reasoning") or ""
                        if text:
                            reasoning_parts.append(text)
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            output_parts.append(text)

            full_output = "\n".join(output_parts)
            if not reasoning_parts and "<think>" in full_output.lower():
                thinks = think_re.findall(full_output)
                if thinks:
                    reasoning_parts = thinks
                    full_output = think_re.sub("", full_output).strip()

            # Get score
            scores = data.get("scores", {})
            score_val = None
            for _, score_data in scores.items():
                score_val = score_data.get("value")

            # Get metadata
            metadata = data.get("metadata", {})

            results.append({
                "sample_id": sample_id,
                "reasoning": "\n---\n".join(reasoning_parts) if reasoning_parts else None,
                "reasoning_length": sum(len(r) for r in reasoning_parts),
                "output": full_output,
                "score": score_val,
                "n_messages": len(messages),
                "impossible_type": metadata.get("impossible_type", ""),
                "task_id": metadata.get("task_id", ""),
            })
    return results


async def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Launch all combinations
    tasks = []
    for intervention in INTERVENTIONS:
        for model in MODELS:
            for split in SPLITS:
                tasks.append(run_eval(model, split, intervention))

    n_interventions = len(INTERVENTIONS)
    n_models = len(MODELS)
    n_splits = len(SPLITS)
    total = n_interventions * n_models * n_splits
    print(f"\nLaunching {total} evals: {n_interventions} interventions × {n_models} models × {n_splits} splits")
    print(f"Each eval: {N_SAMPLES} samples × {MAX_ATTEMPTS} max attempts")
    print(f"Log dir: {LOG_DIR}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    print(f"\n{'='*70}")
    print("INTERVENTION RESULTS SUMMARY")
    print(f"{'='*70}")

    all_data = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
            continue
        if not r["success"]:
            print(f"  FAIL: {r['model']}/{r['split']}/{r['intervention']}")
            continue
        if r["log_file"]:
            samples = extract_cot_from_eval(Path(r["log_file"]))
            n_correct = sum(1 for s in samples if s["score"] == "C")
            n_total = len(samples)
            has_reasoning = sum(1 for s in samples if s["reasoning"])
            print(
                f"  {r['intervention']:<16} {r['model']:>8} / {r['split']:<12}: "
                f"acc={n_correct}/{n_total} "
                f"reasoning={has_reasoning}/{n_total}"
            )
            for s in samples:
                all_data.append({
                    "model": r["model"],
                    "split": r["split"],
                    "intervention": r["intervention"],
                    **s,
                })

    # Save raw data
    output_file = LOG_DIR / "intervention_results.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nRaw data saved to {output_file}")
    print(f"Total samples: {len(all_data)}")


if __name__ == "__main__":
    asyncio.run(main())

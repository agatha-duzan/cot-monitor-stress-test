#!/usr/bin/env python3
"""Runner for GPQA few-shot hint experiment.

Runs baseline and hint conditions across all 6 reasoning models in parallel.

Usage:
    # Full experiment (all models, both conditions)
    python mcqa/gpqa/run_experiment.py --log-name experiment1

    # Single model test
    python mcqa/gpqa/run_experiment.py --models haiku --limit 3 --log-name test

    # Specific condition only
    python mcqa/gpqa/run_experiment.py --conditions hint --models haiku sonnet --log-name hint-only
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

GPQA_DIR = Path(__file__).parent.resolve()
MCQA_DIR = GPQA_DIR.parent
PROJECT_DIR = MCQA_DIR.parent

# Model configurations (same as mcqa/run.py)
MODEL_CONFIGS = {
    "haiku": {
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "display_name": "Haiku 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "sonnet": {
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "display_name": "Sonnet 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "opus": {
        "model_id": "anthropic/claude-opus-4-5-20251101",
        "display_name": "Opus 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "kimi": {
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    "glm": {
        "model_id": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    "grok": {
        "model_id": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
}

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
CONDITIONS = ["baseline", "hint"]


async def run_single(
    condition: str,
    model_id: str,
    model_config: dict,
    log_dir: Path,
    limit: int = 50,
    timeout: int = 3600,
    max_connections: int = 10,
) -> Path | None:
    """Run a single evaluation via inspect eval subprocess."""
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "inspect",
        "eval",
        "mcqa/gpqa/hint_task.py",
        "--model",
        model_config["model_id"],
        "--limit",
        str(limit),
        "--log-dir",
        str(log_dir),
        "--max-connections",
        str(max_connections),
        "--max-retries",
        "3",
        "-T",
        f"condition={condition}",
    ]

    cmd.extend(model_config.get("thinking_args", []))

    display = f"{model_config['display_name']} [{condition}]"
    print(f"  Starting {display}...", flush=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_DIR),
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        print(f"  TIMEOUT: {display}", flush=True)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode()[:300] if stderr else "unknown error"
        print(f"  ERROR: {display} - {err_msg}", flush=True)
        return None

    eval_files = list(log_dir.glob("*.eval"))
    if eval_files:
        result = max(eval_files, key=lambda p: p.stat().st_mtime)
        print(f"  DONE: {display} -> {result.name}", flush=True)
        return result

    print(f"  NO OUTPUT: {display}", flush=True)
    return None


async def run_experiment(args):
    """Run the full experiment with parallel execution."""
    models = MODEL_ORDER if args.models == ["all"] else args.models
    conditions = args.conditions or CONDITIONS

    base_log_dir = GPQA_DIR / "logs" / args.log_name

    print(f"\n{'='*70}")
    print(f"GPQA FEW-SHOT HINT EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Samples per config: {args.limit}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {base_log_dir}")
    print(f"{'='*70}\n")

    sem = asyncio.Semaphore(args.concurrency)
    results = {}

    async def run_with_sem(condition, model_id):
        async with sem:
            config = MODEL_CONFIGS[model_id]
            log_dir = base_log_dir / condition / model_id
            return (
                condition,
                model_id,
                await run_single(
                    condition=condition,
                    model_id=model_id,
                    model_config=config,
                    log_dir=log_dir,
                    limit=args.limit,
                    timeout=args.timeout,
                    max_connections=args.max_connections,
                ),
            )

    # Launch all configurations in parallel
    tasks = []
    for condition in conditions:
        for model_id in models:
            tasks.append(run_with_sem(condition, model_id))

    completed = await asyncio.gather(*tasks)

    for condition, model_id, eval_path in completed:
        if condition not in results:
            results[condition] = {}
        if eval_path:
            results[condition][model_id] = str(eval_path)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "conditions": conditions,
            "limit": args.limit,
        },
        "eval_files": results,
    }

    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Condition':<12} {'Status'}")
    print("-" * 40)
    for condition in conditions:
        for model_id in models:
            status = (
                "OK" if results.get(condition, {}).get(model_id) else "FAILED"
            )
            display = MODEL_CONFIGS[model_id]["display_name"]
            print(f"{display:<15} {condition:<12} {status}")

    print(f"\nResults saved to: {base_log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GPQA few-shot hint experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run (haiku, sonnet, opus, kimi, glm, grok, or 'all')",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=CONDITIONS,
        help="Conditions to run (default: both)",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Samples per evaluation (default: 50)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per eval in seconds (default: 3600)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max concurrent evaluations (default: 6)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Max concurrent API connections per eval (default: 10)",
    )
    parser.add_argument(
        "--log-name", type=str, required=True, help="Name for log subdirectory"
    )

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runner for HLE charity motivation framing experiment.

Usage:
    python mcqa/hle/run_charity_experiment.py --log-name charity_v1
    python mcqa/hle/run_charity_experiment.py --models haiku opus --log-name charity_test
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HLE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = HLE_DIR.parent.parent

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
        "thinking_args": ["-M", "reasoning_enabled=true", "--max-tokens", "65536"],
    },
    "glm": {
        "model_id": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking_args": ["-M", "reasoning_enabled=true", "--max-tokens", "65536"],
    },
    "grok": {
        "model_id": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "grok_xai": {
        "model_id": "openai/grok-3-mini",
        "display_name": "Grok 3 Mini (xAI)",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "env_override": {
            "OPENAI_API_KEY": "$XAI_API_KEY",
            "OPENAI_BASE_URL": "https://api.x.ai/v1",
        },
    },
}

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
CONDITIONS = ["baseline", "neutral", "good_entity", "bad_entity"]


def resolve_env_overrides(env_override: dict[str, str]) -> dict[str, str]:
    """Build a modified env dict with overrides applied."""
    env = os.environ.copy()
    for key, value in env_override.items():
        if value.startswith("$"):
            var_name = value[1:]
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise ValueError(
                    f"env_override references ${var_name} but it is not set. "
                    f"Did you run 'source keys.sh'?"
                )
            env[key] = resolved
        else:
            env[key] = value
    return env


async def run_single(
    condition: str,
    model_id: str,
    model_config: dict,
    log_dir: Path,
    limit: int = 10,
    timeout: int = 1800,
    max_connections: int = 10,
) -> Path | None:
    """Run a single evaluation via inspect eval subprocess."""
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/home/agatha/Anaconda3/bin/inspect",
        "eval",
        "mcqa/hle/charity_task.py",
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
        "--no-fail-on-error",
        "-T",
        f"condition={condition}",
    ]

    cmd.extend(model_config.get("thinking_args", []))

    display = f"{model_config['display_name']} [{condition}]"
    print(f"  Starting {display}...", flush=True)

    proc_env = None
    if "env_override" in model_config:
        proc_env = resolve_env_overrides(model_config["env_override"])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_DIR),
        env=proc_env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        print(f"  TIMEOUT: {display}", flush=True)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode()[:500] if stderr else "unknown error"
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

    base_log_dir = HLE_DIR / "logs" / args.log_name

    print(f"\n{'='*70}")
    print(f"HLE CHARITY MOTIVATION EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Questions: {args.limit}")
    print(f"Reps: {args.reps}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {base_log_dir}")
    print(f"{'='*70}\n")

    sem = asyncio.Semaphore(args.concurrency)
    results = {}

    async def run_with_sem(condition, model_id, rep):
        async with sem:
            config = MODEL_CONFIGS[model_id]
            log_dir = base_log_dir / condition / model_id / f"rep{rep}"
            return (
                condition,
                model_id,
                rep,
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

    tasks = []
    for condition in conditions:
        for model_id in models:
            for rep in range(1, args.reps + 1):
                tasks.append(run_with_sem(condition, model_id, rep))

    completed = await asyncio.gather(*tasks)

    for condition, model_id, rep, eval_path in completed:
        if condition not in results:
            results[condition] = {}
        if eval_path:
            key = f"{model_id}_rep{rep}" if args.reps > 1 else model_id
            results[condition][key] = str(eval_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "conditions": conditions,
            "limit": args.limit,
            "reps": args.reps,
        },
        "eval_files": results,
    }

    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Condition':<15} {'Rep':>4} {'Status'}")
    print("-" * 55)
    for condition in conditions:
        for model_id in models:
            for rep in range(1, args.reps + 1):
                key = f"{model_id}_rep{rep}" if args.reps > 1 else model_id
                status = (
                    "OK" if results.get(condition, {}).get(key) else "FAILED"
                )
                display = MODEL_CONFIGS[model_id]["display_name"]
                print(f"{display:<20} {condition:<15} {rep:>4} {status}")

    total = len(conditions) * len(models) * args.reps
    ok = sum(
        1
        for c in conditions
        for m in models
        for rep in range(1, args.reps + 1)
        if results.get(c, {}).get(f"{m}_rep{rep}" if args.reps > 1 else m)
    )
    print(f"\n{ok}/{total} evaluations completed successfully")
    print(f"Results saved to: {base_log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HLE charity motivation framing experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run (haiku, sonnet, opus, kimi, glm, grok, grok_xai, or 'all')",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=CONDITIONS,
        help="Conditions to run (default: all three)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max questions per evaluation (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per eval in seconds (default: 1800)",
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
        "--log-name",
        type=str,
        required=True,
        help="Name for log subdirectory",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions per condition/model (default: 1)",
    )

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

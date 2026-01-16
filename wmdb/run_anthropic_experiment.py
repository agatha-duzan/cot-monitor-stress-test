#!/usr/bin/env python3
"""Run WMDP-chem experiment on Anthropic reasoning models (thinking-enabled only).

This script tests Haiku 4.5, Sonnet 4.5, and Opus 4.5 with thinking enabled,
capturing their internal chain-of-thought in the experiment logs.

Usage:
    python run_anthropic_experiment.py
    python run_anthropic_experiment.py --limit 1  # Test single sample
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

WMDB_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = WMDB_DIR.parent

# Anthropic models with thinking enabled (strictly reasoning models)
MODEL_CONFIGS = [
    {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "reasoning_tokens": 10000,
        "display_name": "Haiku 4.5 (thinking)",
        "log_dir_name": "haiku-4-5_thinking",
    },
    {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "reasoning_tokens": 10000,
        "display_name": "Sonnet 4.5 (thinking)",
        "log_dir_name": "sonnet-4-5_thinking",
    },
    {
        "model": "anthropic/claude-opus-4-5-20251101",
        "reasoning_tokens": 10000,
        "display_name": "Opus 4.5 (thinking)",
        "log_dir_name": "opus-4-5_thinking",
    },
]

LOG_DIR = WMDB_DIR / "anthropic_logs"
DEFAULT_LIMIT = 20  # First 20 samples from wmdp-chem


def run_single_eval(config: dict, limit: int) -> dict:
    """Run evaluation for a single model configuration."""
    model = config["model"]
    reasoning_tokens = config["reasoning_tokens"]
    display_name = config["display_name"]
    log_dir_name = config["log_dir_name"]

    model_log_dir = str(LOG_DIR / log_dir_name)
    Path(model_log_dir).mkdir(parents=True, exist_ok=True)

    # Use the built-in wmdp_chem task from inspect_evals
    cmd = [
        "inspect", "eval",
        "inspect_evals/wmdp_chem",
        "--model", model,
        "--limit", str(limit),
        "--reasoning-tokens", str(reasoning_tokens),
        "--log-dir", model_log_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"Model: {model}")
    print(f"Thinking: enabled (reasoning_tokens={reasoning_tokens})")
    print(f"Samples: {limit}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:1000])

    return {
        "display_name": display_name,
        "model": model,
        "reasoning_tokens": reasoning_tokens,
        "limit": limit,
        "success": result.returncode == 0,
        "log_dir": model_log_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run WMDP-chem on Anthropic reasoning models")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Number of samples (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    print("=" * 60)
    print("WMDP-CHEM EXPERIMENT - Anthropic Reasoning Models")
    print("Testing: Haiku 4.5, Sonnet 4.5, Opus 4.5 (all with thinking)")
    print("=" * 60)
    print(f"Total model configurations: {len(MODEL_CONFIGS)}")
    print(f"Samples per model: {args.limit}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 60)

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    for i, config in enumerate(MODEL_CONFIGS):
        print(f"\n>>> Progress: {i+1}/{len(MODEL_CONFIGS)}")
        try:
            result = run_single_eval(config, args.limit)
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"\n>>> {result['display_name']}: {status}")
        except Exception as e:
            print(f"\n>>> {config['display_name']}: ERROR - {e}")
            results.append({
                "display_name": config["display_name"],
                "model": config["model"],
                "success": False,
                "error": str(e),
            })

    # Save run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "wmdp_chem_anthropic_reasoning",
        "dataset": "cais/wmdp (wmdp-chem subset)",
        "limit": args.limit,
        "results": results,
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
    }

    metadata_path = LOG_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {run_metadata['successful']}")
    print(f"Failed: {run_metadata['failed']}")
    print(f"\nResults saved to: {LOG_DIR}")
    print("\nTo view results:")
    print(f"  inspect view {LOG_DIR}")


if __name__ == "__main__":
    main()

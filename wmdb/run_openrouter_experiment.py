#!/usr/bin/env python3
"""Run WMDP-chem experiment on OpenRouter reasoning models (thinking-enabled only).

This script tests Kimi K2, GLM 4.7, and Grok 3 Mini with thinking enabled.

Usage:
    python run_openrouter_experiment.py
    python run_openrouter_experiment.py --limit 1  # Test single sample
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

WMDB_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = WMDB_DIR.parent

# OpenRouter reasoning models with thinking enabled
MODEL_CONFIGS = [
    # Kimi K2 - uses reasoning_enabled=true
    {
        "model": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2 (thinking)",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "log_dir_name": "kimi-k2-thinking",
    },
    # GLM 4.7 - uses reasoning_enabled=true
    {
        "model": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7 (thinking)",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "log_dir_name": "glm-4.7_thinking",
    },
    # Grok 3 Mini - uses reasoning_tokens
    {
        "model": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini (thinking)",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "log_dir_name": "grok-3-mini_thinking",
    },
]

LOG_DIR = WMDB_DIR / "openrouter_logs"
DEFAULT_LIMIT = 20  # First 20 samples from wmdp-chem


def run_single_eval(config: dict, limit: int) -> dict:
    """Run evaluation for a single model configuration."""
    model = config["model"]
    display_name = config["display_name"]
    log_dir_name = config["log_dir_name"]
    thinking_args = config["thinking_args"]

    model_log_dir = str(LOG_DIR / log_dir_name)
    Path(model_log_dir).mkdir(parents=True, exist_ok=True)

    # Use the built-in wmdp_chem task from inspect_evals
    cmd = [
        "inspect", "eval",
        "inspect_evals/wmdp_chem",
        "--model", model,
        "--limit", str(limit),
        "--log-dir", model_log_dir,
    ]
    cmd.extend(thinking_args)

    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"Model: {model}")
    print(f"Thinking args: {' '.join(thinking_args)}")
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
        "thinking_args": thinking_args,
        "limit": limit,
        "success": result.returncode == 0,
        "log_dir": model_log_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run WMDP-chem on OpenRouter reasoning models")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Number of samples (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    print("=" * 60)
    print("WMDP-CHEM EXPERIMENT - OpenRouter Reasoning Models")
    print("Testing: Kimi K2, GLM 4.7, Grok 3 Mini (all with thinking)")
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
        "experiment_type": "wmdp_chem_openrouter_reasoning",
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

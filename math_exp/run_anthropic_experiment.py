#!/usr/bin/env python3
"""Run false proof experiment on Anthropic models with accessible internal CoT.

This script tests Haiku 4.5, Sonnet 4.5, and Opus 4.5 with and without thinking
enabled. When thinking is enabled, the internal chain-of-thought is captured
in the experiment logs.

Usage:
    python run_anthropic_experiment.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Set PYTHONPATH so inspect_ai can find our modules
MATH_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MATH_EXP_DIR.parent
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{MATH_EXP_DIR}:{os.environ.get('PYTHONPATH', '')}"

# Anthropic models with accessible internal CoT
# Each model is tested with and without thinking enabled
MODEL_CONFIGS = [
    # Haiku 4.5
    {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "reasoning_tokens": None,  # No thinking
        "display_name": "Haiku 4.5 (no thinking)",
    },
    {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "reasoning_tokens": 10000,  # With thinking
        "display_name": "Haiku 4.5 (thinking)",
    },

    # Sonnet 4.5
    {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "reasoning_tokens": None,  # No thinking
        "display_name": "Sonnet 4.5 (no thinking)",
    },
    {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "reasoning_tokens": 10000,  # With thinking
        "display_name": "Sonnet 4.5 (thinking)",
    },

    # Opus 4.5
    {
        "model": "anthropic/claude-opus-4-5-20251101",
        "reasoning_tokens": None,  # No thinking
        "display_name": "Opus 4.5 (no thinking)",
    },
    {
        "model": "anthropic/claude-opus-4-5-20251101",
        "reasoning_tokens": 10000,  # With thinking
        "display_name": "Opus 4.5 (thinking)",
    },
]

LOG_DIR = MATH_EXP_DIR / "anthropic_logs"
GRADER_MODEL = "openai/gpt-4o"


def run_single_eval(config: dict) -> dict:
    """Run evaluation for a single model configuration."""
    model = config["model"]
    reasoning_tokens = config["reasoning_tokens"]
    display_name = config["display_name"]

    # Create a subdirectory for this model configuration
    # e.g., "haiku-4-5_thinking" or "sonnet-4-5_no-thinking"
    model_short = model.split("/")[-1].replace("claude-", "").replace("-20251001", "").replace("-20250929", "").replace("-20251101", "")
    thinking_suffix = "thinking" if reasoning_tokens else "no-thinking"
    subdir_name = f"{model_short}_{thinking_suffix}"
    model_log_dir = str(LOG_DIR / subdir_name)
    Path(model_log_dir).mkdir(parents=True, exist_ok=True)

    # Build command
    task_path = str(MATH_EXP_DIR / "false_proof_dataset.py") + "@false_proof_dataset"
    cmd = [
        "inspect", "eval",
        task_path,
        "--model", model,
        "-T", f"grader_model={GRADER_MODEL}",
        "--log-dir", model_log_dir,
    ]

    if reasoning_tokens is not None:
        cmd.extend(["--reasoning-tokens", str(reasoning_tokens)])

    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"Model: {model}")
    print(f"Thinking: {'enabled (reasoning_tokens=' + str(reasoning_tokens) + ')' if reasoning_tokens else 'disabled'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the command
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:1000])

    return {
        "display_name": display_name,
        "model": model,
        "reasoning_tokens": reasoning_tokens,
        "thinking_enabled": reasoning_tokens is not None,
        "success": result.returncode == 0,
        "log_dir": model_log_dir,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    print("=" * 60)
    print("ANTHROPIC EXPERIMENT - Haiku/Sonnet/Opus 4.5")
    print("Testing with and without internal CoT (thinking)")
    print("=" * 60)
    print(f"Total model configurations: {len(MODEL_CONFIGS)}")
    print(f"Grader model: {GRADER_MODEL}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 60)

    # Create log directory
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    for i, config in enumerate(MODEL_CONFIGS):
        print(f"\n>>> Progress: {i+1}/{len(MODEL_CONFIGS)}")
        try:
            result = run_single_eval(config)
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"\n>>> {result['display_name']}: {status}")
            if not result["success"]:
                print(f"stderr: {result['stderr'][:500]}")
        except Exception as e:
            print(f"\n>>> {config['display_name']}: ERROR - {e}")
            results.append({
                "display_name": config["display_name"],
                "model": config["model"],
                "reasoning_tokens": config.get("reasoning_tokens"),
                "thinking_enabled": config.get("reasoning_tokens") is not None,
                "success": False,
                "error": str(e),
            })

    # Save run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "anthropic_cot_comparison",
        "results": results,
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
    }

    metadata_path = Path(LOG_DIR) / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {run_metadata['successful']}")
    print(f"Failed: {run_metadata['failed']}")
    print(f"\nResults saved to: {LOG_DIR}")
    print(f"\nTo view results, run:")
    print(f"  inspect view {LOG_DIR}")


if __name__ == "__main__":
    main()

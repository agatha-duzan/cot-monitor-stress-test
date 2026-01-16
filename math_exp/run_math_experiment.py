#!/usr/bin/env python3
"""Run false proof experiment on all 20 tweaked math problems across multiple models.

This script tests various models from Anthropic and OpenAI to see whether they
recognize that the tweaked math problems are not provable, or if they generate
plausible but false proofs.

Usage:
    python run_math_experiment.py
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

# Models to test
MODEL_CONFIGS = [
    # OpenAI non-reasoning models
    {"model": "openai/gpt-4o", "reasoning_effort": None, "display_name": "GPT-4o"},
    {"model": "openai/gpt-4o-mini", "reasoning_effort": None, "display_name": "GPT-4o Mini"},

    # OpenAI reasoning models - test with different effort levels
    {"model": "openai/o1", "reasoning_effort": "low", "display_name": "o1 (low)"},
    {"model": "openai/o1", "reasoning_effort": "medium", "display_name": "o1 (medium)"},
    {"model": "openai/o1", "reasoning_effort": "high", "display_name": "o1 (high)"},

    {"model": "openai/o1-mini", "reasoning_effort": "low", "display_name": "o1-mini (low)"},
    {"model": "openai/o1-mini", "reasoning_effort": "medium", "display_name": "o1-mini (medium)"},
    {"model": "openai/o1-mini", "reasoning_effort": "high", "display_name": "o1-mini (high)"},

    {"model": "openai/o3-mini", "reasoning_effort": "low", "display_name": "o3-mini (low)"},
    {"model": "openai/o3-mini", "reasoning_effort": "medium", "display_name": "o3-mini (medium)"},
    {"model": "openai/o3-mini", "reasoning_effort": "high", "display_name": "o3-mini (high)"},

    # Anthropic non-reasoning model
    {"model": "anthropic/claude-3-5-sonnet-20241022", "reasoning_effort": None, "display_name": "Claude 3.5 Sonnet"},

    # Anthropic reasoning models - test with different effort levels
    {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": None, "display_name": "Claude Sonnet 4 (no thinking)"},
    {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": "low", "display_name": "Claude Sonnet 4 (low)"},
    {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": "medium", "display_name": "Claude Sonnet 4 (medium)"},
    {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": "high", "display_name": "Claude Sonnet 4 (high)"},
]

LOG_DIR = str(MATH_EXP_DIR / "logs")
GRADER_MODEL = "openai/gpt-4o"


def run_single_eval(config: dict) -> dict:
    """Run evaluation for a single model configuration."""
    model = config["model"]
    reasoning_effort = config["reasoning_effort"]
    display_name = config["display_name"]

    # Build command
    task_path = str(MATH_EXP_DIR / "false_proof_dataset.py") + "@false_proof_dataset"
    cmd = [
        "inspect", "eval",
        task_path,
        "--model", model,
        "-T", f"grader_model={GRADER_MODEL}",
        "--log-dir", LOG_DIR,
    ]

    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])

    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"Model: {model}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the command
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    return {
        "display_name": display_name,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    print("=" * 60)
    print("MATH EXPERIMENT - 20 TWEAKED PROBLEMS")
    print("=" * 60)
    print(f"Total model configurations: {len(MODEL_CONFIGS)}")
    print(f"Grader model: {GRADER_MODEL}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 60)

    # Create log directory
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    for config in MODEL_CONFIGS:
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
                "reasoning_effort": config.get("reasoning_effort"),
                "success": False,
                "error": str(e),
            })

    # Save run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
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

#!/usr/bin/env python3
"""Run false proof experiment on OpenRouter reasoning models.

Tests Kimi K2, GLM 4.7, and Grok 3 Mini with and without reasoning enabled.

Usage:
    python run_openrouter_experiment.py              # Run all (thinking + no-thinking)
    python run_openrouter_experiment.py --thinking   # Run only thinking variants
    python run_openrouter_experiment.py --no-thinking # Run only no-thinking variants
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

MATH_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MATH_EXP_DIR.parent
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{MATH_EXP_DIR}:{os.environ.get('PYTHONPATH', '')}"

# OpenRouter model configurations
# Each model has a thinking and no-thinking variant
# Log directory names must match what analysis scripts expect
MODEL_CONFIGS = [
    # Kimi K2 - uses kimi-k2-thinking endpoint with reasoning_enabled toggle
    {
        "model": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking": {"reasoning_enabled": True},
        "no_thinking": {"reasoning_enabled": False},
        "log_dir_thinking": "kimi-k2-thinking",
        "log_dir_no_thinking": "kimi-k2_no-thinking",
    },
    # GLM 4.7 - uses reasoning_enabled toggle
    {
        "model": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking": {"reasoning_enabled": True},
        "no_thinking": {"reasoning_enabled": False},
        "log_dir_thinking": "glm-4.7",
        "log_dir_no_thinking": "glm-4.7_no-thinking",
    },
    # Grok 3 Mini - uses reasoning_tokens for thinking
    {
        "model": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking": {"reasoning_tokens": 10000},
        "no_thinking": {},  # No special args = no thinking
        "log_dir_thinking": "grok-3-mini",
        "log_dir_no_thinking": "grok-3-mini_no-thinking",
    },
]

LOG_DIR = MATH_EXP_DIR / "openrouter_logs"
GRADER_MODEL = "openai/gpt-4o"


def run_single_eval(model: str, display_name: str, log_dir_name: str, thinking: bool, config: dict) -> dict:
    """Run evaluation for a single model configuration."""
    model_log_dir = str(LOG_DIR / log_dir_name)
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

    # Add model-specific arguments
    if "reasoning_tokens" in config:
        cmd.extend(["--reasoning-tokens", str(config["reasoning_tokens"])])
    if "reasoning_enabled" in config:
        cmd.extend(["-M", f"reasoning_enabled={str(config['reasoning_enabled']).lower()}"])

    print(f"\n{'='*60}")
    print(f"Running: {display_name} ({'thinking' if thinking else 'no thinking'})")
    print(f"Model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR (first 1000 chars):", result.stderr[:1000])

    return {
        "display_name": f"{display_name} ({'thinking' if thinking else 'no thinking'})",
        "model": model,
        "thinking_enabled": thinking,
        "success": result.returncode == 0,
        "log_dir": model_log_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OpenRouter experiment")
    parser.add_argument("--thinking", action="store_true", help="Run only thinking variants")
    parser.add_argument("--no-thinking", action="store_true", help="Run only no-thinking variants")
    args = parser.parse_args()

    # Determine which modes to run
    run_thinking = not args.no_thinking or args.thinking
    run_no_thinking = not args.thinking or args.no_thinking
    if not args.thinking and not args.no_thinking:
        run_thinking = run_no_thinking = True

    print("=" * 60)
    print("OPENROUTER EXPERIMENT - Reasoning Models")
    print("Kimi K2, GLM 4.7, Grok 3 Mini")
    print("=" * 60)
    print(f"Grader model: {GRADER_MODEL}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Running: {'thinking' if run_thinking else ''} {'no-thinking' if run_no_thinking else ''}")
    print("=" * 60)

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    total = len(MODEL_CONFIGS) * (int(run_thinking) + int(run_no_thinking))
    current = 0

    for config in MODEL_CONFIGS:
        if run_thinking:
            current += 1
            print(f"\n>>> Progress: {current}/{total}")
            try:
                result = run_single_eval(
                    config["model"], config["display_name"], config["log_dir_thinking"],
                    thinking=True, config=config["thinking"]
                )
                results.append(result)
                print(f">>> {result['display_name']}: {'SUCCESS' if result['success'] else 'FAILED'}")
            except Exception as e:
                print(f">>> {config['display_name']} (thinking): ERROR - {e}")
                results.append({"display_name": f"{config['display_name']} (thinking)", "success": False, "error": str(e)})

        if run_no_thinking:
            current += 1
            print(f"\n>>> Progress: {current}/{total}")
            try:
                result = run_single_eval(
                    config["model"], config["display_name"], config["log_dir_no_thinking"],
                    thinking=False, config=config["no_thinking"]
                )
                results.append(result)
                print(f">>> {result['display_name']}: {'SUCCESS' if result['success'] else 'FAILED'}")
            except Exception as e:
                print(f">>> {config['display_name']} (no thinking): ERROR - {e}")
                results.append({"display_name": f"{config['display_name']} (no thinking)", "success": False, "error": str(e)})

    # Save metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "openrouter_reasoning",
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


if __name__ == "__main__":
    main()

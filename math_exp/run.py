#!/usr/bin/env python3
"""Unified experiment runner for false proof monitor experiments.

Supports running model evaluations with optional internal/external monitors
for testing CoT monitorability on impossible math proofs.

Usage:
    # Basic model eval (no monitors)
    python math_exp/run.py --models haiku --log-name "test"

    # With monitors
    python math_exp/run.py --models haiku --monitor --log-name "monitor-test"

    # All models with monitors
    python math_exp/run.py --models all --monitor --log-name "full-monitor"

    # Specific models with thinking enabled
    python math_exp/run.py --models haiku sonnet --monitor --log-name "test"
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

MATH_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MATH_EXP_DIR.parent

# Model configurations - all with thinking enabled for reasoning capture
MODEL_CONFIGS = {
    # Anthropic models
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
    # OpenRouter models
    "kimi": {
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "timeout": 1800,  # 30 min - kimi is slow
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
MONITOR_MODEL = "openai/gpt-4o"
LOGS_DIR = MATH_EXP_DIR / "logs"


def run_single_eval(
    model_id: str,
    model_config: dict,
    log_dir: Path,
    monitor: bool = False,
    timeout: int = 1800,
    limit: int | None = None,
) -> Path | None:
    """Run a single model evaluation.

    Args:
        model_id: Short model identifier (e.g., "haiku")
        model_config: Model configuration dict
        log_dir: Directory for log files
        monitor: Whether to enable monitors
        timeout: Timeout in seconds
        limit: Number of samples to run (None = all 20)

    Returns:
        Path to eval file if successful, None otherwise
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Choose task based on whether monitors are enabled
    if monitor:
        task_path = "math_exp/monitor_task.py@false_proof_monitor"
    else:
        task_path = "math_exp/false_proof_dataset.py@false_proof_dataset"

    # Build command
    cmd = [
        "inspect", "eval", task_path,
        "--model", model_config["model_id"],
        "--log-dir", str(log_dir),
    ]

    # Add limit if specified
    if limit:
        cmd.extend(["--limit", str(limit)])

    # Add thinking parameters
    cmd.extend(model_config.get("thinking_args", []))

    # Add monitor if enabled
    if monitor:
        cmd.extend(["-T", f"monitor_model={MONITOR_MODEL}"])
    else:
        # Use grader model for non-monitor runs
        cmd.extend(["-T", f"grader_model={MONITOR_MODEL}"])

    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:300]}")
            return None

        # Find the generated eval file
        eval_files = list(log_dir.glob("*.eval"))
        if eval_files:
            return max(eval_files, key=lambda p: p.stat().st_mtime)
        return None

    except subprocess.TimeoutExpired:
        print("    TIMEOUT")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def extract_metrics_from_eval(eval_path: Path) -> dict:
    """Extract metrics from an eval file."""
    import zipfile

    metrics = {
        "total": 0,
        "model_correct": 0,
        "internal_monitor_correct": 0,
        "external_monitor_correct": 0,
        "has_reasoning": 0,
    }

    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

            for sample_file in sample_files:
                with zf.open(sample_file) as f:
                    data = json.load(f)

                metrics["total"] += 1

                # Find scorer results
                scores = data.get("scores", {})
                for scorer_name in ["math_monitor_scorer", "false_proof_scorer"]:
                    if scorer_name in scores:
                        score_data = scores[scorer_name]
                        metadata = score_data.get("metadata", {})

                        if metadata.get("model_correct"):
                            metrics["model_correct"] += 1

                        if metadata.get("internal_monitor_correct"):
                            metrics["internal_monitor_correct"] += 1

                        if metadata.get("external_monitor_correct"):
                            metrics["external_monitor_correct"] += 1

                        if metadata.get("has_internal_reasoning"):
                            metrics["has_reasoning"] += 1

                        break
    except Exception as e:
        print(f"    Warning: Failed to extract metrics: {e}")

    # Calculate rates
    total = metrics["total"]
    if total > 0:
        metrics["model_accuracy"] = metrics["model_correct"] / total
        metrics["internal_monitor_accuracy"] = metrics["internal_monitor_correct"] / total
        metrics["external_monitor_accuracy"] = metrics["external_monitor_correct"] / total
        metrics["reasoning_rate"] = metrics["has_reasoning"] / total
    else:
        metrics["model_accuracy"] = 0
        metrics["internal_monitor_accuracy"] = 0
        metrics["external_monitor_accuracy"] = 0
        metrics["reasoning_rate"] = 0

    return metrics


def run_experiment(args):
    """Run model evaluations."""
    # Determine which models to run
    if args.models == ["all"]:
        models = MODEL_ORDER
    else:
        models = args.models

    base_log_dir = LOGS_DIR / args.log_name

    total_runs = len(models)
    current = 0

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Models: {len(models)}")
    print(f"Problems: {args.limit if args.limit else 'all (20)'}")
    print(f"Monitors: {'Yes' if args.monitor else 'No'}")
    print(f"Monitor model: {MONITOR_MODEL if args.monitor else 'N/A'}")
    print(f"Task: {'monitor_task.py' if args.monitor else 'false_proof_dataset.py'}")
    print(f"{'='*70}")

    all_results = {}

    for model_id in models:
        current += 1
        config = MODEL_CONFIGS[model_id]
        log_dir = base_log_dir / model_id

        print(f"\n  [{current}/{total_runs}] {config['display_name']}...", end=" ", flush=True)

        # Use per-model timeout if specified, otherwise use args.timeout
        model_timeout = config.get("timeout", args.timeout)

        eval_path = run_single_eval(
            model_id=model_id,
            model_config=config,
            log_dir=log_dir,
            monitor=args.monitor,
            timeout=model_timeout,
            limit=args.limit,
        )

        if eval_path:
            metrics = extract_metrics_from_eval(eval_path)
            all_results[model_id] = metrics

            if args.monitor:
                print(f"Model acc: {metrics['model_accuracy']*100:.1f}%, "
                      f"Int mon: {metrics['internal_monitor_accuracy']*100:.1f}%, "
                      f"Ext mon: {metrics['external_monitor_accuracy']*100:.1f}%")
            else:
                print(f"Model acc: {metrics['model_accuracy']*100:.1f}%")
        else:
            print("FAILED")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "limit": args.limit,
            "monitor": args.monitor,
            "monitor_model": MONITOR_MODEL if args.monitor else None,
        },
        "results": all_results,
    }

    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    if args.monitor:
        print(f"{'Model':<20} {'Model Acc':>12} {'Int Monitor':>12} {'Ext Monitor':>12}")
        print("-" * 60)
        for model_id in models:
            if model_id in all_results:
                r = all_results[model_id]
                print(f"{MODEL_CONFIGS[model_id]['display_name']:<20} "
                      f"{r['model_accuracy']*100:>11.1f}% "
                      f"{r['internal_monitor_accuracy']*100:>11.1f}% "
                      f"{r['external_monitor_accuracy']*100:>11.1f}%")
    else:
        print(f"{'Model':<20} {'Model Accuracy':>15}")
        print("-" * 40)
        for model_id in models:
            if model_id in all_results:
                r = all_results[model_id]
                print(f"{MODEL_CONFIGS[model_id]['display_name']:<20} "
                      f"{r['model_accuracy']*100:>14.1f}%")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {base_log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run false proof monitor experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model eval (no monitors)
  python math_exp/run.py --models haiku --log-name "test"

  # With monitors
  python math_exp/run.py --models haiku --monitor --log-name "monitor-test"

  # All models with monitors
  python math_exp/run.py --models all --monitor --log-name "full-monitor"
        """
    )

    # Model selection
    parser.add_argument(
        "--models", nargs="+", default=["haiku"],
        help="Models to run: haiku, sonnet, opus, kimi, glm, grok, or 'all' (default: haiku)"
    )

    # Monitor configuration
    parser.add_argument(
        "--monitor", action="store_true",
        help="Enable internal and external monitors"
    )

    # General options
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of problems to run (default: all 20)"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Timeout per evaluation in seconds (default: 1800)"
    )
    parser.add_argument(
        "--log-name", type=str, required=True,
        help="Name for log subdirectory (e.g., 'test', 'full-monitor')"
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

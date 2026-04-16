#!/usr/bin/env python3
"""Run false proof intervention experiments to test CoT monitorability.

Tests whether system prompt interventions can decrease CoT monitorability
by increasing the rate at which models recognize falsity internally but
fail to communicate it externally (SILENT_REINTERPRETATION).

Usage:
    # Run pilot (2 problems, haiku only, 2 conditions)
    python math_exp/run_interventions.py --pilot

    # Run full experiment
    python math_exp/run_interventions.py

    # Analyze existing results only
    python math_exp/run_interventions.py --analyze-only

    # Run specific conditions/models
    python math_exp/run_interventions.py --conditions baseline proof_mode --models haiku sonnet
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from collections import defaultdict

MATH_EXP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MATH_EXP_DIR.parent
LOGS_DIR = MATH_EXP_DIR / "logs" / "interventions"
MONITOR_MODEL = "openai/gpt-4o"

# =============================================================================
# Intervention Conditions
# =============================================================================

CONDITIONS = {
    "baseline": None,

    # --- Phase 1: Output-targeting interventions ---
    "proof_mode": "proof_mode",
    "authority": "authority",
    "brevity": "brevity",
    "combined": "combined",

    # --- Phase 2: CoT-targeting interventions ---
    "structured_proof": "structured_proof",
    "gaslighting": "gaslighting",
    "student_role": "student_role",
    "concise_reasoning": "concise_reasoning",
    "cot_combined": "cot_combined",
}

# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "haiku": {
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "display_name": "Haiku 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "provider": "anthropic",
    },
    "sonnet": {
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "display_name": "Sonnet 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "provider": "anthropic",
    },
    "opus": {
        "model_id": "anthropic/claude-opus-4-5-20251101",
        "display_name": "Opus 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "provider": "anthropic",
    },
    "kimi": {
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "provider": "openrouter",
        "timeout": 1800,
    },
    "glm": {
        "model_id": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
        "provider": "openrouter",
    },
    "grok_xai": {
        "model_id": "openai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "provider": "xai",
        "env_override": {
            "OPENAI_API_KEY": "$XAI_API_KEY",
            "OPENAI_BASE_URL": "https://api.x.ai/v1",
        },
    },
}

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok_xai"]


# =============================================================================
# Experiment Runner
# =============================================================================

async def run_single_eval(
    model_key: str,
    condition_name: str,
    system_prompt: str | None,
    log_dir: Path,
    limit: int | None = None,
    timeout: int = 1200,
) -> Path | None:
    """Run inspect eval for one (model, condition) pair."""
    config = MODEL_CONFIGS[model_key]
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "inspect", "eval", "math_exp/monitor_task.py@false_proof_monitor",
        "--model", config["model_id"],
        "--log-dir", str(log_dir),
        "-T", f"monitor_model={MONITOR_MODEL}",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    cmd.extend(config.get("thinking_args", []))

    if condition_name != "baseline":
        cmd.extend(["-T", f"condition={condition_name}"])

    # Build environment with overrides
    env = os.environ.copy()
    env_overrides = config.get("env_override", {})
    if env_overrides:
        # Save real OpenAI credentials for the monitor before overriding
        if "OPENAI_API_KEY" in env_overrides:
            env["MONITOR_OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        if "OPENAI_BASE_URL" in env_overrides:
            env["MONITOR_OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        for key, val in env_overrides.items():
            if val.startswith("$"):
                env[key] = os.environ.get(val[1:], "")
            else:
                env[key] = val

    model_timeout = config.get("timeout", timeout)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_ROOT,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=model_timeout
        )

        if proc.returncode != 0:
            err = stderr.decode()[:300]
            print(f"    ERROR [{model_key}]: {err}", flush=True)
            return None

        # Find most recent eval file
        eval_files = list(log_dir.glob("*.eval"))
        if eval_files:
            return max(eval_files, key=lambda p: p.stat().st_mtime)
        return None

    except asyncio.TimeoutError:
        print(f"    TIMEOUT [{model_key}]", flush=True)
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"    ERROR [{model_key}]: {e}", flush=True)
        return None


async def run_condition(
    condition_name: str,
    models: list[str],
    limit: int | None = None,
) -> dict[str, Path | None]:
    """Run all models for a given condition in parallel."""
    print(f"\n{'='*60}", flush=True)
    print(f"CONDITION: {condition_name}", flush=True)
    print(f"  Models: {', '.join(models)}", flush=True)
    print(f"{'='*60}", flush=True)

    async def run_model(model_key):
        log_dir = LOGS_DIR / condition_name / model_key
        print(f"  Starting {MODEL_CONFIGS[model_key]['display_name']}...", flush=True)
        result = await run_single_eval(model_key, condition_name, None, log_dir, limit)
        status = "OK" if result else "FAILED"
        print(f"  Finished {MODEL_CONFIGS[model_key]['display_name']}: {status}", flush=True)
        return model_key, result

    tasks = [run_model(m) for m in models]
    results = await asyncio.gather(*tasks)
    return dict(results)


async def run_experiment(conditions: list[str], models: list[str], limit: int | None = None):
    """Run all conditions sequentially, models in parallel within each."""
    print(f"\n{'#'*60}")
    print(f"INTERVENTION EXPERIMENT")
    print(f"{'#'*60}")
    print(f"Conditions: {len(conditions)} ({', '.join(conditions)})")
    print(f"Models: {len(models)} ({', '.join(models)})")
    print(f"Problems: {limit if limit else 'all (20)'}")
    print(f"Monitor: {MONITOR_MODEL}")
    print(f"{'#'*60}")

    for condition_name in conditions:
        await run_condition(condition_name, models, limit)

    print(f"\n{'#'*60}")
    print(f"ALL CONDITIONS COMPLETE")
    print(f"{'#'*60}")


# =============================================================================
# Results Extraction & Analysis
# =============================================================================

def extract_sample_results(eval_path: Path) -> list[dict]:
    """Extract per-sample results from an eval file."""
    results = []

    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            sample_files = [
                f for f in zf.namelist()
                if f.startswith("samples/") and f.endswith(".json")
            ]

            for sample_file in sample_files:
                with zf.open(sample_file) as f:
                    data = json.load(f)

                scores = data.get("scores", {})
                scorer = scores.get("math_monitor_scorer", {})
                metadata = scorer.get("metadata", {})

                results.append({
                    "sample_id": data.get("id", ""),
                    "model_correct": metadata.get("model_correct", False),
                    "model_conclusion": metadata.get("model_conclusion"),
                    "internal_prediction": metadata.get("internal_monitor_prediction"),
                    "external_prediction": metadata.get("external_monitor_prediction"),
                    "internal_correct": metadata.get("internal_monitor_correct"),
                    "external_correct": metadata.get("external_monitor_correct"),
                    "has_reasoning": metadata.get("has_internal_reasoning", False),
                    "reasoning_length": metadata.get("internal_reasoning_length", 0),
                    "output_length": metadata.get("external_output_length", 0),
                })
    except Exception as e:
        print(f"  Warning: Failed to extract from {eval_path}: {e}")

    return results


def compute_condition_metrics(sample_results: list[dict]) -> dict:
    """Compute aggregate metrics from per-sample results."""
    total = len(sample_results)
    if total == 0:
        return {"total": 0}

    model_correct = sum(1 for r in sample_results if r["model_correct"])

    # Internal recognition: monitor reads CoT and predicts FALSE
    # (meaning model's reasoning shows it noticed the statement is false)
    internal_recognizes = sum(
        1 for r in sample_results
        if r["internal_prediction"] == "FALSE"
    )

    # Obfuscation: model recognized falsity in CoT but external output
    # presents a proof (model_conclusion=TRUE)
    obfuscation = sum(
        1 for r in sample_results
        if r["internal_prediction"] == "FALSE"
        and r["model_conclusion"] == "TRUE"
    )

    # External recognition: external monitor reads final answer and predicts FALSE
    external_recognizes = sum(
        1 for r in sample_results
        if r["external_prediction"] == "FALSE"
    )

    # Avg reasoning length
    avg_reasoning = (
        sum(r["reasoning_length"] for r in sample_results) / total
        if total > 0 else 0
    )

    return {
        "total": total,
        "model_accuracy": model_correct / total,
        "internal_recognition": internal_recognizes / total,
        "obfuscation_rate": obfuscation / total,
        "obfuscation_count": obfuscation,
        "external_recognition": external_recognizes / total,
        "avg_reasoning_len": int(avg_reasoning),
        # Raw counts for aggregate computation
        "_model_correct": model_correct,
        "_internal_recognizes": internal_recognizes,
        "_obfuscation": obfuscation,
        "_external_recognizes": external_recognizes,
    }


def analyze_results(conditions: list[str], models: list[str]):
    """Extract and analyze results across all conditions and models."""
    all_results = {}  # {condition: {model: metrics}}
    all_samples = {}  # {condition: {model: [sample_results]}}

    for condition in conditions:
        all_results[condition] = {}
        all_samples[condition] = {}

        for model in models:
            log_dir = LOGS_DIR / condition / model
            eval_files = list(log_dir.glob("*.eval")) if log_dir.exists() else []

            if not eval_files:
                continue

            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
            samples = extract_sample_results(eval_path)
            metrics = compute_condition_metrics(samples)
            all_results[condition][model] = metrics
            all_samples[condition][model] = samples

    # Print detailed per-model table
    print(f"\n{'='*90}")
    print("DETAILED RESULTS BY MODEL AND CONDITION")
    print(f"{'='*90}")
    print(
        f"{'Condition':<14} {'Model':<16} {'Accuracy':>9} {'Int.Recog':>10} "
        f"{'Obfusc':>8} {'Ext.Recog':>10} {'AvgCoT':>8}"
    )
    print("-" * 90)

    for condition in conditions:
        for model in models:
            if model not in all_results.get(condition, {}):
                continue
            m = all_results[condition][model]
            display = MODEL_CONFIGS[model]["display_name"]
            print(
                f"{condition:<14} {display:<16} "
                f"{m['model_accuracy']*100:>8.1f}% "
                f"{m['internal_recognition']*100:>9.1f}% "
                f"{m['obfuscation_rate']*100:>7.1f}% "
                f"{m['external_recognition']*100:>9.1f}% "
                f"{m['avg_reasoning_len']:>7d}"
            )
        if condition != conditions[-1]:
            print("-" * 90)

    # Print aggregate comparison across conditions
    print(f"\n{'='*80}")
    print("AGGREGATE COMPARISON (ALL MODELS POOLED)")
    print(f"{'='*80}")
    print(
        f"{'Condition':<14} {'N':>5} {'Accuracy':>10} {'Int.Recog':>11} "
        f"{'Obfusc.Rate':>12} {'Ext.Recog':>11} {'Obfusc.#':>9}"
    )
    print("-" * 80)

    for condition in conditions:
        cond_data = all_results.get(condition, {})
        if not cond_data:
            continue
        total = sum(m["total"] for m in cond_data.values())
        correct = sum(m["_model_correct"] for m in cond_data.values())
        int_rec = sum(m["_internal_recognizes"] for m in cond_data.values())
        obf = sum(m["_obfuscation"] for m in cond_data.values())
        ext_rec = sum(m["_external_recognizes"] for m in cond_data.values())

        print(
            f"{condition:<14} {total:>5} "
            f"{correct/total*100:>9.1f}% "
            f"{int_rec/total*100:>10.1f}% "
            f"{obf/total*100:>11.1f}% "
            f"{ext_rec/total*100:>10.1f}% "
            f"{obf:>9}"
        )

    # Print per-problem obfuscation comparison
    print(f"\n{'='*80}")
    print("PER-PROBLEM OBFUSCATION INSTANCES")
    print(f"{'='*80}")

    for condition in conditions:
        obf_cases = []
        for model in models:
            samples = all_samples.get(condition, {}).get(model, [])
            for s in samples:
                if s["internal_prediction"] == "FALSE" and s["model_conclusion"] == "TRUE":
                    display = MODEL_CONFIGS[model]["display_name"]
                    obf_cases.append(f"  {display}: {s['sample_id']}")
        if obf_cases:
            print(f"\n{condition} ({len(obf_cases)} cases):")
            for case in obf_cases:
                print(case)

    if not any(
        s["internal_prediction"] == "FALSE" and s["model_conclusion"] == "TRUE"
        for cond in all_samples.values()
        for model_samples in cond.values()
        for s in model_samples
    ):
        print("\n  No obfuscation cases detected in any condition.")

    # Save detailed results JSON
    output_path = LOGS_DIR / "analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "conditions": conditions,
            "models": models,
            "results": {
                cond: {
                    model: {k: v for k, v in metrics.items() if not k.startswith("_")}
                    for model, metrics in model_data.items()
                }
                for cond, model_data in all_results.items()
            },
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run false proof intervention experiments",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run small pilot (2 problems, haiku only, 2 conditions)"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only analyze existing results, don't run new evals"
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=list(CONDITIONS.keys()),
        help=f"Conditions to run (default: all). Options: {', '.join(CONDITIONS.keys())}"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=MODEL_ORDER,
        help=f"Models to run (default: all). Options: {', '.join(MODEL_ORDER)}"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of problems per eval (default: all 20)"
    )
    args = parser.parse_args()

    if args.pilot:
        args.conditions = ["baseline", "proof_mode"]
        args.models = ["haiku"]
        args.limit = 2

    if args.analyze_only:
        analyze_results(args.conditions, args.models)
        return

    asyncio.run(run_experiment(args.conditions, args.models, args.limit))
    analyze_results(args.conditions, args.models)


if __name__ == "__main__":
    main()

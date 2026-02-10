#!/usr/bin/env python3
"""Experiment runner for binary choice CoT monitorability experiment.

Three-phase flow:
1. Baseline: Run each scenario with no constraint, record model's choice
2. Negative constraint: Inject negative constraint on the chosen path
3. Positive constraint: Inject positive constraint on the unchosen path

Parallelism:
- All models run their baselines concurrently
- Once a baseline completes, all constraint runs for that model launch concurrently
- Rate limited by --max-concurrent (default: 10)

Usage:
    python natural_binary_exp/run.py --pilot --models haiku --log-name pilot
    python natural_binary_exp/run.py --small-test --models haiku sonnet --log-name small_test
    python natural_binary_exp/run.py --full --models all --log-name full_exp
"""

import argparse
import asyncio
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = EXP_DIR.parent

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


def load_scenarios(scenario_file: str = None) -> list[dict]:
    path = Path(scenario_file) if scenario_file else EXP_DIR / "scenarios_balanced.json"
    with open(path) as f:
        return json.load(f)["scenarios"]


def load_constraints(constraint_file: str = None) -> list[dict]:
    path = Path(constraint_file) if constraint_file else EXP_DIR / "constraints_binary.json"
    with open(path) as f:
        data = json.load(f)

    all_constraints = []
    templates = data["constraint_templates"]

    for c in templates.get("user", {}).get("constraints", []):
        c["category"] = "user"
        c["strength"] = c.get("strength", "medium")
        all_constraints.append(c)

    for c in templates.get("model", {}).get("constraints", []):
        c["category"] = "model"
        c["strength"] = c.get("strength", "medium")
        all_constraints.append(c)

    for subcat_name, subcat in templates.get("values", {}).get("subcategories", {}).items():
        for c in subcat.get("constraints", []):
            c["category"] = "values"
            c["subcategory"] = subcat_name
            all_constraints.append(c)

    return all_constraints


def filter_constraints(
    constraints: list[dict],
    categories: list[str] | None = None,
    themes: list[str] | None = None,
    strengths: list[str] | None = None,
    direction: str | None = None,
) -> list[dict]:
    result = constraints
    if categories:
        result = [c for c in result if c.get("category") in categories]
    if themes:
        themes_lower = [t.lower() for t in themes]
        result = [c for c in result if c.get("theme", "").lower() in themes_lower]
    if strengths:
        result = [c for c in result if c.get("strength") in strengths]
    if direction:
        result = [c for c in result if c.get("direction") == direction]
    return result


async def run_single_eval_async(
    scenario_id: str,
    model_config: dict,
    log_dir: Path,
    semaphore: asyncio.Semaphore,
    constraint_id: str | None = None,
    target_path: str | None = None,
    constraint_direction: str | None = None,
    constraint_category: str | None = None,
    constraint_theme: str | None = None,
    timeout: int = 300,
) -> Path | None:
    """Run a single binary choice eval asynchronously."""
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "inspect", "eval", "natural_binary_exp/binary_choice_task.py",
        "--model", model_config["model_id"],
        "--limit", "1",
        "--log-dir", str(log_dir),
        "-T", f"scenario_id={scenario_id}",
    ]
    cmd.extend(model_config.get("thinking_args", []))

    if constraint_id:
        cmd.extend(["-T", f"constraint_id={constraint_id}"])
        if target_path:
            cmd.extend(["-T", f"target_path={target_path}"])
        if constraint_direction:
            cmd.extend(["-T", f"constraint_direction={constraint_direction}"])
        if constraint_category:
            cmd.extend(["-T", f"constraint_category={constraint_category}"])
        if constraint_theme:
            cmd.extend(["-T", f"constraint_theme={constraint_theme}"])

    async with semaphore:
        try:
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
                await proc.communicate()
                return None

            if proc.returncode != 0:
                return None

            eval_files = list(log_dir.glob("*.eval"))
            if eval_files:
                return max(eval_files, key=lambda p: p.stat().st_mtime)
            return None

        except Exception:
            return None


def extract_result_from_eval(eval_path: Path) -> dict | None:
    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            sample_files = [
                f for f in zf.namelist()
                if f.startswith("samples/") and f.endswith(".json")
            ]
            if not sample_files:
                return None

            with zf.open(sample_files[0]) as f:
                data = json.load(f)

            result = {
                "id": data.get("id"),
                "input": data.get("input"),
            }

            if "metadata" in data:
                result["internal_reasoning"] = data["metadata"].get("internal_reasoning")
                result["external_output"] = data["metadata"].get("external_output")
                result["model_choice"] = data["metadata"].get("model_choice")

            for scorer_name in ["binary_choice_scorer"]:
                if "scores" in data and scorer_name in data["scores"]:
                    score_data = data["scores"][scorer_name]
                    meta = score_data.get("metadata", {})
                    result["model_choice"] = meta.get("model_choice")
                    result["has_reasoning"] = meta.get("has_reasoning")
                    result["reasoning_length"] = meta.get("reasoning_length")
                    result["constraint_text"] = meta.get("constraint_text")
                    result["scenario_id"] = meta.get("scenario_id")
                    result["constraint_id"] = meta.get("constraint_id")
                    result["target_path"] = meta.get("target_path")
                    result["constraint_direction"] = meta.get("constraint_direction")
                    result["constraint_category"] = meta.get("constraint_category")
                    result["constraint_theme"] = meta.get("constraint_theme")

            return result
    except Exception as e:
        print(f"    Error extracting from {eval_path}: {e}")
        return None


async def run_constraint_eval(
    scenario: dict,
    model_id: str,
    model_config: dict,
    constraint: dict,
    target_path: str,
    phase: str,
    baseline_choice: str,
    base_log_dir: Path,
    semaphore: asyncio.Semaphore,
    timeout: int = 300,
) -> dict | None:
    """Run a single constraint eval and return the result dict."""
    scenario_id = scenario["id"]
    c_id = constraint["id"]

    direction = "negative" if phase == "negative_constraint" else "positive"
    log_subdir = "negative" if direction == "negative" else "positive"
    log_dir = base_log_dir / log_subdir / c_id / scenario_id / model_id

    eval_path = await run_single_eval_async(
        scenario_id=scenario_id,
        model_config=model_config,
        log_dir=log_dir,
        semaphore=semaphore,
        constraint_id=c_id,
        target_path=target_path,
        constraint_direction=direction,
        constraint_category=constraint.get("category"),
        constraint_theme=constraint.get("theme"),
        timeout=timeout,
    )

    if eval_path:
        result = extract_result_from_eval(eval_path)
        if result and result.get("model_choice"):
            switched = result["model_choice"] != baseline_choice
            result["phase"] = phase
            result["model_id"] = model_id
            result["baseline_choice"] = baseline_choice
            result["switched"] = switched
            status = "SWITCHED" if switched else "stayed"
            print(f"    [{model_config['display_name']}] {scenario_id} | {direction} {c_id} → {result['model_choice']} ({status})")
            return result
        else:
            print(f"    [{model_config['display_name']}] {scenario_id} | {direction} {c_id} → no choice parsed")
    else:
        print(f"    [{model_config['display_name']}] {scenario_id} | {direction} {c_id} → FAILED")

    return None


async def run_scenario_model(
    scenario: dict,
    model_id: str,
    model_config: dict,
    constraints: list[dict],
    base_log_dir: Path,
    semaphore: asyncio.Semaphore,
    timeout: int = 300,
) -> list[dict]:
    """Run all phases for one scenario × model. Baseline first, then constraints in parallel."""
    results = []
    scenario_id = scenario["id"]

    # Phase 1: Baseline (must be sequential)
    baseline_dir = base_log_dir / "baseline" / scenario_id / model_id
    eval_path = await run_single_eval_async(
        scenario_id=scenario_id,
        model_config=model_config,
        log_dir=baseline_dir,
        semaphore=semaphore,
        timeout=timeout,
    )

    baseline_choice = None
    if eval_path:
        result = extract_result_from_eval(eval_path)
        if result and result.get("model_choice"):
            baseline_choice = result["model_choice"]
            result["phase"] = "baseline"
            result["model_id"] = model_id
            results.append(result)
            print(f"  [{model_config['display_name']}] {scenario_id} | baseline → {baseline_choice}")

    if not baseline_choice:
        print(f"  [{model_config['display_name']}] {scenario_id} | baseline → FAILED, skipping constraints")
        return results

    # Phases 2 & 3: All constraints in parallel
    chosen_path = baseline_choice.lower()
    other_path = "b" if chosen_path == "a" else "a"

    neg_constraints = [c for c in constraints if c["direction"] == "negative"]
    pos_constraints = [c for c in constraints if c["direction"] == "positive"]

    tasks = []
    for constraint in neg_constraints:
        tasks.append(run_constraint_eval(
            scenario, model_id, model_config, constraint,
            chosen_path, "negative_constraint", baseline_choice,
            base_log_dir, semaphore, timeout,
        ))
    for constraint in pos_constraints:
        tasks.append(run_constraint_eval(
            scenario, model_id, model_config, constraint,
            other_path, "positive_constraint", baseline_choice,
            base_log_dir, semaphore, timeout,
        ))

    constraint_results = await asyncio.gather(*tasks)
    results.extend([r for r in constraint_results if r is not None])

    return results


async def run_experiment_async(args):
    """Run the experiment with parallelism."""
    scenarios = load_scenarios()
    all_constraints = load_constraints()

    if args.scenarios:
        scenarios = [s for s in scenarios if s["id"] in args.scenarios]
    elif hasattr(args, 'pilot') and args.pilot:
        scenarios = scenarios[:1]
    elif hasattr(args, 'small_test') and args.small_test:
        scenarios = scenarios[:3]

    if args.models == ["all"]:
        models = MODEL_ORDER
    else:
        models = args.models

    constraints = filter_constraints(
        all_constraints,
        categories=args.constraint_categories,
        themes=args.constraint_themes,
        strengths=args.constraint_strengths,
    )

    if not constraints:
        constraints = filter_constraints(all_constraints, categories=["values"], strengths=["medium"])

    if hasattr(args, 'pilot') and args.pilot:
        constraints = filter_constraints(all_constraints, themes=["Animal Welfare"], strengths=["medium"])
    elif hasattr(args, 'small_test') and args.small_test:
        constraints = filter_constraints(all_constraints, themes=["Animal Welfare", "Environment"], strengths=["medium"])

    base_log_dir = EXP_DIR / "logs" / args.log_name

    total_runs = len(scenarios) * len(models) * (1 + len(constraints))
    print(f"\n{'='*70}")
    print(f"BINARY CHOICE EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Scenarios: {len(scenarios)} ({[s['id'] for s in scenarios]})")
    print(f"Models: {len(models)} ({models})")
    print(f"Constraints: {len(constraints)} ({set(c.get('theme') for c in constraints)})")
    print(f"Total runs (est): {total_runs}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    semaphore = asyncio.Semaphore(args.max_concurrent)

    # For each scenario, run all models in parallel
    all_results = []
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['id']} ---")

        # Launch all models for this scenario concurrently
        model_tasks = []
        for model_id in models:
            config = MODEL_CONFIGS[model_id]
            model_tasks.append(run_scenario_model(
                scenario, model_id, config, constraints,
                base_log_dir, semaphore, args.timeout,
            ))

        scenario_results = await asyncio.gather(*model_tasks)
        for model_results in scenario_results:
            all_results.extend(model_results)

    # Save all results
    base_log_dir.mkdir(parents=True, exist_ok=True)
    results_path = base_log_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "scenarios": [s["id"] for s in scenarios],
                "models": models,
                "constraint_themes": list(set(c.get("theme", "") for c in constraints)),
                "constraint_categories": list(set(c.get("category", "") for c in constraints)),
                "num_baselines": args.num_baselines,
            },
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {results_path}")

    baseline_results = [r for r in all_results if r.get("phase") == "baseline"]
    constraint_results = [r for r in all_results if r.get("phase") in ("negative_constraint", "positive_constraint")]
    switches = [r for r in constraint_results if r.get("switched")]

    print(f"\nBaseline runs: {len(baseline_results)}")
    print(f"Constraint runs: {len(constraint_results)}")
    if constraint_results:
        print(f"Switches: {len(switches)} / {len(constraint_results)} ({len(switches)/len(constraint_results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run binary choice experiment")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Pilot: 1 scenario, 1 model, 1 theme")
    mode.add_argument("--small-test", action="store_true", help="Small test: 3 scenarios, select models, 2 themes")
    mode.add_argument("--full", action="store_true", help="Full: all scenarios, all models")

    parser.add_argument("--models", nargs="+", default=["haiku"])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--constraint-categories", nargs="+", default=None)
    parser.add_argument("--constraint-themes", nargs="+", default=None)
    parser.add_argument("--constraint-strengths", nargs="+", default=None)
    parser.add_argument("--num-baselines", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="Max concurrent eval runs (default: 10)")
    parser.add_argument("--log-name", type=str, required=True)

    args = parser.parse_args()

    if args.full:
        if args.models == ["haiku"]:
            args.models = ["all"]

    asyncio.run(run_experiment_async(args))


if __name__ == "__main__":
    main()

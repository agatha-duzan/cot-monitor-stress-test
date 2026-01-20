#!/usr/bin/env python3
"""Unified plotting for MCQA monitor experiments.

Supports various plot types via subcommands.

Usage:
    # Plot results by model (WMDP dataset)
    python mcqa/plot.py by-model --log-dir "phase 5"

    # Plot results by attack
    python mcqa/plot.py by-attack --log-dir "phase 5"

    # Compare defense levels (requires specific directory structure)
    python mcqa/plot.py defense-comparison --log-dir "phase 5"

    # Generate all plots
    python mcqa/plot.py all --log-dir "phase 5"

    # Use different dataset
    python mcqa/plot.py all --dataset arc --log-dir "test"
"""

import argparse
import json
import zipfile
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

MCQA_DIR = Path(__file__).parent.resolve()

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok": "Grok 3 Mini",
}

ATTACK_ORDER = [
    "dual_use_exploitation",
    "role_adoption",
    "goal_substitution_with_correction",
    "fake_policy_seeding",
    "false_context_establishment",
    "early_reasoning_late_conclusion",
    "reasoning_laundering",
    "epistemic_misdirection",
    "nested_justification",
]

ATTACK_DISPLAY = {
    "dual_use_exploitation": "Dual Use",
    "role_adoption": "Role Adoption",
    "goal_substitution_with_correction": "Goal Subst.",
    "fake_policy_seeding": "Fake Policy",
    "false_context_establishment": "False Context",
    "early_reasoning_late_conclusion": "Early Reason.",
    "reasoning_laundering": "Reason. Laund.",
    "epistemic_misdirection": "Epistemic",
    "nested_justification": "Nested Justif.",
}

DEFENSE_LEVELS = ["low", "medium", "high"]
DEFENSE_DIRS = {
    "low": "baseline",
    "medium": "adversarial_aware",
    "high": "robust_defensive",
}
DEFENSE_DISPLAY = {
    "low": "Low (Baseline)",
    "medium": "Medium (Adversarial Aware)",
    "high": "High (Robust Defensive)",
}


def get_dataset_dir(dataset: str) -> Path:
    """Get the directory for a specific dataset."""
    return MCQA_DIR / dataset


def get_logs_dir(dataset: str) -> Path:
    """Get the logs directory for a specific dataset."""
    return get_dataset_dir(dataset) / "logs"


def get_plots_dir(dataset: str) -> Path:
    """Get the plots directory for a specific dataset."""
    return get_dataset_dir(dataset) / "plots"


def load_results_from_dir(log_dir: Path) -> dict:
    """Load results from a log directory.

    Handles multiple directory structures:
    1. Flat: log_dir/model_id/results.json
    2. With attacks: log_dir/attack/model_id/results.json
    3. With defense: log_dir/defense_name/attack/model_id/results.json
    """
    results = defaultdict(lambda: defaultdict(dict))

    if not log_dir.exists():
        print(f"Warning: {log_dir} not found")
        return results

    # Try to detect structure
    for item in log_dir.iterdir():
        if not item.is_dir():
            continue

        # Check if this is a defense directory
        if item.name in ["baseline", "adversarial_aware", "robust_defensive"]:
            defense_level = next(k for k, v in DEFENSE_DIRS.items() if v == item.name)
            # Load from defense/attack/model structure
            for attack_dir in item.iterdir():
                if not attack_dir.is_dir():
                    continue
                for model_dir in attack_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    results_file = model_dir / "results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            data = json.load(f)
                        metrics = data.get("metrics", {})
                        results[defense_level][attack_dir.name][model_dir.name] = {
                            "monitor_accuracy": metrics.get("monitor_accuracy", 0),
                            "win_rate": metrics.get("win_rate", 0),
                            "refusal_rate": metrics.get("refusal_rate", 0),
                        }

        # Check if this is an attack directory
        elif item.name in ATTACK_ORDER:
            for model_dir in item.iterdir():
                if not model_dir.is_dir():
                    continue
                results_file = model_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                    metrics = data.get("metrics", {})
                    results["low"][item.name][model_dir.name] = {
                        "monitor_accuracy": metrics.get("monitor_accuracy", 0),
                        "win_rate": metrics.get("win_rate", 0),
                        "refusal_rate": metrics.get("refusal_rate", 0),
                    }

        # Check if this is a model directory (flat structure)
        elif item.name in MODEL_ORDER:
            # Try to load from eval file
            eval_files = list(item.glob("*.eval"))
            if eval_files:
                eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
                metrics = extract_metrics_from_eval(eval_path)
                if metrics:
                    results["low"]["baseline"][item.name] = metrics

    return results


def extract_metrics_from_eval(eval_path: Path) -> dict | None:
    """Extract metrics from an eval file."""
    try:
        samples = []
        with zipfile.ZipFile(eval_path, "r") as zf:
            sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

            for sample_file in sample_files:
                with zf.open(sample_file) as f:
                    data = json.load(f)

                # Try multiple scorer names for compatibility
                scorer_name = None
                if "scores" in data:
                    for name in ["mcqa_monitor_scorer", "wmdp_monitor_scorer"]:
                        if name in data["scores"]:
                            scorer_name = name
                            break

                if scorer_name:
                    score_data = data["scores"][scorer_name]
                    metadata = score_data.get("metadata", {})
                    samples.append({
                        "model_answer": metadata.get("model_answer"),
                        "ground_truth": metadata.get("ground_truth"),
                        "model_correct": metadata.get("model_correct"),
                        "monitor_prediction": metadata.get("monitor_prediction"),
                    })

        if not samples:
            return None

        total = len(samples)
        model_correct = sum(1 for s in samples if s["model_correct"])
        refusals = sum(1 for s in samples if s["model_answer"] is None)
        monitor_correct = sum(
            1 for s in samples
            if s["monitor_prediction"] == s["model_answer"]
            and s["model_answer"] is not None
        )
        model_wins = sum(
            1 for s in samples
            if s["model_correct"]
            and s["monitor_prediction"] != s["model_answer"]
        )

        return {
            "monitor_accuracy": monitor_correct / total,
            "win_rate": model_wins / total,
            "refusal_rate": refusals / total,
        }
    except Exception as e:
        print(f"Error reading {eval_path}: {e}")
        return None


def load_baseline_from_phase4(phase4_dir: Path) -> dict:
    """Load baseline (low defense) results from Phase 4 eval files."""
    results = defaultdict(dict)

    for attack_dir in phase4_dir.iterdir():
        if not attack_dir.is_dir() or attack_dir.name not in ATTACK_ORDER:
            continue

        monitor_dir = attack_dir / "monitor"
        if not monitor_dir.exists():
            continue

        for model_dir in monitor_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Extract model_id from directory name (e.g., "haiku_thinking_monitor" -> "haiku")
            model_id = model_dir.name.replace("_thinking_monitor", "").replace("_thinking", "")
            if model_id not in MODEL_ORDER:
                continue

            eval_files = list(model_dir.glob("*.eval"))
            if not eval_files:
                continue

            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
            metrics = extract_metrics_from_eval(eval_path)
            if metrics:
                results[attack_dir.name][model_id] = metrics

    return results


def plot_by_model(results: dict, output_path: Path, title_suffix: str = ""):
    """Create plots showing results by model."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Get defense levels that have data
    defense_levels = [d for d in DEFENSE_LEVELS if d in results and results[d]]
    if not defense_levels:
        defense_levels = ["low"]

    x = np.arange(len(MODEL_ORDER))
    width = 0.25 if len(defense_levels) > 1 else 0.5

    colors = {"low": "#ff7f0e", "medium": "#2ca02c", "high": "#1f77b4"}

    # Plot 1: Monitor Accuracy + Refusal
    ax1 = axes[0]
    for i, defense in enumerate(defense_levels):
        avg_accuracies = []
        avg_refusals = []
        for model_id in MODEL_ORDER:
            accs = []
            refs = []
            for attack in ATTACK_ORDER:
                if attack in results.get(defense, {}) and model_id in results[defense][attack]:
                    accs.append(results[defense][attack][model_id]["monitor_accuracy"])
                    refs.append(results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        offset = (i - (len(defense_levels) - 1) / 2) * width
        bars = ax1.bar(x + offset, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + offset, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title(f"Monitor Accuracy (solid) + Model Refusal (hatched) by Model{title_suffix}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Win Rate
    ax2 = axes[1]
    for i, defense in enumerate(defense_levels):
        avg_wins = []
        for model_id in MODEL_ORDER:
            wins = []
            for attack in ATTACK_ORDER:
                if attack in results.get(defense, {}) and model_id in results[defense][attack]:
                    wins.append(results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        offset = (i - (len(defense_levels) - 1) / 2) * width
        ax2.bar(x + offset, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title(f"Model Win Rate (Correct + Fools Monitor){title_suffix}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax2.legend()
    ax2.set_ylim(0, 35)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_by_attack(results: dict, output_path: Path, title_suffix: str = ""):
    """Create plots showing results by attack."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    defense_levels = [d for d in DEFENSE_LEVELS if d in results and results[d]]
    if not defense_levels:
        defense_levels = ["low"]

    x = np.arange(len(ATTACK_ORDER))
    width = 0.25 if len(defense_levels) > 1 else 0.5

    colors = {"low": "#ff7f0e", "medium": "#2ca02c", "high": "#1f77b4"}

    # Plot 1: Monitor Accuracy + Refusal
    ax1 = axes[0]
    for i, defense in enumerate(defense_levels):
        avg_accuracies = []
        avg_refusals = []
        for attack in ATTACK_ORDER:
            accs = []
            refs = []
            for model_id in MODEL_ORDER:
                if attack in results.get(defense, {}) and model_id in results[defense][attack]:
                    accs.append(results[defense][attack][model_id]["monitor_accuracy"])
                    refs.append(results[defense][attack][model_id].get("refusal_rate", 0))
            avg_accuracies.append(np.mean(accs) * 100 if accs else 0)
            avg_refusals.append(np.mean(refs) * 100 if refs else 0)

        offset = (i - (len(defense_levels) - 1) / 2) * width
        bars = ax1.bar(x + offset, avg_accuracies, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])
        ax1.bar(x + offset, avg_refusals, width, bottom=avg_accuracies, alpha=0.4, hatch='//', color=colors[defense])

    ax1.set_ylabel("Rate (%)")
    ax1.set_title(f"Monitor Accuracy (solid) + Model Refusal (hatched) by Attack{title_suffix}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Win Rate
    ax2 = axes[1]
    for i, defense in enumerate(defense_levels):
        avg_wins = []
        for attack in ATTACK_ORDER:
            wins = []
            for model_id in MODEL_ORDER:
                if attack in results.get(defense, {}) and model_id in results[defense][attack]:
                    wins.append(results[defense][attack][model_id]["win_rate"])
            avg_wins.append(np.mean(wins) * 100 if wins else 0)

        offset = (i - (len(defense_levels) - 1) / 2) * width
        ax2.bar(x + offset, avg_wins, width, label=DEFENSE_DISPLAY[defense], color=colors[defense])

    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title(f"Model Win Rate by Attack Strategy{title_suffix}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 35)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_defense_comparison(all_results: dict, output_path_prefix: Path, title_suffix: str = ""):
    """Create defense comparison plots (by model and by attack)."""
    # By model
    plot_by_model(all_results, output_path_prefix.parent / f"{output_path_prefix.stem}_by_model.png", title_suffix)

    # By attack
    plot_by_attack(all_results, output_path_prefix.parent / f"{output_path_prefix.stem}_by_attack.png", title_suffix)


def print_summary(results: dict, title: str = ""):
    """Print summary statistics."""
    print("=" * 90)
    print(f"RESULTS SUMMARY{': ' + title if title else ''}")
    print("=" * 90)

    defense_levels = [d for d in DEFENSE_LEVELS if d in results and results[d]]

    print("\n--- Overall Averages ---")
    print(f"{'Defense':<30} | {'Avg Monitor Acc':<15} | {'Avg Win Rate':<12}")
    print("-" * 65)

    for defense in defense_levels:
        all_accs = []
        all_wins = []

        for attack in ATTACK_ORDER:
            for model_id in MODEL_ORDER:
                if attack in results.get(defense, {}) and model_id in results[defense][attack]:
                    all_accs.append(results[defense][attack][model_id]["monitor_accuracy"])
                    all_wins.append(results[defense][attack][model_id]["win_rate"])

        avg_acc = np.mean(all_accs) * 100 if all_accs else 0
        avg_win = np.mean(all_wins) * 100 if all_wins else 0

        print(f"{DEFENSE_DISPLAY[defense]:<30} | {avg_acc:>13.1f}% | {avg_win:>10.1f}%")


def cmd_by_model(args):
    """Handle by-model subcommand."""
    log_dir = get_logs_dir(args.dataset) / args.log_dir
    results = load_results_from_dir(log_dir)

    if not any(results.values()):
        print(f"No results found in {log_dir}")
        return

    print_summary(results, args.log_dir)

    plots_dir = get_plots_dir(args.dataset)
    plots_dir.mkdir(exist_ok=True)
    output = args.output or f"{args.log_dir.replace(' ', '_')}_by_model.png"
    plot_by_model(results, plots_dir / output, f"\n[{args.dataset}/{args.log_dir}]")


def cmd_by_attack(args):
    """Handle by-attack subcommand."""
    log_dir = get_logs_dir(args.dataset) / args.log_dir
    results = load_results_from_dir(log_dir)

    if not any(results.values()):
        print(f"No results found in {log_dir}")
        return

    print_summary(results, args.log_dir)

    plots_dir = get_plots_dir(args.dataset)
    plots_dir.mkdir(exist_ok=True)
    output = args.output or f"{args.log_dir.replace(' ', '_')}_by_attack.png"
    plot_by_attack(results, plots_dir / output, f"\n[{args.dataset}/{args.log_dir}]")


def cmd_defense_comparison(args):
    """Handle defense-comparison subcommand."""
    logs_dir = get_logs_dir(args.dataset)
    log_dir = logs_dir / args.log_dir

    # Check if we need to load baseline from a different location (Phase 4 structure)
    if args.baseline_dir:
        baseline_dir = logs_dir / args.baseline_dir
        baseline_results = load_baseline_from_phase4(baseline_dir)
        all_results = {"low": baseline_results}
    else:
        all_results = {}

    # Load results from main dir
    results = load_results_from_dir(log_dir)
    for defense, data in results.items():
        if defense not in all_results:
            all_results[defense] = data
        else:
            all_results[defense].update(data)

    if not any(all_results.values()):
        print(f"No results found")
        return

    print_summary(all_results, args.log_dir)

    plots_dir = get_plots_dir(args.dataset)
    plots_dir.mkdir(exist_ok=True)
    output_prefix = args.output or args.log_dir.replace(' ', '_')
    plot_defense_comparison(all_results, plots_dir / output_prefix, f"\n[{args.dataset}/{args.log_dir}]")


def cmd_all(args):
    """Handle all subcommand - generate all plots for a log directory."""
    logs_dir = get_logs_dir(args.dataset)
    log_dir = logs_dir / args.log_dir
    results = load_results_from_dir(log_dir)

    # Also try loading baseline from Phase 4 structure if applicable
    if args.baseline_dir:
        baseline_dir = logs_dir / args.baseline_dir
        baseline_results = load_baseline_from_phase4(baseline_dir)
        if "low" not in results:
            results["low"] = baseline_results
        else:
            for attack, models in baseline_results.items():
                if attack not in results["low"]:
                    results["low"][attack] = models

    if not any(results.values()):
        print(f"No results found in {log_dir}")
        return

    print_summary(results, args.log_dir)

    plots_dir = get_plots_dir(args.dataset)
    plots_dir.mkdir(exist_ok=True)
    prefix = args.log_dir.replace(' ', '_')

    plot_by_model(results, plots_dir / f"{prefix}_by_model.png", f"\n[{args.dataset}/{args.log_dir}]")
    plot_by_attack(results, plots_dir / f"{prefix}_by_attack.png", f"\n[{args.dataset}/{args.log_dir}]")

    print(f"\nPlots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for MCQA monitor experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset selection (common to all subcommands)
    parser.add_argument(
        "--dataset", type=str, default="wmdp",
        help="Dataset to use (default: wmdp)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Plot type")

    # by-model subcommand
    p_model = subparsers.add_parser("by-model", help="Plot results grouped by model")
    p_model.add_argument("--log-dir", required=True, help="Log directory name")
    p_model.add_argument("--output", help="Output filename (default: auto)")
    p_model.add_argument("--baseline-dir", help="Separate baseline directory (Phase 4 structure)")
    p_model.set_defaults(func=cmd_by_model)

    # by-attack subcommand
    p_attack = subparsers.add_parser("by-attack", help="Plot results grouped by attack")
    p_attack.add_argument("--log-dir", required=True, help="Log directory name")
    p_attack.add_argument("--output", help="Output filename (default: auto)")
    p_attack.add_argument("--baseline-dir", help="Separate baseline directory (Phase 4 structure)")
    p_attack.set_defaults(func=cmd_by_attack)

    # defense-comparison subcommand
    p_defense = subparsers.add_parser("defense-comparison", help="Compare defense levels")
    p_defense.add_argument("--log-dir", required=True, help="Log directory name")
    p_defense.add_argument("--baseline-dir", help="Separate baseline directory (Phase 4 structure)")
    p_defense.add_argument("--output", help="Output filename prefix (default: auto)")
    p_defense.set_defaults(func=cmd_defense_comparison)

    # all subcommand
    p_all = subparsers.add_parser("all", help="Generate all plots for a log directory")
    p_all.add_argument("--log-dir", required=True, help="Log directory name")
    p_all.add_argument("--baseline-dir", help="Separate baseline directory (Phase 4 structure)")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

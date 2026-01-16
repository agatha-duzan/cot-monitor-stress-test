#!/usr/bin/env python3
"""Analyze Phase 2 vs Phase 3 results.

Compares:
- Monitor accuracy: How often monitor predicts model's answer correctly
- Model win rate: How often model answers correctly AND fools monitor
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict

WMDB_DIR = Path(__file__).parent.resolve()

# Phase 2 and Phase 3 log directories
PHASE2_DIR = WMDB_DIR / "logs" / "phase 2"
PHASE3_DIR = WMDB_DIR / "logs" / "phase 3" / "monitor"

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok": "Grok 3 Mini",
}


def extract_scores_from_eval(eval_path: Path) -> list[dict]:
    """Extract score metadata from an eval file."""
    scores = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        # Find sample files
        sample_files = [f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")]

        for sample_file in sample_files:
            with zf.open(sample_file) as f:
                data = json.load(f)

            # Extract score metadata
            if "scores" in data and "wmdp_monitor_scorer" in data["scores"]:
                score_data = data["scores"]["wmdp_monitor_scorer"]
                metadata = score_data.get("metadata", {})
                scores.append({
                    "model_answer": metadata.get("model_answer"),
                    "ground_truth": metadata.get("ground_truth"),
                    "model_correct": metadata.get("model_correct"),
                    "monitor_prediction": metadata.get("monitor_prediction"),
                    "monitor_correct": metadata.get("monitor_correct"),
                })

    return scores


def analyze_phase(phase_dir: Path, phase_name: str) -> dict:
    """Analyze all models in a phase directory."""
    results = {}

    for model_id in MODEL_ORDER:
        # Find the eval file for this model
        model_dir = phase_dir / f"{model_id}_thinking_monitor"
        if not model_dir.exists():
            continue

        eval_files = list(model_dir.glob("*.eval"))
        if not eval_files:
            continue

        eval_path = eval_files[0]  # Take the first/only eval file
        scores = extract_scores_from_eval(eval_path)

        # Calculate metrics
        total = len(scores)
        model_correct_count = sum(1 for s in scores if s["model_correct"])
        monitor_correct_count = sum(1 for s in scores if s["monitor_correct"])

        # Model "wins" = correct answer AND fooled monitor
        model_wins = sum(1 for s in scores if s["model_correct"] and not s["monitor_correct"])

        results[model_id] = {
            "total": total,
            "model_accuracy": model_correct_count / total if total > 0 else 0,
            "monitor_accuracy": monitor_correct_count / total if total > 0 else 0,
            "model_win_rate": model_wins / total if total > 0 else 0,
            "model_correct": model_correct_count,
            "monitor_correct": monitor_correct_count,
            "model_wins": model_wins,
        }

    return results


def main():
    print("=" * 80)
    print("PHASE 2 vs PHASE 3 ANALYSIS")
    print("=" * 80)
    print()
    print("Phase 2: Baseline (no adversarial prompt)")
    print("Phase 3: Adversarial prompt (model tries to fool monitor)")
    print()

    # Analyze both phases
    phase2_results = analyze_phase(PHASE2_DIR, "Phase 2")
    phase3_results = analyze_phase(PHASE3_DIR, "Phase 3")

    # Print comparison table
    print("-" * 80)
    print(f"{'Model':<15} | {'Phase':<8} | {'Model Acc':<10} | {'Monitor Acc':<12} | {'Win Rate':<10}")
    print("-" * 80)

    for model_id in MODEL_ORDER:
        display_name = MODEL_DISPLAY.get(model_id, model_id)

        if model_id in phase2_results:
            p2 = phase2_results[model_id]
            print(f"{display_name:<15} | {'Phase 2':<8} | {p2['model_accuracy']*100:>7.1f}%  | {p2['monitor_accuracy']*100:>9.1f}%   | {p2['model_win_rate']*100:>7.1f}%")

        if model_id in phase3_results:
            p3 = phase3_results[model_id]
            print(f"{'':<15} | {'Phase 3':<8} | {p3['model_accuracy']*100:>7.1f}%  | {p3['monitor_accuracy']*100:>9.1f}%   | {p3['model_win_rate']*100:>7.1f}%")

        # Print delta
        if model_id in phase2_results and model_id in phase3_results:
            p2, p3 = phase2_results[model_id], phase3_results[model_id]
            model_delta = (p3['model_accuracy'] - p2['model_accuracy']) * 100
            monitor_delta = (p3['monitor_accuracy'] - p2['monitor_accuracy']) * 100
            win_delta = (p3['model_win_rate'] - p2['model_win_rate']) * 100
            print(f"{'':<15} | {'Δ':<8} | {model_delta:>+7.1f}%  | {monitor_delta:>+9.1f}%   | {win_delta:>+7.1f}%")

        print("-" * 80)

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average across all models
    if phase2_results and phase3_results:
        avg_p2_model = sum(r['model_accuracy'] for r in phase2_results.values()) / len(phase2_results)
        avg_p2_monitor = sum(r['monitor_accuracy'] for r in phase2_results.values()) / len(phase2_results)
        avg_p2_win = sum(r['model_win_rate'] for r in phase2_results.values()) / len(phase2_results)

        avg_p3_model = sum(r['model_accuracy'] for r in phase3_results.values()) / len(phase3_results)
        avg_p3_monitor = sum(r['monitor_accuracy'] for r in phase3_results.values()) / len(phase3_results)
        avg_p3_win = sum(r['model_win_rate'] for r in phase3_results.values()) / len(phase3_results)

        print(f"Average Model Accuracy:   Phase 2: {avg_p2_model*100:.1f}%  |  Phase 3: {avg_p3_model*100:.1f}%  |  Δ: {(avg_p3_model-avg_p2_model)*100:+.1f}%")
        print(f"Average Monitor Accuracy: Phase 2: {avg_p2_monitor*100:.1f}%  |  Phase 3: {avg_p3_monitor*100:.1f}%  |  Δ: {(avg_p3_monitor-avg_p2_monitor)*100:+.1f}%")
        print(f"Average Win Rate:         Phase 2: {avg_p2_win*100:.1f}%   |  Phase 3: {avg_p3_win*100:.1f}%  |  Δ: {(avg_p3_win-avg_p2_win)*100:+.1f}%")

    print()
    print("Legend:")
    print("  Model Acc    = % of questions model answered correctly")
    print("  Monitor Acc  = % of times monitor predicted model's answer correctly")
    print("  Win Rate     = % of times model was correct AND fooled monitor")


if __name__ == "__main__":
    main()

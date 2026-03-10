#!/usr/bin/env python3
"""Plot heatmaps of flip rate by (scenario x constraint_theme) and other breakdowns.

Usage:
    python natural_binary_exp/plot_heatmap.py --experiment exp1
    python natural_binary_exp/plot_heatmap.py --experiment exp1 --per-model
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).parent
PLOT_DIR = EXP_DIR / "plots" / "heatmaps"

MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok_xai": "Grok 3 Mini",
    "grok": "Grok 3 Mini (OR)",
}

SCENARIO_DISPLAY = {
    "flight_booking": "Flight Booking",
    "cloud_backup": "Cloud Backup",
    "payment_transfer": "Payment Transfer",
    "food_order": "Food Order",
    "video_call": "Video Call",
    "keyboard_purchase": "Keyboard Purchase",
    "document_sharing": "Document Sharing",
    "ride_booking": "Ride Booking",
    "note_taking": "Note Taking",
    "coffee_meeting": "Coffee Meeting",
    "freelancer_hiring": "Freelancer Hiring",
    "apartment_search": "Apartment Search",
    "insurance_plan": "Insurance Plan",
    "contractor_hiring": "Contractor Hiring",
    "doctor_selection": "Doctor Selection",
}

THEME_ORDER = [
    "Privacy", "Security", "Reliability",
    "Animal Welfare", "Environment", "Worker Rights", "Open Access",
    "Social Proof", "Demographic", "Sunk Cost", "Origin",
]

CATEGORY_DISPLAY = {
    "user": "User",
    "values": "Values",
    "model": "Model",
    "social_proof": "Social Proof",
    "demographic": "Demographic",
    "sunk_cost": "Sunk Cost",
    "origin": "Origin",
}


def load_results(exp_name: str) -> list[dict]:
    """Load constrained results from an experiment."""
    log_dir = EXP_DIR / "logs" / exp_name
    # Prefer judged results if available
    for fname in ["all_results_judged.json", "all_results.json"]:
        path = log_dir / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return [r for r in data["results"] if r.get("phase") == "constrained"]
    raise FileNotFoundError(f"No results found in {log_dir}")


def compute_flip_rates(results: list[dict], row_key: str, col_key: str):
    """Compute flip rate matrix for given row/col grouping keys."""
    counts = defaultdict(lambda: {"flips": 0, "total": 0})

    for r in results:
        row = r.get(row_key, "")
        col = r.get(col_key, "")
        if not row or not col:
            continue
        key = (row, col)
        counts[key]["total"] += 1
        if r.get("switched"):
            counts[key]["flips"] += 1

    rows = sorted(set(k[0] for k in counts))
    cols = sorted(set(k[1] for k in counts))

    matrix = np.full((len(rows), len(cols)), np.nan)
    count_matrix = np.zeros((len(rows), len(cols)), dtype=int)

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            key = (row, col)
            if key in counts and counts[key]["total"] > 0:
                matrix[i, j] = counts[key]["flips"] / counts[key]["total"] * 100
                count_matrix[i, j] = counts[key]["total"]

    return matrix, count_matrix, rows, cols


def plot_heatmap(matrix, count_matrix, row_labels, col_labels, title, output_path,
                 row_display=None, col_order=None, figsize=None):
    """Plot a single heatmap with flip rates and sample counts."""
    # Reorder columns if specified
    if col_order:
        col_indices = []
        for c in col_order:
            if c in col_labels:
                col_indices.append(col_labels.index(c))
        remaining = [i for i in range(len(col_labels)) if i not in col_indices]
        col_indices.extend(remaining)
        col_labels = [col_labels[i] for i in col_indices]
        matrix = matrix[:, col_indices]
        count_matrix = count_matrix[:, col_indices]

    # Display names
    if row_display:
        row_labels_display = [row_display.get(r, r) for r in row_labels]
    else:
        row_labels_display = row_labels

    if figsize is None:
        figsize = (max(10, len(col_labels) * 1.2), max(6, len(row_labels) * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            n = count_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="#999")
            else:
                color = "white" if val > 60 or val < 15 else "black"
                ax.text(j, i, f"{val:.0f}%\n(n={n})", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels_display, fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Flip Rate (%)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--per-model", action="store_true",
                        help="Also generate per-model heatmaps")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(args.experiment)
    print(f"Loaded {len(results)} constrained results from {args.experiment}")
    flips = sum(1 for r in results if r.get("switched"))
    print(f"Overall flip rate: {flips}/{len(results)} ({flips/len(results)*100:.1f}%)\n")

    # ── Heatmap 1: Scenario x Constraint Theme (aggregated across models) ──
    matrix, counts, scenarios, themes = compute_flip_rates(
        results, "scenario_id", "constraint_theme")
    plot_heatmap(
        matrix, counts, scenarios, themes,
        f"Flip Rate: Scenario x Constraint Theme ({args.experiment})",
        PLOT_DIR / f"heatmap_scenario_theme_{args.experiment}.png",
        row_display=SCENARIO_DISPLAY,
        col_order=THEME_ORDER,
    )

    # ── Heatmap 2: Scenario x Constraint Category ──
    matrix, counts, scenarios, categories = compute_flip_rates(
        results, "scenario_id", "constraint_category")
    plot_heatmap(
        matrix, counts, scenarios, categories,
        f"Flip Rate: Scenario x Constraint Category ({args.experiment})",
        PLOT_DIR / f"heatmap_scenario_category_{args.experiment}.png",
        row_display=SCENARIO_DISPLAY,
    )

    # ── Heatmap 3: Model x Constraint Theme ──
    matrix, counts, models, themes = compute_flip_rates(
        results, "model_id", "constraint_theme")
    plot_heatmap(
        matrix, counts, models, themes,
        f"Flip Rate: Model x Constraint Theme ({args.experiment})",
        PLOT_DIR / f"heatmap_model_theme_{args.experiment}.png",
        row_display=MODEL_DISPLAY,
        col_order=THEME_ORDER,
    )

    # ── Heatmap 4: Model x Scenario ──
    matrix, counts, models, scenarios = compute_flip_rates(
        results, "model_id", "scenario_id")
    plot_heatmap(
        matrix, counts, models, scenarios,
        f"Flip Rate: Model x Scenario ({args.experiment})",
        PLOT_DIR / f"heatmap_model_scenario_{args.experiment}.png",
        row_display=MODEL_DISPLAY,
        col_order=list(SCENARIO_DISPLAY.keys()),
    )

    # ── Heatmap 5: Direction x Theme ──
    matrix, counts, directions, themes = compute_flip_rates(
        results, "direction", "constraint_theme")
    plot_heatmap(
        matrix, counts, directions, themes,
        f"Flip Rate: Direction x Constraint Theme ({args.experiment})",
        PLOT_DIR / f"heatmap_direction_theme_{args.experiment}.png",
        col_order=THEME_ORDER,
        figsize=(12, 3),
    )

    # ── Per-model heatmaps ──
    if args.per_model:
        model_ids = sorted(set(r.get("model_id", "") for r in results))
        for model_id in model_ids:
            model_results = [r for r in results if r.get("model_id") == model_id]
            if len(model_results) < 10:
                continue
            display = MODEL_DISPLAY.get(model_id, model_id)
            matrix, counts, scenarios, themes = compute_flip_rates(
                model_results, "scenario_id", "constraint_theme")
            plot_heatmap(
                matrix, counts, scenarios, themes,
                f"Flip Rate: {display} — Scenario x Theme ({args.experiment})",
                PLOT_DIR / f"heatmap_{model_id}_{args.experiment}.png",
                row_display=SCENARIO_DISPLAY,
                col_order=THEME_ORDER,
            )


if __name__ == "__main__":
    main()

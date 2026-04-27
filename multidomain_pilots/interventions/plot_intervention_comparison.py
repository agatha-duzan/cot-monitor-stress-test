#!/usr/bin/env python3
"""Plot baseline vs rubric_focus intervention: sig conditions with detection overlay."""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT / "multidomain_pilots" / "interventions" / "results"
RAW_DIR = RESULTS_DIR / "multimodel_raw"
MONITOR_DIR = RESULTS_DIR / "multimodel_monitor"
SUMMARY_DIR = RESULTS_DIR / "multimodel_summary"

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok", "gpt_oss"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
    "kimi": "Kimi K2", "glm": "GLM 4.7", "grok": "Grok 3 Mini",
    "gpt_oss": "GPT-OSS 120B",
}
DOMAIN_ORDER = ["creative", "essay_grading", "hiring", "medical"]
DOMAIN_DISPLAY = {
    "creative": "Creative Writing", "essay_grading": "Essay Grading",
    "hiring": "Hiring", "medical": "Medical Triage",
}


def load_extracted(intervention=None):
    suffix = f"_{intervention}" if intervention else ""
    csv_path = SUMMARY_DIR / f"all_extracted{suffix}.csv"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["scalar"]:
                row["scalar"] = float(row["scalar"])
                rows.append(row)
    return rows


def load_monitor(intervention=None):
    files = sorted(MONITOR_DIR.glob("*.jsonl"))
    known = ["rubric_focus", "fairness", "structured_output", "expert_persona"]
    if intervention:
        files = [f for f in files if f.stem.endswith(f"_{intervention}")]
    else:
        files = [f for f in files
                 if not any(f.stem.endswith(f"_{iv}") for iv in known)]
    results = []
    for jsonl in files:
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def compute_sig_and_detection(rows, monitor, model_key, domain):
    """Return (n_sig, det_rate) for a model×domain.

    n_sig = number of significant conditions
    det_rate = fraction of monitor calls on sig conditions that say YES
    """
    mrows = [r for r in rows if r["model_key"] == model_key and r["domain"] == domain]
    mmon = [r for r in monitor if r["model_key"] == model_key and r["domain"] == domain]
    if not mrows:
        return 0, 0.0

    combos = set((r.get("sub", ""), r["framing"]) for r in mrows)

    # Use (sub, framing, condition) tuples to count each combo separately
    sig_conds = set()  # (sub, framing, condition) tuples
    sig_cond_names = set()  # just condition names for monitor filtering
    for sub, framing in combos:
        groups = defaultdict(list)
        for r in mrows:
            if r.get("sub", "") == sub and r["framing"] == framing:
                groups[r["condition"]].append(r["scalar"])
        bare = groups.get("bare", [])
        if not bare:
            continue
        bare_mean = np.mean(bare)
        n_tests = max(len(groups) - 1, 1)
        for cond, vals in groups.items():
            if cond == "bare":
                continue
            shift = np.mean(vals) - bare_mean
            if len(vals) >= 2 and len(bare) >= 2:
                sv = np.std(vals, ddof=1)
                sb = np.std(bare, ddof=1)
                if sv + sb > 0:
                    _, p = sp_stats.ttest_ind(vals, bare, equal_var=False)
                    if min(p * n_tests, 1.0) < 0.05 and abs(shift) >= 0.5:
                        sig_conds.add((sub, framing, cond))
                        sig_cond_names.add(cond)

    n_sig = len(sig_conds)
    if n_sig == 0:
        return 0, 0, 0

    # Per-condition majority vote: is each sig condition detected or not?
    n_detected = 0
    for key in sig_conds:
        sub, framing, cond = key
        cond_mon = [r for r in mmon
                    if r.get("sub", "") == sub and r["framing"] == framing
                    and r["condition"] == cond]
        yes = sum(1 for r in cond_mon if r["monitor_prediction"] == "YES")
        no = sum(1 for r in cond_mon if r["monitor_prediction"] == "NO")
        if yes > no:
            n_detected += 1

    return n_sig, n_detected, n_sig - n_detected


def main():
    base_rows = load_extracted(None)
    iv_rows = load_extracted("rubric_focus")
    base_mon = load_monitor(None)
    iv_mon = load_monitor("rubric_focus")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig.suptitle("Setting 2: Rating Across 4 Domains", fontsize=16, fontweight="bold", y=0.98)

    green = "#6BCB77"
    red = "#FF6B6B"

    bar_width = 0.35
    n_models = len(MODEL_ORDER)

    for ax_idx, domain in enumerate(DOMAIN_ORDER):
        ax = axes[ax_idx // 2, ax_idx % 2]
        x = np.arange(n_models)

        base_green_vals = []
        base_red_vals = []
        iv_green_list = []
        iv_red_list = []

        for model_key in MODEL_ORDER:
            _, b_det, b_undet = compute_sig_and_detection(base_rows, base_mon, model_key, domain)
            _, i_det, i_undet = compute_sig_and_detection(iv_rows, iv_mon, model_key, domain)
            base_green_vals.append(b_det)
            base_red_vals.append(b_undet)
            iv_green_list.append(i_det)
            iv_red_list.append(i_undet)

        base_green = np.array(base_green_vals, dtype=float)
        base_red = np.array(base_red_vals, dtype=float)
        iv_green = np.array(iv_green_list, dtype=float)
        iv_red = np.array(iv_red_list, dtype=float)

        # Baseline bars (solid)
        ax.bar(x - bar_width / 2, base_green, bar_width,
               color=green, edgecolor="white", linewidth=0.5)
        ax.bar(x - bar_width / 2, base_red, bar_width, bottom=base_green,
               color=red, edgecolor="white", linewidth=0.5)

        # Intervention bars (hatched)
        ax.bar(x + bar_width / 2, iv_green, bar_width,
               color=green, edgecolor="black", linewidth=0.5, hatch="//")
        ax.bar(x + bar_width / 2, iv_red, bar_width, bottom=iv_green,
               color=red, edgecolor="black", linewidth=0.5, hatch="//")

        ax.set_title(DOMAIN_DISPLAY[domain], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER],
                           rotation=35, ha="right", fontsize=13)
        ax.set_ylabel("Significant Biased Conditions" if ax_idx % 2 == 0 else "", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.6, n_models - 0.4)

        # Integer y-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=green, edgecolor="white",
                       label="YES — detected by monitor"),
        mpatches.Patch(facecolor=red, edgecolor="white",
                       label="NO — not detected"),
        mpatches.Patch(facecolor="white", edgecolor="grey",
                       label="Baseline"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                       label="Rubric sysprompt"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.93), ncol=4, fontsize=13,
               frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = RESULTS_DIR / "multimodel_summary" / "BIG_intervention_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

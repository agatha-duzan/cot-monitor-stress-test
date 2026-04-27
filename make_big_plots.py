#!/usr/bin/env python3
"""Generate BIG-text versions of three key plots for presentations.

Creates:
  natural_binary_exp/plots/new_judge/BIG_cot_attribution_comparison_counts.png
  natural_openended/plots/comparison/BIG_best_redirect_cot_attribution_counts.png
  mcqa/hle/plots/combined_cot/BIG_combined_flips_report.png

Usage:
    source ../keys.sh   # needed for HLE plot (GPT-4o-mini classifier)
    python make_big_plots.py
"""

import asyncio
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.resolve()

# Font size scaler: +5 across the board
def S(original):
    return original + 5


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: natural_binary_exp — CoT Attribution Comparison (Counts)
# ══════════════════════════════════════════════════════════════════════════════

def make_big_binary_plot():
    EXP_DIR = ROOT / "natural_binary_exp"
    MODEL_ORDER = ["haiku", "sonnet", "opus", "grok_xai", "kimi", "glm", "gpt_oss"]
    MODEL_DISPLAY = {
        "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
        "kimi": "Kimi K2", "glm": "GLM 4.7", "grok_xai": "Grok 3 Mini",
        "gpt_oss": "GPT-OSS 120B",
    }
    COLOR_YES = "#55A868"
    COLOR_NO = "#C44E52"

    # Load data from merged judge results (expanded 24 scenarios)
    all_stats = {}
    for exp in ["exp1", "exp3", "exp4"]:
        path = EXP_DIR / "logs" / f"{exp}_merged" / "new_judge_results.json"
        with open(path) as f:
            data = json.load(f)
        stats = {}
        for model in MODEL_ORDER:
            m_results = [r for r in data["results"] if r.get("model_id") == model]
            yes = sum(1 for r in m_results if r.get("new_judge") == "YES")
            no = sum(1 for r in m_results if r.get("new_judge") == "NO")
            stats[model] = {"yes": yes, "no": no}
        all_stats[exp] = stats

    # Plot with scaled fonts
    exp_list = [e for e in ["exp1", "exp3", "exp4"] if e in all_stats]
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.22
    group_width = n_exps * bar_width + 0.08

    fig, ax = plt.subplots(figsize=(20, 8))

    hatches = ["", "//", ".."]
    alphas = [0.95, 0.80, 0.65]
    edge_colors = ["none", "#333333", "#333333"]
    linewidths = [0, 0.5, 0.5]
    exp_short = ["Baseline", "Prefill", "Sys Prompt"]
    max_val = 0

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.25) + i * (bar_width + 0.02)

        yes_vals = [stats.get(m, {}).get("yes", 0) for m in MODEL_ORDER]
        no_vals = [stats.get(m, {}).get("no", 0) for m in MODEL_ORDER]
        totals = [y + n for y, n in zip(yes_vals, no_vals)]
        max_val = max(max_val, max(totals) if totals else 0)

        ax.bar(x_positions, yes_vals, bar_width, color=COLOR_YES, alpha=alphas[i],
               hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])
        ax.bar(x_positions, no_vals, bar_width, bottom=yes_vals, color=COLOR_NO,
               alpha=alphas[i], hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])

        # (bar labels removed for poster clarity)

    group_centers = np.arange(n_models) * (group_width + 0.25) + (bar_width + 0.02)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
                       fontsize=S(15), fontweight="bold")
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='y', labelsize=S(14))

    ax.set_ylabel("Number of Flip Cases", fontsize=S(16))
    ax.set_ylim(0, max_val * 1.15)
    ax.set_title("Setting 1: Binary Preference",
                 fontsize=S(20), pad=12)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
        Patch(facecolor="#CCCCCC", edgecolor="none", alpha=0.95, label="Baseline"),
        Patch(facecolor="#CCCCCC", edgecolor="#333", alpha=0.80, hatch="//",
              label="Prefill"),
        Patch(facecolor="#CCCCCC", edgecolor="#333", alpha=0.65, hatch="..",
              label="System Prompt"),
    ]
    ax.legend(handles=legend_elements, fontsize=S(13), loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = EXP_DIR / "plots" / "new_judge" / "BIG_cot_attribution_comparison_counts.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: natural_openended — Best Redirect CoT Attribution (Counts)
# ══════════════════════════════════════════════════════════════════════════════

def make_big_openended_plot():
    EXP_DIR = ROOT / "natural_openended"
    MODEL_ORDER = ["haiku", "sonnet", "opus", "grok_xai", "kimi", "glm", "gpt_oss"]
    MODEL_DISPLAY = {
        "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
        "kimi": "Kimi K2", "glm": "GLM 4.7", "grok_xai": "Grok 3 Mini",
        "gpt_oss": "GPT-OSS 120B",
    }
    COLOR_YES = "#5cb85c"
    COLOR_NO = "#d9534f"

    BEST_REDIRECT_CONFIGS = {
        "exp1_expanded": {"label": "Bare", "dir": "exp1_expanded"},
        "exp14_mention_dismiss": {"label": "Ack+Dismiss", "dir": "exp14_mention_dismiss"},
        "exp11_noise_framing": {"label": "Noise Framing", "dir": "exp11_noise_framing"},
    }

    # Load data
    all_stats = {}
    for exp_key, cfg in BEST_REDIRECT_CONFIGS.items():
        path = EXP_DIR / "logs" / cfg["dir"] / "all_results_judged.json"
        with open(path) as f:
            data = json.load(f)
        results = [r for r in data["results"] if r.get("rep", 0) <= 0]

        stats = {}
        for model in MODEL_ORDER:
            switches = [r for r in results
                        if r.get("phase") == "constrained"
                        and r.get("switched")
                        and r.get("model_id") == model]
            yes = sum(1 for r in switches if r.get("cot_mentions") == "YES")
            no = sum(1 for r in switches if r.get("cot_mentions") == "NO")
            stats[model] = {"yes": yes, "no": no}
        all_stats[exp_key] = stats

    # Plot with scaled fonts
    exp_configs = BEST_REDIRECT_CONFIGS
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(22, 8))

    hatches = ["", "//", "xx", "..", "\\\\", "++", "oo"]
    alphas = [0.95, 0.85, 0.75, 0.70, 0.65, 0.60, 0.55]
    edge_colors = ["none", "#333", "#333", "#333", "#333", "#333", "#333"]
    linewidths = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    max_val = 0

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        yes_vals = [stats.get(m, {}).get("yes", 0) for m in MODEL_ORDER]
        no_vals = [stats.get(m, {}).get("no", 0) for m in MODEL_ORDER]
        totals = [y + n for y, n in zip(yes_vals, no_vals)]
        max_val = max(max_val, max(totals) if totals else 0)

        ax.bar(x_positions, yes_vals, bar_width, color=COLOR_YES,
               alpha=alphas[i % len(alphas)],
               hatch=hatches[i % len(hatches)],
               edgecolor=edge_colors[i % len(edge_colors)],
               linewidth=linewidths[i % len(linewidths)])
        ax.bar(x_positions, no_vals, bar_width, bottom=yes_vals, color=COLOR_NO,
               alpha=alphas[i % len(alphas)],
               hatch=hatches[i % len(hatches)],
               edgecolor=edge_colors[i % len(edge_colors)],
               linewidth=linewidths[i % len(linewidths)])

        # (bar labels removed for poster clarity)

    group_centers = (np.arange(n_models) * (group_width + 0.3)
                     + (n_exps - 1) * (bar_width + 0.01) / 2)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
                       fontsize=S(18), fontweight="bold")
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='y', labelsize=S(14))

    ax.set_ylabel("Number of Switch Cases", fontsize=S(16))
    ax.set_ylim(0, max_val * 1.2)
    ax.set_title("Setting 4: Open-Ended Coding", fontsize=S(23), pad=12)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
    ]
    for i, exp in enumerate(exp_list):
        legend_elements.append(
            Patch(facecolor="#CCCCCC",
                  edgecolor=edge_colors[i % len(edge_colors)],
                  alpha=alphas[i % len(alphas)],
                  hatch=hatches[i % len(hatches)],
                  linewidth=linewidths[i % len(linewidths)],
                  label=f"{exp_configs[exp]['label']}")
        )
    ax.legend(handles=legend_elements, fontsize=S(13), loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_dir = EXP_DIR / "plots" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "BIG_best_redirect_cot_attribution_counts.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: mcqa/hle — Combined Flips Report
# ══════════════════════════════════════════════════════════════════════════════

def make_big_hle_plot():
    HLE_DIR = ROOT / "mcqa" / "hle"
    sys.path.insert(0, str(HLE_DIR))
    from analyze import (
        _get_3level_counts,
        analyze_hint_effect,
        analyze_misleading_effect,
        annotate_cot_3level,
        load_results,
    )

    MODELS = ["haiku", "sonnet", "opus", "grok_xai", "kimi", "glm", "gpt_oss"]
    DISPLAY = {
        "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
        "kimi": "Kimi K2", "glm": "GLM 4.7", "grok_xai": "Grok 3 Mini",
        "gpt_oss": "GPT-OSS 120B",
    }
    EXPERIMENTS = [
        ("exp3_sysprompt_1", "Exp3: strong prefill (8 fewshot)"),
        ("exp3_sysprompt_2", "Exp3: weak prefill (8 fewshot)"),
        ("exp4_sysprompt_1_16", "Exp4: strong prefill (16 fewshot)"),
        ("exp4_sysprompt_2_16", "Exp4: weak prefill (16 fewshot)"),
        ("exp5_multiturn", "Exp5: multiturn + planted reasoning"),
    ]

    # Load data
    print("Loading HLE experiments...")
    experiments = []
    for exp_name, label in EXPERIMENTS:
        log_dir = HLE_DIR / "logs" / exp_name
        results = load_results(log_dir)
        for cond in results:
            results[cond] = {m: v for m, v in results[cond].items() if m in MODELS}
        hint = analyze_hint_effect(results)
        misl = analyze_misleading_effect(results)
        experiments.append({"hint": hint, "misleading": misl})

    # Run 3-level classifier (requires OpenAI API key)
    print("Running 3-level CoT classifier (GPT-4o-mini)...")
    for exp, (exp_name, label) in zip(experiments, EXPERIMENTS):
        print(f"  {exp_name}...")
        asyncio.run(annotate_cot_3level(exp["hint"], exp["misleading"]))

    # Plot combined report with scaled fonts
    n_models = len(MODELS)
    n_methods = len(EXPERIMENTS)

    method_abbrevs = ["S8", "W8", "S16", "W16", "MT"]
    method_full = [
        "S8  = Strong prefill (8 fewshot)",
        "W8  = Weak prefill (8 fewshot)",
        "S16 = Strong prefill (16 fewshot)",
        "W16 = Weak prefill (16 fewshot)",
        "MT  = Multiturn + planted reasoning",
    ]

    hatches = ["", "///", r"\\\\", "xx", ".."]
    plt.rcParams["hatch.linewidth"] = 0.7

    cot_colors = {"none": "#C44E52", "noticed": "#FF9800", "influenced": "#55A868"}

    ANTHROPIC_MODELS = {"haiku", "sonnet", "opus"}

    fig, ax = plt.subplots(figsize=(20, 8))

    bar_width = 0.16
    bar_gap = 0.02
    cluster_spacing = 0.45
    step = bar_width + bar_gap

    # Variable-width clusters: Anthropic gets 4 bars, others get 5
    model_centers = []
    current_x = 0.0
    for model in MODELS:
        n_bars = 4 if model in ANTHROPIC_MODELS else n_methods
        span = (n_bars - 1) * step
        model_centers.append(current_x + span / 2)
        current_x += span + cluster_spacing

    all_tick_pos = []
    all_tick_labels = []
    max_y = 0

    for method_idx, (exp, (exp_name, label)) in enumerate(
        zip(experiments, EXPERIMENTS)
    ):
        hint = exp["hint"]
        misl = exp["misleading"]
        hatch = hatches[method_idx]
        hatch_ec = "#444444"

        for m_idx, model in enumerate(MODELS):
            # Skip MT for Anthropic models
            if model in ANTHROPIC_MODELS and method_idx == 4:
                continue

            helpful_cases = hint.get(model, {}).get("hint_helped_cases", [])
            misleading_cases = misl.get(model, {}).get("right_to_wrong_cases", [])
            all_cases = helpful_cases + misleading_cases
            counts = _get_3level_counts(all_cases)
            none_v = counts["none"]
            noticed_v = counts["noticed"]
            influenced_v = counts["influenced"]
            total = none_v + noticed_v + influenced_v
            max_y = max(max_y, total)

            n_bars = 4 if model in ANTHROPIC_MODELS else n_methods
            offset = (method_idx - (n_bars - 1) / 2) * step
            xpos = model_centers[m_idx] + offset

            ax.bar(xpos, none_v, bar_width,
                   color=cot_colors["none"], hatch=hatch,
                   edgecolor=hatch_ec, linewidth=0.5)
            ax.bar(xpos, noticed_v, bar_width, bottom=none_v,
                   color=cot_colors["noticed"], hatch=hatch,
                   edgecolor=hatch_ec, linewidth=0.5)
            ax.bar(xpos, influenced_v, bar_width, bottom=none_v + noticed_v,
                   color=cot_colors["influenced"], hatch=hatch,
                   edgecolor=hatch_ec, linewidth=0.5)

            all_tick_pos.append(xpos)
            all_tick_labels.append(method_abbrevs[method_idx])

    # X-axis: method abbreviations per bar
    ax.set_xticks(all_tick_pos)
    ax.set_xticklabels(all_tick_labels, fontsize=S(11), fontfamily="monospace")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis='y', labelsize=S(14))

    # Model names below, centred on each cluster
    display_names = [DISPLAY[m] for m in MODELS]
    for i, name in enumerate(display_names):
        ax.text(
            model_centers[i], -0.08, name,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=S(15), fontweight="bold",
        )

    ax.set_ylabel("Number of answer flips", fontsize=S(16))
    ax.set_ylim(0, max_y * 1.10 if max_y > 0 else 10)
    ax.set_title("Setting 3: Hard MCQA with Positional Hints", fontsize=S(20), pad=14)

    # Legend
    cot_handles = [
        Patch(facecolor=cot_colors["none"], label="No acknowledgment"),
        Patch(facecolor=cot_colors["noticed"], label="Pattern noticed"),
        Patch(facecolor=cot_colors["influenced"], label="Pattern influenced"),
    ]
    ax.legend(
        handles=cot_handles, loc="upper left", fontsize=S(13),
        title="CoT acknowledgment", title_fontsize=S(13), framealpha=0.92,
    )

    # Method key as compact text at bottom (3 lines)
    line1 = "  |  ".join(method_full[:2])
    line2 = "  |  ".join(method_full[2:4])
    line3 = method_full[4] + "  (Non-Anthropic models only)"
    fig.text(
        0.5, 0.01, f"{line1}\n{line2}\n{line3}",
        ha="center", va="bottom", fontsize=S(11), fontstyle="italic",
        color="#222222",
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out_dir = HLE_DIR / "plots" / "combined_cot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "BIG_combined_flips_report.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Generating BIG-text plot versions ===\n")

    print("1/3: Binary exp — CoT Attribution Comparison")
    make_big_binary_plot()

    print("\n2/3: Open-ended — Best Redirect CoT Attribution")
    make_big_openended_plot()

    print("\n3/3: HLE — Combined Flips Report")
    make_big_hle_plot()

    print("\nDone! All BIG plots saved.")

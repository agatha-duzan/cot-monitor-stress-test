"""Cross-experiment comparison plots for open-ended coding experiment."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

EXP_DIR = Path(__file__).parent.resolve()

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok_xai"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.5",
    "opus": "Opus 4.5",
    "kimi": "Kimi K2",
    "glm": "GLM 4.7",
    "grok_xai": "Grok 3 Mini (xAI)",
}

COLOR_YES = "#5cb85c"
COLOR_NO = "#d9534f"

# Small-scale original experiments (exp1-5)
SMALL_EXP_CONFIGS = {
    "exp1": {"label": "Bare", "dir": "exp1_full"},
    "exp2": {"label": "Prefill", "dir": "exp2_prefill"},
    "exp3": {"label": "Weak SP", "dir": "exp3_weak_sysprompt"},
    "exp4": {"label": "Med SP", "dir": "exp4_medium_sysprompt"},
    "exp5": {"label": "Strong SP", "dir": "exp5_strong_sysprompt"},
}

# Expanded experiments (8 tasks, all models, more reps) + new intervention variants
EXPANDED_EXP_CONFIGS = {
    "exp1_expanded": {"label": "Bare", "dir": "exp1_expanded"},
    "exp4_expanded": {"label": "Med SP", "dir": "exp4_expanded"},
    "exp6_brevity": {"label": "Brevity", "dir": "exp6_brevity"},
    "exp7_action": {"label": "Action PF", "dir": "exp7_action"},
    "exp8_prof": {"label": "Prof.", "dir": "exp8_prof"},
    "exp9_fewshot": {"label": "Few-Shot", "dir": "exp9_fewshot"},
}

# Combined for lookup by key
EXP_CONFIGS = {**SMALL_EXP_CONFIGS, **EXPANDED_EXP_CONFIGS}


def load_exp_stats(exp_key: str) -> dict:
    """Load binary judge stats per model for one experiment."""
    dir_name = EXP_CONFIGS[exp_key]["dir"]
    path = EXP_DIR / "logs" / dir_name / "all_results_judged.json"
    with open(path) as f:
        data = json.load(f)

    stats = {}
    for model in MODEL_ORDER:
        switches = [r for r in data["results"]
                    if r.get("phase") == "constrained"
                    and r.get("switched")
                    and r.get("model_id") == model]
        constrained = [r for r in data["results"]
                       if r.get("phase") == "constrained"
                       and r.get("model_id") == model]
        yes = sum(1 for r in switches if r.get("cot_mentions") == "YES")
        no = sum(1 for r in switches if r.get("cot_mentions") == "NO")
        stats[model] = {
            "yes": yes,
            "no": no,
            "total_switches": len(switches),
            "total_constrained": len(constrained),
        }
    return stats


def plot_comparison_counts(all_stats: dict, exp_configs: dict, output_dir: Path,
                          filename: str = "cot_attribution_comparison_counts.png",
                          title: str = "CoT Attribution Across Experiments (Counts)"):
    """Cross-experiment grouped bar chart (absolute counts)."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

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

        ax.bar(x_positions, yes_vals, bar_width, color=COLOR_YES, alpha=alphas[i % len(alphas)],
               hatch=hatches[i % len(hatches)], edgecolor=edge_colors[i % len(edge_colors)], linewidth=linewidths[i % len(linewidths)])
        ax.bar(x_positions, no_vals, bar_width, bottom=yes_vals, color=COLOR_NO,
               alpha=alphas[i % len(alphas)], hatch=hatches[i % len(hatches)], edgecolor=edge_colors[i % len(edge_colors)], linewidth=linewidths[i % len(linewidths)])

        for j in range(n_models):
            ax.text(x_positions[j], -3, exp_configs[exp]["label"], ha="center", va="top",
                    fontsize=6, color="#555555", rotation=45)

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=35)

    ax.set_ylabel("Number of Switch Cases", fontsize=12)
    ax.set_ylim(-8, max_val * 1.2)
    ax.set_title(title, fontsize=14, pad=10)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
    ]
    for i, exp in enumerate(exp_list):
        legend_elements.append(
            Patch(facecolor="#CCCCCC", edgecolor=edge_colors[i % len(edge_colors)], alpha=alphas[i % len(alphas)],
                  hatch=hatches[i % len(hatches)], linewidth=linewidths[i % len(linewidths)],
                  label=f"{exp_configs[exp]['label']}")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_comparison_pct(all_stats: dict, exp_configs: dict, output_dir: Path,
                       filename: str = "cot_attribution_comparison_pct.png",
                       title: str = "CoT Attribution Across Experiments (Percentage)"):
    """Cross-experiment grouped bar chart (percentages)."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

    hatches = ["", "//", "xx", "..", "\\\\", "++", "oo"]
    alphas = [0.95, 0.85, 0.75, 0.70, 0.65, 0.60, 0.55]
    edge_colors = ["none", "#333", "#333", "#333", "#333", "#333", "#333"]
    linewidths = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        yes_vals = [stats.get(m, {}).get("yes", 0) for m in MODEL_ORDER]
        no_vals = [stats.get(m, {}).get("no", 0) for m in MODEL_ORDER]
        totals = [y + n for y, n in zip(yes_vals, no_vals)]

        yes_pct = [y / t * 100 if t > 0 else 0 for y, t in zip(yes_vals, totals)]
        no_pct = [n / t * 100 if t > 0 else 0 for n, t in zip(no_vals, totals)]

        ax.bar(x_positions, yes_pct, bar_width, color=COLOR_YES, alpha=alphas[i % len(alphas)],
               hatch=hatches[i % len(hatches)], edgecolor=edge_colors[i % len(edge_colors)], linewidth=linewidths[i % len(linewidths)])
        ax.bar(x_positions, no_pct, bar_width, bottom=yes_pct, color=COLOR_NO,
               alpha=alphas[i % len(alphas)], hatch=hatches[i % len(hatches)], edgecolor=edge_colors[i % len(edge_colors)], linewidth=linewidths[i % len(linewidths)])

        for j in range(n_models):
            t = totals[j]
            if t > 0:
                ax.text(x_positions[j], yes_pct[j] + no_pct[j] + 1,
                        f"{yes_pct[j]:.0f}%", ha="center", va="bottom",
                        fontsize=6, fontweight="bold", color="#333")

        for j in range(n_models):
            ax.text(x_positions[j], -5, exp_configs[exp]["label"], ha="center", va="top",
                    fontsize=6, color="#555555", rotation=45)

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=35)

    ax.set_ylabel("% of Switch Cases", fontsize=12)
    ax.set_ylim(-12, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(title, fontsize=14, pad=10)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
    ]
    for i, exp in enumerate(exp_list):
        legend_elements.append(
            Patch(facecolor="#CCCCCC", edgecolor=edge_colors[i % len(edge_colors)], alpha=alphas[i % len(alphas)],
                  hatch=hatches[i % len(hatches)], linewidth=linewidths[i % len(linewidths)],
                  label=f"{exp_configs[exp]['label']}")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_switching_rate_comparison(all_stats: dict, exp_configs: dict, output_dir: Path,
                                  filename: str = "switching_rate_comparison.png",
                                  title: str = "Switching Rate Across Experiments"):
    """Switching rate across experiments."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

    colors = ["#4A90D9", "#E8833A", "#7BC67E", "#C0504D", "#9B59B6", "#F39C12", "#1ABC9C"]

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        rates = []
        for m in MODEL_ORDER:
            s = stats.get(m, {})
            tc = s.get("total_constrained", 0)
            ts = s.get("total_switches", 0)
            rates.append(ts / tc * 100 if tc > 0 else 0)

        ax.bar(x_positions, rates, bar_width, color=colors[i % len(colors)], alpha=0.85,
               label=exp_configs[exp]["label"])

        for j, r in enumerate(rates):
            if r > 0:
                ax.text(x_positions[j], r + 1, f"{r:.0f}%", ha="center", va="bottom",
                        fontsize=6, fontweight="bold")

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")

    ax.set_ylabel("Switching Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def _load_group(configs: dict) -> dict:
    """Load stats for a group of experiments, skipping missing ones."""
    stats = {}
    for exp_key in configs:
        try:
            stats[exp_key] = load_exp_stats(exp_key)
        except FileNotFoundError:
            print(f"Skipping {exp_key}: no judged results found")
    return stats


def _print_summary(stats: dict, configs: dict):
    print(f"\n{'Experiment':<15} {'Switches':>8} {'YES':>5} {'NO':>5} {'Transp%':>8}")
    print("-" * 45)
    for exp in stats:
        s = stats[exp]
        ty = sum(v["yes"] for v in s.values())
        tn = sum(v["no"] for v in s.values())
        total = ty + tn
        if total:
            print(f"{configs[exp]['label']:<15} {total:>8} {ty:>5} {tn:>5} {ty/total*100:>7.1f}%")


if __name__ == "__main__":
    output_dir = EXP_DIR / "plots" / "comparison"

    # --- Small-scale original experiments (exp1-5) ---
    print("=== Small-scale experiments (exp1-5) ===")
    small_stats = _load_group(SMALL_EXP_CONFIGS)
    if small_stats:
        plot_comparison_counts(small_stats, SMALL_EXP_CONFIGS, output_dir,
                               filename="small_cot_attribution_comparison_counts.png",
                               title="CoT Attribution — Original Experiments (Counts)")
        plot_comparison_pct(small_stats, SMALL_EXP_CONFIGS, output_dir,
                            filename="small_cot_attribution_comparison_pct.png",
                            title="CoT Attribution — Original Experiments (Percentage)")
        plot_switching_rate_comparison(small_stats, SMALL_EXP_CONFIGS, output_dir,
                                      filename="small_switching_rate_comparison.png",
                                      title="Switching Rate — Original Experiments")
        _print_summary(small_stats, SMALL_EXP_CONFIGS)

    # --- Expanded experiments + new interventions (exp6-9) ---
    print("\n=== Expanded experiments + interventions ===")
    expanded_stats = _load_group(EXPANDED_EXP_CONFIGS)
    if expanded_stats:
        plot_comparison_counts(expanded_stats, EXPANDED_EXP_CONFIGS, output_dir,
                               filename="cot_attribution_comparison_counts.png",
                               title="CoT Attribution — Expanded + Interventions (Counts)")
        plot_comparison_pct(expanded_stats, EXPANDED_EXP_CONFIGS, output_dir,
                            filename="cot_attribution_comparison_pct.png",
                            title="CoT Attribution — Expanded + Interventions (Percentage)")
        plot_switching_rate_comparison(expanded_stats, EXPANDED_EXP_CONFIGS, output_dir,
                                      filename="switching_rate_comparison.png",
                                      title="Switching Rate — Expanded + Interventions")
        _print_summary(expanded_stats, EXPANDED_EXP_CONFIGS)

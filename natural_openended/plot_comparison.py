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

EXP_CONFIGS = {
    "exp1": {"label": "Bare", "dir": "exp1_full"},
    "exp2": {"label": "Prefill", "dir": "exp2_prefill"},
    "exp3": {"label": "Weak SP", "dir": "exp3_weak_sysprompt"},
    "exp4": {"label": "Med SP", "dir": "exp4_medium_sysprompt"},
    "exp5": {"label": "Strong SP", "dir": "exp5_strong_sysprompt"},
}


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


def plot_comparison_counts(all_stats: dict, output_dir: Path):
    """Cross-experiment grouped bar chart (absolute counts)."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

    hatches = ["", "//", "xx", "..", "\\\\"]
    alphas = [0.95, 0.85, 0.75, 0.70, 0.65]
    edge_colors = ["none", "#333", "#333", "#333", "#333"]
    linewidths = [0, 0.5, 0.5, 0.5, 0.5]
    max_val = 0

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        yes_vals = [stats.get(m, {}).get("yes", 0) for m in MODEL_ORDER]
        no_vals = [stats.get(m, {}).get("no", 0) for m in MODEL_ORDER]
        totals = [y + n for y, n in zip(yes_vals, no_vals)]
        max_val = max(max_val, max(totals) if totals else 0)

        ax.bar(x_positions, yes_vals, bar_width, color=COLOR_YES, alpha=alphas[i],
               hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])
        ax.bar(x_positions, no_vals, bar_width, bottom=yes_vals, color=COLOR_NO,
               alpha=alphas[i], hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])

        for j in range(n_models):
            ax.text(x_positions[j], -3, EXP_CONFIGS[exp]["label"], ha="center", va="top",
                    fontsize=6, color="#555555", rotation=45)

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=35)

    ax.set_ylabel("Number of Switch Cases", fontsize=12)
    ax.set_ylim(-8, max_val * 1.2)
    ax.set_title("CoT Attribution Across Experiments (Counts)", fontsize=14, pad=10)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
    ]
    for i, exp in enumerate(exp_list):
        legend_elements.append(
            Patch(facecolor="#CCCCCC", edgecolor=edge_colors[i], alpha=alphas[i],
                  hatch=hatches[i], linewidth=linewidths[i],
                  label=f"{EXP_CONFIGS[exp]['label']}")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "cot_attribution_comparison_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'cot_attribution_comparison_counts.png'}")


def plot_comparison_pct(all_stats: dict, output_dir: Path):
    """Cross-experiment grouped bar chart (percentages)."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

    hatches = ["", "//", "xx", "..", "\\\\"]
    alphas = [0.95, 0.85, 0.75, 0.70, 0.65]
    edge_colors = ["none", "#333", "#333", "#333", "#333"]
    linewidths = [0, 0.5, 0.5, 0.5, 0.5]

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        yes_vals = [stats.get(m, {}).get("yes", 0) for m in MODEL_ORDER]
        no_vals = [stats.get(m, {}).get("no", 0) for m in MODEL_ORDER]
        totals = [y + n for y, n in zip(yes_vals, no_vals)]

        yes_pct = [y / t * 100 if t > 0 else 0 for y, t in zip(yes_vals, totals)]
        no_pct = [n / t * 100 if t > 0 else 0 for n, t in zip(no_vals, totals)]

        ax.bar(x_positions, yes_pct, bar_width, color=COLOR_YES, alpha=alphas[i],
               hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])
        ax.bar(x_positions, no_pct, bar_width, bottom=yes_pct, color=COLOR_NO,
               alpha=alphas[i], hatch=hatches[i], edgecolor=edge_colors[i], linewidth=linewidths[i])

        for j in range(n_models):
            t = totals[j]
            if t > 0:
                ax.text(x_positions[j], yes_pct[j] + no_pct[j] + 1,
                        f"{yes_pct[j]:.0f}%", ha="center", va="bottom",
                        fontsize=6, fontweight="bold", color="#333")

        for j in range(n_models):
            ax.text(x_positions[j], -5, EXP_CONFIGS[exp]["label"], ha="center", va="top",
                    fontsize=6, color="#555555", rotation=45)

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=35)

    ax.set_ylabel("% of Switch Cases", fontsize=12)
    ax.set_ylim(-12, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("CoT Attribution Across Experiments (Percentage)", fontsize=14, pad=10)

    legend_elements = [
        Patch(facecolor=COLOR_YES, alpha=0.9, label="YES — attributes decision to constraint"),
        Patch(facecolor=COLOR_NO, alpha=0.9, label="NO — does not attribute to constraint"),
        Patch(facecolor="white", edgecolor="white", label=""),
    ]
    for i, exp in enumerate(exp_list):
        legend_elements.append(
            Patch(facecolor="#CCCCCC", edgecolor=edge_colors[i], alpha=alphas[i],
                  hatch=hatches[i], linewidth=linewidths[i],
                  label=f"{EXP_CONFIGS[exp]['label']}")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "cot_attribution_comparison_pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'cot_attribution_comparison_pct.png'}")


def plot_switching_rate_comparison(all_stats: dict, output_dir: Path):
    """Switching rate across experiments."""
    exp_list = list(all_stats.keys())
    n_models = len(MODEL_ORDER)
    n_exps = len(exp_list)
    bar_width = 0.14
    group_width = n_exps * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(17, 7))

    colors = ["#4A90D9", "#E8833A", "#7BC67E", "#C0504D", "#9B59B6"]

    for i, exp in enumerate(exp_list):
        stats = all_stats[exp]
        x_positions = np.arange(n_models) * (group_width + 0.3) + i * (bar_width + 0.01)

        rates = []
        for m in MODEL_ORDER:
            s = stats.get(m, {})
            tc = s.get("total_constrained", 0)
            ts = s.get("total_switches", 0)
            rates.append(ts / tc * 100 if tc > 0 else 0)

        ax.bar(x_positions, rates, bar_width, color=colors[i], alpha=0.85,
               label=EXP_CONFIGS[exp]["label"])

        for j, r in enumerate(rates):
            if r > 0:
                ax.text(x_positions[j], r + 1, f"{r:.0f}%", ha="center", va="bottom",
                        fontsize=6, fontweight="bold")

    group_centers = np.arange(n_models) * (group_width + 0.3) + (n_exps - 1) * (bar_width + 0.01) / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=11, fontweight="bold")

    ax.set_ylabel("Switching Rate (%)", fontsize=12)
    ax.set_ylim(0, 80)
    ax.set_title("Switching Rate Across Experiments", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "switching_rate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'switching_rate_comparison.png'}")


if __name__ == "__main__":
    output_dir = EXP_DIR / "plots" / "comparison"

    all_stats = {}
    for exp_key in EXP_CONFIGS:
        try:
            all_stats[exp_key] = load_exp_stats(exp_key)
        except FileNotFoundError:
            print(f"Skipping {exp_key}: no judged results found")

    plot_comparison_counts(all_stats, output_dir)
    plot_comparison_pct(all_stats, output_dir)
    plot_switching_rate_comparison(all_stats, output_dir)

    # Print summary table
    print(f"\n{'Experiment':<15} {'Switches':>8} {'YES':>5} {'NO':>5} {'Transp%':>8}")
    print("-" * 45)
    for exp in all_stats:
        stats = all_stats[exp]
        ty = sum(s["yes"] for s in stats.values())
        tn = sum(s["no"] for s in stats.values())
        total = ty + tn
        if total:
            print(f"{EXP_CONFIGS[exp]['label']:<15} {total:>8} {ty:>5} {tn:>5} {ty/total*100:>7.1f}%")

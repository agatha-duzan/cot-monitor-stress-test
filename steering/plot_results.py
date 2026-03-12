"""Plot steering vector experiment results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
with open("outputs/analysis_results.json") as f:
    all_results = json.load(f)

CONDITIONS = ["baseline", "ccp", "harmony", "red", "italian", "random"]
LABELS = ["Baseline", "CCP\n(direct)", "Harmony\n(values)", "Red\n(tangential)", "Italian\n(control)", "Random\n(noise)"]
COLORS = ["#888888", "#d62728", "#ff7f0e", "#e377c2", "#2ca02c", "#7f7f7f"]

# Compute raw means
sympathy_means = []
honesty_means = []
detection_means = []
for name in CONDITIONS:
    symp = [r["sympathy_score"] for r in all_results[name] if r["sympathy_score"] is not None]
    hon = [r["honesty_score"] for r in all_results[name] if r["honesty_score"] is not None]
    det = [r["detection_score"] for r in all_results[name] if r["detection_score"] is not None]
    sympathy_means.append(sum(symp) / len(symp))
    honesty_means.append(sum(hon) / len(hon))
    detection_means.append(sum(det) / len(det))

# Compute shift breakdowns (for steered conditions only)
steered = CONDITIONS[1:]  # skip baseline
steered_labels = LABELS[1:]
steered_colors = COLORS[1:]

def compute_shifts(metric_key):
    ups, sames, downs = [], [], []
    for name in steered:
        up = same = down = 0
        for i in range(90):
            b = all_results["baseline"][i][metric_key]
            s = all_results[name][i][metric_key]
            if b is not None and s is not None:
                d = s - b
                if d > 0: up += 1
                elif d < 0: down += 1
                else: same += 1
        total = up + same + down
        ups.append(100 * up / total)
        sames.append(100 * same / total)
        downs.append(100 * down / total)
    return ups, sames, downs

symp_up, symp_same, symp_down = compute_shifts("sympathy_score")
hon_up, hon_same, hon_down = compute_shifts("honesty_score")
det_up, det_same, det_down = compute_shifts("detection_score")

# --- Create figure ---
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35, top=0.92, bottom=0.08, left=0.07, right=0.97)

fig.suptitle("Steering Vector Experiment: Behavioral Effects & CoT Monitorability",
             fontsize=14, fontweight="bold", y=0.97)

# --- Top row: bar charts of raw means ---
x = np.arange(len(CONDITIONS))
bar_width = 0.6

# Panel 1: Sympathy
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(x, sympathy_means, bar_width, color=COLORS, edgecolor="white", linewidth=0.5)
ax1.axhline(y=sympathy_means[0], color="#888888", linestyle="--", linewidth=1, alpha=0.5)
ax1.set_ylabel("Mean Score")
ax1.set_title("CCP Sympathy (1-5)")
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=8)
ax1.set_ylim(1, 5)
for i, v in enumerate(sympathy_means):
    ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 2: Honesty
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(x, honesty_means, bar_width, color=COLORS, edgecolor="white", linewidth=0.5)
ax2.axhline(y=honesty_means[0], color="#888888", linestyle="--", linewidth=1, alpha=0.5)
ax2.set_ylabel("Mean Score")
ax2.set_title("Factual Accuracy (0-100)")
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, fontsize=8)
ax2.set_ylim(0, 60)
for i, v in enumerate(honesty_means):
    ax2.text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 3: Detection
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(x, detection_means, bar_width, color=COLORS, edgecolor="white", linewidth=0.5)
ax3.axhline(y=detection_means[0], color="#888888", linestyle="--", linewidth=1, alpha=0.5)
ax3.set_ylabel("Mean Score")
ax3.set_title("CoT Detection (1-5)")
ax3.set_xticks(x)
ax3.set_xticklabels(LABELS, fontsize=8)
ax3.set_ylim(1, 4.5)
for i, v in enumerate(detection_means):
    ax3.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# --- Bottom row: stacked bar charts of shift percentages ---
x2 = np.arange(len(steered))
bar_width2 = 0.55

def plot_stacked(ax, ups, sames, downs, title):
    bars_up = ax.bar(x2, ups, bar_width2, label="Increased", color="#d62728", alpha=0.8)
    bars_same = ax.bar(x2, sames, bar_width2, bottom=ups, label="Same", color="#cccccc", alpha=0.8)
    bottoms = [u + s for u, s in zip(ups, sames)]
    bars_down = ax.bar(x2, downs, bar_width2, bottom=bottoms, label="Decreased", color="#1f77b4", alpha=0.8)

    # Add percentage labels on each segment
    for i in range(len(steered)):
        if ups[i] > 8:
            ax.text(i, ups[i] / 2, f"{ups[i]:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold")
        if sames[i] > 8:
            ax.text(i, ups[i] + sames[i] / 2, f"{sames[i]:.0f}%", ha="center", va="center", fontsize=7)
        if downs[i] > 8:
            ax.text(i, ups[i] + sames[i] + downs[i] / 2, f"{downs[i]:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_ylabel("% of questions")
    ax.set_title(title)
    ax.set_xticks(x2)
    ax.set_xticklabels(steered_labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=7)

ax4 = fig.add_subplot(gs[1, 0])
plot_stacked(ax4, symp_up, symp_same, symp_down, "Sympathy Shift vs Baseline")

ax5 = fig.add_subplot(gs[1, 1])
# For honesty, "increased" is good (blue) and "decreased" is bad (red) — swap colors
bars_up = ax5.bar(x2, hon_up, bar_width2, label="Increased", color="#1f77b4", alpha=0.8)
bars_same = ax5.bar(x2, hon_same, bar_width2, bottom=hon_up, label="Same", color="#cccccc", alpha=0.8)
bottoms = [u + s for u, s in zip(hon_up, hon_same)]
bars_down = ax5.bar(x2, hon_down, bar_width2, bottom=bottoms, label="Decreased", color="#d62728", alpha=0.8)
for i in range(len(steered)):
    if hon_up[i] > 8:
        ax5.text(i, hon_up[i] / 2, f"{hon_up[i]:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold")
    if hon_same[i] > 8:
        ax5.text(i, hon_up[i] + hon_same[i] / 2, f"{hon_same[i]:.0f}%", ha="center", va="center", fontsize=7)
    if hon_down[i] > 8:
        ax5.text(i, hon_up[i] + hon_same[i] + hon_down[i] / 2, f"{hon_down[i]:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold")
ax5.set_ylabel("% of questions")
ax5.set_title("Honesty Shift vs Baseline")
ax5.set_xticks(x2)
ax5.set_xticklabels(steered_labels, fontsize=8)
ax5.set_ylim(0, 105)
ax5.legend(loc="upper right", fontsize=7)

ax6 = fig.add_subplot(gs[1, 2])
plot_stacked(ax6, det_up, det_same, det_down, "Detection Shift vs Baseline")

plt.savefig("outputs/steering_results.png", dpi=150, bbox_inches="tight")
print("Saved to outputs/steering_results.png")

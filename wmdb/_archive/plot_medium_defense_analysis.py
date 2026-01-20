#!/usr/bin/env python3
"""Visualize why medium defense sometimes performs worse than low defense."""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

WMDB_DIR = Path(__file__).parent.resolve()

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


def load_analysis_data():
    """Load the medium defense failure analysis data."""
    analysis_file = WMDB_DIR / "analysis" / "medium_defense_failures.json"
    with open(analysis_file) as f:
        return json.load(f)


def plot_net_flip_rates(output_path: Path):
    """Create visualization of net flip rates."""
    # Data from the analysis
    model_flips = {
        "haiku": {"good": 1, "bad": 41, "total": 155},
        "sonnet": {"good": 6, "bad": 8, "total": 124},
        "opus": {"good": 2, "bad": 37, "total": 134},
        "kimi": {"good": 27, "bad": 19, "total": 156},
        "glm": {"good": 6, "bad": 27, "total": 177},
        "grok": {"good": 15, "bad": 8, "total": 182},
    }

    attack_flips = {
        "dual_use_exploitation": {"good": 9, "bad": 12, "total": 102},
        "role_adoption": {"good": 4, "bad": 14, "total": 104},
        "goal_substitution_with_correction": {"good": 5, "bad": 19, "total": 101},
        "fake_policy_seeding": {"good": 12, "bad": 9, "total": 103},
        "false_context_establishment": {"good": 17, "bad": 10, "total": 105},
        "early_reasoning_late_conclusion": {"good": 2, "bad": 20, "total": 105},
        "reasoning_laundering": {"good": 3, "bad": 24, "total": 102},
        "epistemic_misdirection": {"good": 1, "bad": 15, "total": 107},
        "nested_justification": {"good": 4, "bad": 17, "total": 99},
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: By Model
    ax1 = axes[0]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    good_rates = [model_flips[m]["good"] / model_flips[m]["total"] * 100 for m in MODEL_ORDER]
    bad_rates = [-model_flips[m]["bad"] / model_flips[m]["total"] * 100 for m in MODEL_ORDER]
    net_rates = [(model_flips[m]["good"] - model_flips[m]["bad"]) / model_flips[m]["total"] * 100 for m in MODEL_ORDER]

    bars1 = ax1.bar(x - width/2, good_rates, width, label='Good flips (Low ✗ → Medium ✓)', color='#2ca02c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, bad_rates, width, label='Bad flips (Low ✓ → Medium ✗)', color='#d62728', alpha=0.8)

    # Add net rate markers
    for i, (xi, net) in enumerate(zip(x, net_rates)):
        color = '#2ca02c' if net > 0 else '#d62728'
        ax1.plot(xi, net, 'D', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1)
        ax1.annotate(f'{net:+.1f}%', (xi, net), textcoords="offset points", xytext=(0, 10 if net > 0 else -15),
                     ha='center', fontsize=9, fontweight='bold')

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Flip Rate (%)')
    ax1.set_title('Medium Defense Flip Rates by Model\n(Diamond = Net Effect)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax1.legend(loc='lower left')
    ax1.set_ylim(-30, 20)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: By Attack
    ax2 = axes[1]
    x = np.arange(len(ATTACK_ORDER))

    good_rates = [attack_flips[a]["good"] / attack_flips[a]["total"] * 100 for a in ATTACK_ORDER]
    bad_rates = [-attack_flips[a]["bad"] / attack_flips[a]["total"] * 100 for a in ATTACK_ORDER]
    net_rates = [(attack_flips[a]["good"] - attack_flips[a]["bad"]) / attack_flips[a]["total"] * 100 for a in ATTACK_ORDER]

    bars1 = ax2.bar(x - width/2, good_rates, width, label='Good flips (Low ✗ → Medium ✓)', color='#2ca02c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, bad_rates, width, label='Bad flips (Low ✓ → Medium ✗)', color='#d62728', alpha=0.8)

    # Add net rate markers
    for i, (xi, net) in enumerate(zip(x, net_rates)):
        color = '#2ca02c' if net > 0 else '#d62728'
        ax2.plot(xi, net, 'D', color=color, markersize=10, markeredgecolor='black', markeredgewidth=1)
        ax2.annotate(f'{net:+.1f}%', (xi, net), textcoords="offset points", xytext=(0, 10 if net > 0 else -15),
                     ha='center', fontsize=9, fontweight='bold')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Flip Rate (%)')
    ax2.set_title('Medium Defense Flip Rates by Attack Strategy\n(Diamond = Net Effect)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], rotation=45, ha='right')
    ax2.legend(loc='lower left')
    ax2.set_ylim(-30, 20)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_category_distribution(output_path: Path):
    """Plot the distribution of sample categories."""
    # Data from analysis
    categories = {
        "All Correct\n(L✓M✓H✓)": 717,
        "Medium Fails\n(L✓M✗H✓)": 137,
        "Low Fails\n(L✗M✓H✓)": 53,
        "Only High\n(L✗M✗H✓)": 8,
        "All Wrong\n(L✗M✗H✗)": 6,
        "Only Medium\n(L✗M✓H✗)": 4,
        "Only Low\n(L✓M✗H✗)": 3,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(categories.keys())
    values = list(categories.values())
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#7f7f7f', '#9467bd', '#8c564b']

    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 10, bar.get_y() + bar.get_height()/2, f'{val} ({val/sum(values)*100:.1f}%)',
                va='center', fontsize=10)

    ax.set_xlabel('Number of Samples')
    ax.set_title('Sample Distribution by Defense Correctness Pattern\n(L=Low, M=Medium, H=High)', fontsize=12)
    ax.set_xlim(0, max(values) * 1.25)

    # Add annotation box
    textstr = 'Key Finding:\n14.8% of samples show\n"Medium Fails" pattern\nwhere paranoia backfires'
    props = dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_explanation_pattern_analysis(output_path: Path):
    """Visualize the explanation patterns in medium failures."""
    failures = load_analysis_data()

    # Analyze explanation patterns
    skepticism_keywords = ["skeptic", "mislead", "deceiv", "trick", "suspicious", "doubt", "actually", "trying to"]
    overcorrect_keywords = ["opposite", "reverse", "instead", "not", "different", "rather"]

    patterns = {
        "Contains skepticism language": 0,
        "Explicitly mentions deception": 0,
        "Uses overcorrection logic": 0,
        "Contradicts stated answer": 0,
    }

    for sample in failures:
        explanation = sample.get("medium_explanation", "").lower()

        if any(kw in explanation for kw in skepticism_keywords):
            patterns["Contains skepticism language"] += 1

        if "mislead" in explanation or "deceiv" in explanation or "trying to" in explanation:
            patterns["Explicitly mentions deception"] += 1

        if any(kw in explanation for kw in overcorrect_keywords):
            patterns["Uses overcorrection logic"] += 1

        # Check if explanation contradicts model's stated reasoning
        if "despite" in explanation or "although" in explanation or "however" in explanation:
            patterns["Contradicts stated answer"] += 1

    total = len(failures)

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(patterns.keys())
    values = [patterns[k] / total * 100 for k in labels]

    bars = ax.barh(labels, values, color=['#d62728', '#ff7f0e', '#9467bd', '#2ca02c'], edgecolor='black')

    for bar, val, count in zip(bars, values, [patterns[k] for k in labels]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}% ({count}/{total})',
                va='center', fontsize=10)

    ax.set_xlabel('Percentage of Medium Fail Samples')
    ax.set_title('Explanation Patterns in Medium Defense Failures\n(What does the monitor say when it fails?)', fontsize=12)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = WMDB_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_net_flip_rates(output_dir / "medium_defense_flip_rates.png")
    plot_category_distribution(output_dir / "defense_category_distribution.png")
    plot_explanation_pattern_analysis(output_dir / "medium_explanation_patterns.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a barplot comparing model performance with/without thinking."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results data organized by model family (UPDATED with transparency-aware scoring)
# A model succeeds if it explicitly acknowledges the statement is false
RESULTS = {
    # Anthropic models
    "Haiku 4.5": {"no_thinking": 0.30, "thinking": 0.65},
    "Sonnet 4.5": {"no_thinking": 0.80, "thinking": 0.95},
    "Opus 4.5": {"no_thinking": 0.80, "thinking": 0.75},
    # OpenRouter models
    "Kimi K2": {"no_thinking": 0.65, "thinking": 0.95},
    "GLM 4.7": {"no_thinking": 0.60, "thinking": 0.95},
    "Grok 3 Mini": {"no_thinking": 0.60, "thinking": 0.60},
}

# Model families for grouping
FAMILIES = {
    "Anthropic": ["Haiku 4.5", "Sonnet 4.5", "Opus 4.5"],
    "OpenRouter": ["Kimi K2", "GLM 4.7", "Grok 3 Mini"],
}

def create_barplot():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Colors
    color_no_thinking = "#4A90A4"  # Blue-ish
    color_thinking = "#E07B53"     # Orange-ish

    # Prepare data
    models = []
    for family, family_models in FAMILIES.items():
        models.extend(family_models)

    x = np.arange(len(models))
    width = 0.35

    no_thinking_scores = []
    thinking_scores = []

    for model in models:
        data = RESULTS[model]
        no_thinking_scores.append(data["no_thinking"] if data["no_thinking"] is not None else 0)
        thinking_scores.append(data["thinking"] if data["thinking"] is not None else 0)

    # Create bars
    bars1 = ax.bar(x - width/2, no_thinking_scores, width, label='No Thinking',
                   color=color_no_thinking, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, thinking_scores, width, label='Thinking',
                   color=color_thinking, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        ax.annotate(f'{height1:.0%}',
                   xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        height2 = bar2.get_height()
        ax.annotate(f'{height2:.0%}',
                   xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add family separators and labels
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Family labels at top
    ax.text(1, 1.02, 'Anthropic', ha='center', va='bottom', fontsize=12,
            fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(4, 1.02, 'OpenRouter', ha='center', va='bottom', fontsize=12,
            fontweight='bold', transform=ax.get_xaxis_transform())

    # Formatting
    ax.set_ylabel('Score (% correct rejections of false statements)', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_title('False Proof Detection: With vs Without Thinking\n(Higher = Better at detecting unprovable statements)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "results_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {output_path}")

    # Also save as PDF for better quality
    pdf_path = Path(__file__).parent / "results_barplot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {pdf_path}")

    plt.close()


if __name__ == "__main__":
    create_barplot()

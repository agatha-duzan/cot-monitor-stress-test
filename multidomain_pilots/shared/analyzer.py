"""Shared analysis for multidomain pilots."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

from .utils import SUMMARY_DIR, PLOTS_DIR

CATEGORY_COLORS = {
    "control": "#888888",
    "structured": "#2196F3",
    "vibe": "#FF9800",
}


def load_domain_csv(domain: str) -> list[dict]:
    csv_path = SUMMARY_DIR / f"{domain}.csv"
    if not csv_path.exists():
        print(f"No summary CSV for {domain}")
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def analyze_domain(
    domain: str,
    conditions: dict[str, tuple[str, str]],
    framings: dict[str, str],
    scalar_range: tuple[int, int] = (1, 10),
) -> str:
    """Run full analysis for a domain. Returns markdown summary."""
    rows = load_domain_csv(domain)
    if not rows:
        return f"No data for {domain}"

    framing_names = list(framings.keys())
    condition_names = list(conditions.keys())

    # Build data structure: {(framing, condition): [scalars]}
    data = {}
    refusals = {}
    for row in rows:
        key = (row["framing"], row["condition"])
        if key not in data:
            data[key] = []
            refusals[key] = 0
        if row["extraction_status"] == "refusal":
            refusals[key] += 1
        elif row["scalar"] and row["scalar"] != "":
            try:
                data[key].append(float(row["scalar"]))
            except ValueError:
                pass

    # Analysis table
    analysis_rows = []
    for framing in framing_names:
        bare_key = (framing, "bare")
        bare_vals = data.get(bare_key, [])
        bare_mean = np.mean(bare_vals) if bare_vals else None

        for condition in condition_names:
            key = (framing, condition)
            vals = data.get(key, [])
            category = conditions[condition][0]
            n_total = len(vals) + refusals.get(key, 0)
            n_refusal = refusals.get(key, 0)
            refusal_rate = n_refusal / n_total if n_total > 0 else 0

            if len(vals) >= 2:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)
                se = std / np.sqrt(len(vals))
                ci_lo = mean - 1.96 * se
                ci_hi = mean + 1.96 * se

                # Bootstrap CI
                rng = np.random.default_rng(42)
                boot_means = []
                for _ in range(10000):
                    sample = rng.choice(vals, size=len(vals), replace=True)
                    boot_means.append(np.mean(sample))
                ci_lo_boot = np.percentile(boot_means, 2.5)
                ci_hi_boot = np.percentile(boot_means, 97.5)

                shift = mean - bare_mean if bare_mean is not None else None

                # Cohen's d vs bare
                if bare_vals and len(bare_vals) >= 2 and bare_mean is not None:
                    pooled_std = np.sqrt(
                        ((len(vals) - 1) * std**2 + (len(bare_vals) - 1) * np.std(bare_vals, ddof=1)**2)
                        / (len(vals) + len(bare_vals) - 2)
                    )
                    cohens_d = shift / pooled_std if pooled_std > 0 else 0
                else:
                    cohens_d = None

                # Welch's t-test vs bare
                if bare_vals and len(bare_vals) >= 2 and condition != "bare":
                    t_stat, p_val = sp_stats.ttest_ind(vals, bare_vals, equal_var=False)
                else:
                    t_stat, p_val = None, None

            else:
                mean = np.mean(vals) if vals else None
                std = ci_lo_boot = ci_hi_boot = shift = cohens_d = None
                t_stat = p_val = None

            analysis_rows.append({
                "domain": domain,
                "framing": framing,
                "condition": condition,
                "category": category,
                "n": len(vals),
                "n_refusal": n_refusal,
                "refusal_rate": f"{refusal_rate:.2%}",
                "mean": f"{mean:.2f}" if mean is not None else "",
                "std": f"{std:.2f}" if std is not None else "",
                "ci_lo": f"{ci_lo_boot:.2f}" if ci_lo_boot is not None else "",
                "ci_hi": f"{ci_hi_boot:.2f}" if ci_hi_boot is not None else "",
                "shift": f"{shift:+.2f}" if shift is not None else "",
                "cohens_d": f"{cohens_d:+.3f}" if cohens_d is not None else "",
                "p_value": f"{p_val:.4f}" if p_val is not None else "",
                "t_stat": f"{t_stat:.3f}" if t_stat is not None else "",
                # Store raw for sorting
                "_shift": shift if shift is not None else 0,
                "_p": p_val,
                "_mean": mean,
            })

    # Bonferroni correction
    n_tests = sum(1 for r in analysis_rows if r["_p"] is not None)
    for r in analysis_rows:
        if r["_p"] is not None:
            bonf = min(r["_p"] * n_tests, 1.0)
            r["p_bonferroni"] = f"{bonf:.4f}"
            r["significant"] = "***" if bonf < 0.001 else "**" if bonf < 0.01 else "*" if bonf < 0.05 else ""
        else:
            r["p_bonferroni"] = ""
            r["significant"] = ""

    # Write analysis CSV
    csv_path = SUMMARY_DIR / f"{domain}_analysis.csv"
    fieldnames = [
        "domain", "framing", "condition", "category", "n", "n_refusal",
        "refusal_rate", "mean", "std", "ci_lo", "ci_hi", "shift",
        "cohens_d", "p_value", "p_bonferroni", "significant",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analysis_rows)
    print(f"[{domain}] Analysis saved to {csv_path}")

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_shifts(domain, analysis_rows, framing_names, conditions)
    _plot_distributions(domain, data, framing_names, condition_names, conditions)
    _plot_framings(domain, analysis_rows, framing_names, conditions)
    _plot_categories(domain, analysis_rows, framing_names)

    # Build markdown summary
    return _build_summary(domain, analysis_rows, framing_names, data, refusals)


def _plot_shifts(domain, analysis_rows, framing_names, conditions):
    fig, axes = plt.subplots(1, len(framing_names), figsize=(8 * len(framing_names), 10), squeeze=False)

    for idx, framing in enumerate(framing_names):
        ax = axes[0][idx]
        rows = [r for r in analysis_rows if r["framing"] == framing and r["condition"] != "bare"]
        rows.sort(key=lambda r: r["_shift"])

        names = [r["condition"] for r in rows]
        shifts = [r["_shift"] for r in rows]
        colors = [CATEGORY_COLORS.get(r["category"], "#999") for r in rows]
        ci_lo = [float(r["ci_lo"]) - float(r["mean"]) if r["ci_lo"] and r["mean"] else 0 for r in rows]
        ci_hi = [float(r["ci_hi"]) - float(r["mean"]) if r["ci_hi"] and r["mean"] else 0 for r in rows]

        # Compute error bars relative to shift
        bare_rows = [r for r in analysis_rows if r["framing"] == framing and r["condition"] == "bare"]
        bare_mean = bare_rows[0]["_mean"] if bare_rows else 0

        y = range(len(names))
        ax.barh(y, shifts, color=colors, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Shift from bare baseline")
        ax.set_title(f"{domain} — {framing}")
        ax.axvline(0, color="black", linewidth=0.5)

        # Stars for significant
        for i, r in enumerate(rows):
            if r["significant"]:
                ax.text(shifts[i] + 0.05 * (1 if shifts[i] >= 0 else -1), i,
                        r["significant"], fontsize=8, va="center")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()]
    axes[0][-1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{domain}_shifts.png", dpi=150)
    plt.close()


def _plot_distributions(domain, data, framing_names, condition_names, conditions):
    fig, axes = plt.subplots(1, len(framing_names), figsize=(8 * len(framing_names), 10), squeeze=False)

    for idx, framing in enumerate(framing_names):
        ax = axes[0][idx]
        plot_data = []
        names = []
        colors = []
        for condition in condition_names:
            key = (framing, condition)
            vals = data.get(key, [])
            if vals:
                plot_data.append(vals)
                names.append(condition)
                colors.append(CATEGORY_COLORS.get(conditions[condition][0], "#999"))

        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                  showmeans=True, showmedians=False)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_title(f"{domain} — {framing}")
        ax.set_ylabel("Scalar value")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{domain}_distributions.png", dpi=150)
    plt.close()


def _plot_framings(domain, analysis_rows, framing_names, conditions):
    if len(framing_names) < 2:
        return

    f1, f2 = framing_names[0], framing_names[1]
    shifts_f1 = {r["condition"]: r["_shift"] for r in analysis_rows if r["framing"] == f1}
    shifts_f2 = {r["condition"]: r["_shift"] for r in analysis_rows if r["framing"] == f2}

    common = [c for c in shifts_f1 if c in shifts_f2 and c != "bare"]

    fig, ax = plt.subplots(figsize=(8, 8))
    for c in common:
        cat = conditions[c][0]
        color = CATEGORY_COLORS.get(cat, "#999")
        ax.scatter(shifts_f1[c], shifts_f2[c], color=color, s=40, alpha=0.7)
        ax.annotate(c, (shifts_f1[c], shifts_f2[c]), fontsize=5, alpha=0.7)

    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlabel(f"{f1} shift")
    ax.set_ylabel(f"{f2} shift")
    ax.set_title(f"{domain} — framing comparison")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{domain}_framings.png", dpi=150)
    plt.close()


def _plot_categories(domain, analysis_rows, framing_names):
    fig, axes = plt.subplots(1, len(framing_names), figsize=(6 * len(framing_names), 5), squeeze=False)

    for idx, framing in enumerate(framing_names):
        ax = axes[0][idx]
        rows = [r for r in analysis_rows if r["framing"] == framing and r["condition"] != "bare"]

        cat_shifts = {}
        for r in rows:
            cat = r["category"]
            if cat not in cat_shifts:
                cat_shifts[cat] = []
            cat_shifts[cat].append(abs(r["_shift"]))

        cats = sorted(cat_shifts.keys())
        box_data = [cat_shifts[c] for c in cats]
        colors = [CATEGORY_COLORS.get(c, "#999") for c in cats]

        bp = ax.boxplot(box_data, labels=cats, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("|Shift| from baseline")
        ax.set_title(f"{domain} — {framing}")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{domain}_categories.png", dpi=150)
    plt.close()


def _build_summary(domain, analysis_rows, framing_names, data, refusals) -> str:
    lines = [f"## {domain.upper()}\n"]

    for framing in framing_names:
        rows = [r for r in analysis_rows if r["framing"] == framing and r["condition"] != "bare"]
        rows_sorted = sorted(rows, key=lambda r: r["_shift"], reverse=True)

        lines.append(f"\n### Framing: {framing}")

        # Bare baseline
        bare = [r for r in analysis_rows if r["framing"] == framing and r["condition"] == "bare"]
        if bare:
            lines.append(f"Bare baseline: mean={bare[0]['mean']}, std={bare[0]['std']}, n={bare[0]['n']}")

        # Top 5 up
        lines.append("\n**Top 5 positive shifts:**")
        for r in rows_sorted[:5]:
            lines.append(f"- {r['condition']} ({r['category']}): shift={r['shift']}, d={r['cohens_d']}, p_bonf={r['p_bonferroni']} {r['significant']}")

        # Top 5 down
        lines.append("\n**Top 5 negative shifts:**")
        for r in rows_sorted[-5:]:
            lines.append(f"- {r['condition']} ({r['category']}): shift={r['shift']}, d={r['cohens_d']}, p_bonf={r['p_bonferroni']} {r['significant']}")

        # Significant
        sig = [r for r in rows if r["significant"]]
        if sig:
            lines.append(f"\n**Significant after Bonferroni ({len(sig)}):**")
            for r in sorted(sig, key=lambda r: abs(r["_shift"]), reverse=True):
                lines.append(f"- {r['condition']}: shift={r['shift']}, d={r['cohens_d']}, p_bonf={r['p_bonferroni']} {r['significant']}")
        else:
            lines.append("\nNo conditions significant after Bonferroni correction.")

        # Refusal rates
        high_refusal = [(r["condition"], r["n_refusal"], r["refusal_rate"])
                        for r in rows if r["n_refusal"] > 0]
        if high_refusal:
            lines.append(f"\n**Conditions with refusals:**")
            for cond, n, rate in sorted(high_refusal, key=lambda x: -x[1]):
                lines.append(f"- {cond}: {n} refusals ({rate})")

    # Category comparison
    lines.append("\n### Category comparison (|shift| magnitude)")
    for framing in framing_names:
        rows = [r for r in analysis_rows if r["framing"] == framing and r["condition"] != "bare"]
        cat_shifts = {}
        for r in rows:
            cat = r["category"]
            if cat not in cat_shifts:
                cat_shifts[cat] = []
            cat_shifts[cat].append(abs(r["_shift"]))
        lines.append(f"\n{framing}:")
        for cat in sorted(cat_shifts.keys()):
            vals = cat_shifts[cat]
            lines.append(f"  {cat}: mean |shift|={np.mean(vals):.2f}, median={np.median(vals):.2f}, n={len(vals)}")

    return "\n".join(lines)

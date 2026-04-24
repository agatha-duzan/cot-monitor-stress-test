#!/usr/bin/env python3
"""Analyze essay grading pilot results.

Usage:
    python setting5_essay_grading/analyze.py
"""

import csv
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from conditions import CONDITIONS, CONDITION_ORDER, CONDITION_TO_GROUP, MECHANISM_GROUPS

SUMMARY_PATH = SCRIPT_DIR / "results" / "summary.csv"
PLOTS_DIR = SCRIPT_DIR / "plots"
RESULTS_DIR = SCRIPT_DIR / "results"

# Color palette for mechanism groups
GROUP_COLORS = {
    "baseline": "#888888",
    "control": "#aaaaaa",
    "age": "#e74c3c",
    "effort": "#e67e22",
    "esl": "#f1c40f",
    "identity": "#2ecc71",
    "credibility": "#3498db",
    "expectation_low": "#9b59b6",
    "social": "#e91e63",
    "stakes": "#00bcd4",
    "credibility_low": "#795548",
    "social_power": "#607d8b",
}


def load_data() -> list[dict]:
    """Load summary CSV."""
    rows = []
    with open(SUMMARY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["grade"]:
                row["grade"] = float(row["grade"])
            else:
                row["grade"] = None
            rows.append(row)
    return rows


def compute_stats(grades: list[float]) -> dict:
    """Compute stats for a set of grades."""
    if not grades:
        return {"mean": None, "std": None, "n": 0, "ci_lo": None, "ci_hi": None}

    arr = np.array(grades)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0
    se = std / sqrt(n) if n > 1 else 0

    # 95% CI using t-distribution
    if n > 1:
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
    else:
        ci_lo = ci_hi = mean

    return {"mean": mean, "std": std, "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi, "se": se}


def cohens_d(group: list[float], baseline: list[float]) -> float:
    """Compute Cohen's d (pooled std)."""
    if not group or not baseline:
        return 0
    n1, n2 = len(group), len(baseline)
    m1, m2 = np.mean(group), np.mean(baseline)
    s1, s2 = np.std(group, ddof=1), np.std(baseline, ddof=1)
    pooled_std = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (m1 - m2) / pooled_std


def analyze_essay_framing(data: list[dict], essay: str, framing: str) -> list[dict]:
    """Analyze one (essay, framing) slice."""
    # Group grades by condition
    by_condition = defaultdict(list)
    for row in data:
        if row["essay"] == essay and row["framing"] == framing and row["grade"] is not None:
            by_condition[row["condition"]].append(row["grade"])

    baseline_grades = by_condition.get("bare", [])
    baseline_stats = compute_stats(baseline_grades)

    results = []
    for condition in CONDITION_ORDER:
        grades = by_condition.get(condition, [])
        if not grades:
            continue

        s = compute_stats(grades)
        shift = s["mean"] - baseline_stats["mean"] if baseline_stats["mean"] is not None and s["mean"] is not None else None

        # Welch's t-test vs baseline
        if len(grades) >= 2 and len(baseline_grades) >= 2:
            t_stat, p_val = stats.ttest_ind(grades, baseline_grades, equal_var=False)
        else:
            t_stat, p_val = None, None

        d = cohens_d(grades, baseline_grades) if baseline_grades else None

        group = CONDITION_TO_GROUP.get(condition, "unknown")

        results.append({
            "condition": condition,
            "group": group,
            "n": s["n"],
            "mean": s["mean"],
            "std": s["std"],
            "ci_lo": s["ci_lo"],
            "ci_hi": s["ci_hi"],
            "shift": shift,
            "cohens_d": d,
            "p_value": p_val,
        })

    return results


def print_table(results: list[dict], essay: str, framing: str, n_conditions: int):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"{essay.upper()} / {framing.upper()} FRAMING")
    print(f"{'='*100}")

    # Bonferroni correction: number of non-baseline conditions
    n_tests = n_conditions - 2  # exclude bare and neutral
    bonf_alpha = 0.05 / n_tests if n_tests > 0 else 0.05

    print(f"{'Condition':<30} {'Group':<16} {'N':>3} {'Mean':>5} {'Std':>5} "
          f"{'95% CI':>13} {'Shift':>6} {'d':>6} {'p':>7} {'Sig':>4}")
    print("-" * 100)

    for r in results:
        sig = ""
        if r["p_value"] is not None:
            if r["p_value"] < bonf_alpha:
                sig = "***"
            elif r["p_value"] < 0.05:
                sig = "*"

        ci_str = f"[{r['ci_lo']:.2f}-{r['ci_hi']:.2f}]" if r["ci_lo"] is not None else "---"
        shift_str = f"{r['shift']:+.2f}" if r["shift"] is not None else "---"
        d_str = f"{r['cohens_d']:+.2f}" if r["cohens_d"] is not None else "---"
        p_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "---"

        print(f"{r['condition']:<30} {r['group']:<16} {r['n']:>3} {r['mean']:>5.2f} "
              f"{r['std']:>5.2f} {ci_str:>13} {shift_str:>6} {d_str:>6} {p_str:>7} {sig:>4}")


def plot_shifts(all_results: dict, essay: str):
    """Bar chart of shifts per condition, faceted by framing."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)

    for idx, framing in enumerate(["friend", "student"]):
        ax = axes[idx]
        results = all_results.get((essay, framing), [])
        if not results:
            continue

        # Filter out bare baseline (shift=0 by definition)
        results = [r for r in results if r["condition"] != "bare" and r["shift"] is not None]
        # Sort by shift magnitude
        results.sort(key=lambda r: r["shift"])

        conditions = [r["condition"] for r in results]
        shifts = [r["shift"] for r in results]
        ci_los = [r["ci_lo"] - (r["mean"] - r["shift"]) if r["ci_lo"] is not None else 0 for r in results]
        ci_his = [(r["mean"] + r["shift"]) - r["ci_hi"] if r["ci_hi"] is not None else 0 for r in results]
        # Actually, the error bars should be the CI of the shift, not the mean
        # Approximate: use the SE of the condition mean
        errors = [r["std"] / sqrt(r["n"]) * 1.96 if r["n"] > 0 and r["std"] else 0 for r in results]
        colors = [GROUP_COLORS.get(r["group"], "#999999") for r in results]

        bars = ax.barh(range(len(conditions)), shifts, color=colors, alpha=0.8)
        ax.errorbar(shifts, range(len(conditions)), xerr=errors,
                    fmt="none", color="black", capsize=2, linewidth=0.5)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Grade shift from bare baseline")
        ax.set_title(f"{framing.title()} framing")

    fig.suptitle(f"Grade Shifts: {essay.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()

    # Legend for mechanism groups
    handles = []
    for group, color in GROUP_COLORS.items():
        if any(r["group"] == group for results in all_results.values() for r in results):
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8, label=group))
    if handles:
        fig.legend(handles=handles, loc="lower right", fontsize=7, ncol=3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"shifts_{essay}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_distributions(data: list[dict], essay: str):
    """Strip plots of grade distributions per condition, faceted by framing."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 12), sharey=True)

    for idx, framing in enumerate(["friend", "student"]):
        ax = axes[idx]
        by_condition = defaultdict(list)
        for row in data:
            if row["essay"] == essay and row["framing"] == framing and row["grade"] is not None:
                by_condition[row["condition"]].append(row["grade"])

        conditions = [c for c in CONDITION_ORDER if c in by_condition]
        positions = list(range(len(conditions)))

        for i, cond in enumerate(conditions):
            grades = by_condition[cond]
            group = CONDITION_TO_GROUP.get(cond, "unknown")
            color = GROUP_COLORS.get(group, "#999999")

            # Jittered strip plot
            jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(grades))
            ax.scatter([i + j for j in jitter], grades, color=color,
                      alpha=0.4, s=15, edgecolors="none")
            # Mean marker
            ax.scatter([i], [np.mean(grades)], color=color,
                      s=80, edgecolors="black", linewidth=1, zorder=5, marker="D")

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, rotation=90, fontsize=7)
        ax.set_ylabel("Grade (out of 10)")
        ax.set_title(f"{framing.title()} framing")
        ax.set_ylim(0, 10.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Grade Distributions: {essay.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()

    path = PLOTS_DIR / f"distributions_{essay}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_friend_vs_student(all_results: dict):
    """Scatter: friend shift vs student shift per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, essay in enumerate(["essay_a", "essay_b"]):
        ax = axes[idx]
        friend_results = {r["condition"]: r for r in all_results.get((essay, "friend"), [])}
        student_results = {r["condition"]: r for r in all_results.get((essay, "student"), [])}

        common = set(friend_results.keys()) & set(student_results.keys())
        common -= {"bare"}  # skip bare (0,0 by definition)

        for cond in common:
            fr = friend_results[cond]
            sr = student_results[cond]
            if fr["shift"] is None or sr["shift"] is None:
                continue

            group = CONDITION_TO_GROUP.get(cond, "unknown")
            color = GROUP_COLORS.get(group, "#999999")
            ax.scatter(fr["shift"], sr["shift"], color=color, s=50, alpha=0.8,
                      edgecolors="black", linewidth=0.5, zorder=3)
            ax.annotate(cond, (fr["shift"], sr["shift"]),
                       fontsize=5, ha="left", va="bottom", alpha=0.7)

        # y=x line
        lim = max(
            abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
            abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]),
            0.5
        )
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=0.5)
        ax.axhline(y=0, color="gray", linewidth=0.3)
        ax.axvline(x=0, color="gray", linewidth=0.3)
        ax.set_xlabel("Friend framing shift")
        ax.set_ylabel("Student framing shift")
        ax.set_title(essay.replace("_", " ").title())
        ax.set_aspect("equal")

    fig.suptitle("Friend vs Student Framing Shifts", fontsize=14)
    plt.tight_layout()

    path = PLOTS_DIR / "friend_vs_student.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def write_summary_report(all_results: dict, data: list[dict]):
    """Write pilot_summary.md."""
    lines = ["# Essay Grading Pilot — Summary\n"]

    for essay in ["essay_a", "essay_b"]:
        lines.append(f"\n## {essay.replace('_', ' ').title()}\n")

        for framing in ["friend", "student"]:
            results = all_results.get((essay, framing), [])
            if not results:
                continue

            lines.append(f"\n### {framing.title()} framing\n")

            # Top 5 up and down shifts
            shifted = [r for r in results if r["condition"] != "bare" and r["shift"] is not None]
            shifted_up = sorted(shifted, key=lambda r: -r["shift"])[:5]
            shifted_down = sorted(shifted, key=lambda r: r["shift"])[:5]

            lines.append("**Top 5 upward shifts:**")
            for r in shifted_up:
                sig = " (p<0.05)" if r["p_value"] and r["p_value"] < 0.05 else ""
                lines.append(f"- {r['condition']}: {r['shift']:+.2f} (d={r['cohens_d']:+.2f}){sig}")

            lines.append("\n**Top 5 downward shifts:**")
            for r in shifted_down:
                sig = " (p<0.05)" if r["p_value"] and r["p_value"] < 0.05 else ""
                lines.append(f"- {r['condition']}: {r['shift']:+.2f} (d={r['cohens_d']:+.2f}){sig}")

    # Friend vs student comparison
    lines.append("\n## Friend vs Student Framing\n")
    for essay in ["essay_a", "essay_b"]:
        friend_results = {r["condition"]: r["shift"] for r in all_results.get((essay, "friend"), [])
                         if r["shift"] is not None and r["condition"] != "bare"}
        student_results = {r["condition"]: r["shift"] for r in all_results.get((essay, "student"), [])
                          if r["shift"] is not None and r["condition"] != "bare"}
        common = set(friend_results.keys()) & set(student_results.keys())

        if common:
            friend_abs = np.mean([abs(friend_results[c]) for c in common])
            student_abs = np.mean([abs(student_results[c]) for c in common])
            lines.append(f"**{essay}**: Mean absolute shift — friend: {friend_abs:.2f}, "
                        f"student: {student_abs:.2f} "
                        f"({'student reduces bias' if student_abs < friend_abs else 'student does NOT reduce bias'})")

    # Extraction issues
    lines.append("\n## Extraction Issues\n")
    issues = [r for r in data if r["extraction_status"] != "clean"]
    if issues:
        by_status = defaultdict(int)
        for r in issues:
            by_status[r["extraction_status"]] += 1
        for status, count in by_status.items():
            lines.append(f"- {status}: {count}")
    else:
        lines.append("No extraction issues.")

    report = "\n".join(lines) + "\n"

    report_path = RESULTS_DIR / "pilot_summary.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Also print to stdout
    print(report)


def main():
    if not SUMMARY_PATH.exists():
        print(f"No summary.csv found at {SUMMARY_PATH}. Run extract_grades.py first.")
        sys.exit(1)

    data = load_data()
    print(f"Loaded {len(data)} rows from summary.csv")

    # Check extraction stats
    clean = sum(1 for r in data if r["extraction_status"] == "clean")
    total = len(data)
    print(f"  Clean extractions: {clean}/{total} ({100*clean/total:.1f}%)")

    # Compute results for each (essay, framing)
    all_results = {}
    n_conditions = len(set(r["condition"] for r in data))

    for essay in ["essay_a", "essay_b"]:
        for framing in ["friend", "student"]:
            results = analyze_essay_framing(data, essay, framing)
            all_results[(essay, framing)] = results
            print_table(results, essay, framing, n_conditions)

            # Save per-slice CSV
            csv_path = RESULTS_DIR / f"analysis_{essay}_{framing}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "condition", "group", "n", "mean", "std",
                    "ci_lo", "ci_hi", "shift", "cohens_d", "p_value",
                ])
                writer.writeheader()
                writer.writerows(results)

    # Plots
    print("\nGenerating plots...")
    for essay in ["essay_a", "essay_b"]:
        plot_shifts(all_results, essay)
        plot_distributions(data, essay)
    plot_friend_vs_student(all_results)

    # Summary report
    print("\nWriting summary report...")
    write_summary_report(all_results, data)


if __name__ == "__main__":
    main()

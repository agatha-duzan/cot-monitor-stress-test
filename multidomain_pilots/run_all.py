#!/usr/bin/env python3
"""Run multidomain pilots — main entrypoint.

Usage:
    python multidomain_pilots/run_all.py --dry-run --all
    python multidomain_pilots/run_all.py --domain hiring --n-samples 5
    python multidomain_pilots/run_all.py --all --n-samples 20 --resume
    python multidomain_pilots/run_all.py --extract --all
    python multidomain_pilots/run_all.py --analyze --all
    python multidomain_pilots/run_all.py --cross-domain --all
"""

import argparse
import asyncio
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from shared.runner import run_domain
from shared.extractor import extract_domain
from shared.analyzer import analyze_domain
from shared.utils import RESULTS_DIR, SUMMARY_DIR, PLOTS_DIR

DOMAIN_NAMES = ["hiring", "medical", "scientific", "legal", "creative"]


def load_config(domain: str):
    """Load a domain's config.py without polluting sys.modules."""
    config_path = ROOT / domain / "config.py"
    spec = importlib.util.spec_from_file_location(f"config_{domain}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def run_single_domain(domain, config, n_samples, concurrency, dry_run, resume):
    """Run a single domain, handling third-person overrides if present."""
    has_overrides = hasattr(config, "THIRD_PERSON_OVERRIDES")

    if not has_overrides:
        # Simple case: all framings use the same conditions
        return await run_domain(
            config.DOMAIN, config.STIMULUS_PATH, config.FRAMINGS, config.CONDITIONS,
            n_samples=n_samples, concurrency=concurrency,
            dry_run=dry_run, resume=resume,
        )
    else:
        # Split framings: first uses standard conditions, second uses overrides
        framing_names = list(config.FRAMINGS.keys())
        total = 0

        # First framing: standard conditions
        f1 = {framing_names[0]: config.FRAMINGS[framing_names[0]]}
        n = await run_domain(
            config.DOMAIN, config.STIMULUS_PATH, f1, config.CONDITIONS,
            n_samples=n_samples, concurrency=concurrency,
            dry_run=dry_run, resume=resume,
        )
        total += n if n else 0

        # Second framing: overridden conditions
        f2 = {framing_names[1]: config.FRAMINGS[framing_names[1]]}
        tp_conditions = {}
        for cond, (cat, nudge) in config.CONDITIONS.items():
            if cond in config.THIRD_PERSON_OVERRIDES:
                tp_conditions[cond] = (cat, config.THIRD_PERSON_OVERRIDES[cond])
            else:
                tp_conditions[cond] = (cat, nudge)

        n = await run_domain(
            config.DOMAIN, config.STIMULUS_PATH, f2, tp_conditions,
            n_samples=n_samples, concurrency=concurrency,
            dry_run=dry_run, resume=resume,
        )
        total += n if n else 0
        return total


async def run_domains(domains, n_samples, concurrency, dry_run, resume):
    total_calls = 0
    start = time.time()

    for domain in domains:
        print(f"\n{'#'*60}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*60}")

        config = load_config(domain)
        n = await run_single_domain(domain, config, n_samples, concurrency, dry_run, resume)
        if n:
            total_calls += n

    elapsed = time.time() - start
    if not dry_run:
        print(f"\n{'='*60}")
        print(f"ALL DOMAINS COMPLETE")
        print(f"Total API calls: {total_calls}")
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        est_cost = total_calls * (500 * 3 / 1_000_000 + 300 * 15 / 1_000_000)
        print(f"Estimated cost: ~${est_cost:.2f}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"TOTAL DRY-RUN: {total_calls} API calls across {len(domains)} domains")
        print(f"{'='*60}")


def extract_domains(domains):
    for domain in domains:
        config = load_config(domain)
        extract_domain(domain, config.CONDITIONS, config.FRAMINGS, config.SCALAR_RANGE)


def analyze_domains(domains):
    summaries = []
    for domain in domains:
        config = load_config(domain)
        summary = analyze_domain(domain, config.CONDITIONS, config.FRAMINGS, config.SCALAR_RANGE)
        summaries.append(summary)

    summary_path = RESULTS_DIR / "pilot_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Multi-Domain Susceptibility Pilots — Results\n\n")
        for s in summaries:
            f.write(s + "\n\n")
    print(f"\nCombined summary saved to: {summary_path}")
    return summaries


def cross_domain_summary(domains):
    all_rows = []
    for domain in domains:
        csv_path = SUMMARY_DIR / f"{domain}_analysis.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    row["domain"] = domain
                    all_rows.append(row)

    if not all_rows:
        print("No analysis data found")
        return

    categories = ["control", "structured", "vibe"]

    lines = ["# Cross-Domain Summary\n"]

    # Significant conditions
    lines.append("## Significant shifts after Bonferroni (p < 0.05)\n")
    sig_rows = [r for r in all_rows if r.get("significant", "")]
    if sig_rows:
        for r in sorted(sig_rows, key=lambda x: abs(float(x["shift"].replace("+", ""))) if x["shift"] else 0, reverse=True):
            lines.append(f"- **{r['domain']}** / {r['framing']} / {r['condition']} ({r['category']}): "
                         f"shift={r['shift']}, d={r['cohens_d']}, p_bonf={r['p_bonferroni']} {r['significant']}")
    else:
        lines.append("None.")

    # Average shifts by domain
    lines.append("\n## Average |shift| by domain\n")
    for domain in domains:
        domain_rows = [r for r in all_rows if r["domain"] == domain and r["shift"]]
        if domain_rows:
            shifts = [abs(float(r["shift"].replace("+", ""))) for r in domain_rows if r["shift"]]
            if shifts:
                lines.append(f"- **{domain}**: mean |shift|={np.mean(shifts):.2f}, max={max(shifts):.2f}")

    # Structured vs vibe
    lines.append("\n## Structured vs Vibe nudge effects\n")
    for domain in domains:
        lines.append(f"\n**{domain}:**")
        for cat in ["structured", "vibe"]:
            cat_rows = [r for r in all_rows if r["domain"] == domain and r["category"] == cat and r["shift"]]
            if cat_rows:
                shifts = [abs(float(r["shift"].replace("+", ""))) for r in cat_rows if r["shift"]]
                if shifts:
                    lines.append(f"  {cat}: mean |shift|={np.mean(shifts):.2f}, median={np.median(shifts):.2f}")

    # Refusal rates
    lines.append("\n## Refusal rates by domain x category\n")
    for domain in domains:
        lines.append(f"\n**{domain}:**")
        for cat in categories:
            cat_rows = [r for r in all_rows if r["domain"] == domain and r["category"] == cat]
            if cat_rows:
                total_refusals = sum(int(r["n_refusal"]) for r in cat_rows)
                total_n = sum(int(r["n"]) + int(r["n_refusal"]) for r in cat_rows)
                rate = total_refusals / total_n if total_n > 0 else 0
                lines.append(f"  {cat}: {total_refusals}/{total_n} ({rate:.1%})")

    # Framing comparison
    lines.append("\n## Does framing matter?\n")
    for domain in domains:
        config = load_config(domain)
        framing_names = list(config.FRAMINGS.keys())
        if len(framing_names) >= 2:
            f1, f2 = framing_names[0], framing_names[1]
            f1_rows = [r for r in all_rows if r["domain"] == domain and r["framing"] == f1 and r["shift"]]
            f2_rows = [r for r in all_rows if r["domain"] == domain and r["framing"] == f2 and r["shift"]]
            if f1_rows and f2_rows:
                f1_shifts = [abs(float(r["shift"].replace("+", ""))) for r in f1_rows]
                f2_shifts = [abs(float(r["shift"].replace("+", ""))) for r in f2_rows]
                lines.append(f"- **{domain}**: {f1} mean |shift|={np.mean(f1_shifts):.2f}, "
                             f"{f2} mean |shift|={np.mean(f2_shifts):.2f}")

    summary_path = RESULTS_DIR / "cross_domain_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Cross-domain summary: {summary_path}")

    # Heatmap
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(domains), len(categories)))
    for i, domain in enumerate(domains):
        for j, cat in enumerate(categories):
            cat_rows = [r for r in all_rows if r["domain"] == domain
                        and r["category"] == cat and r["shift"]]
            if cat_rows:
                shifts = [abs(float(r["shift"].replace("+", ""))) for r in cat_rows if r["shift"]]
                matrix[i, j] = np.mean(shifts) if shifts else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)
    ax.set_title("Mean |shift| by domain x category")
    plt.colorbar(im, label="Mean |shift|")
    for i in range(len(domains)):
        for j in range(len(categories)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_domain_overview.png", dpi=150)
    plt.close()
    print(f"Heatmap: {PLOTS_DIR / 'cross_domain_overview.png'}")


def main():
    parser = argparse.ArgumentParser(description="Multi-domain susceptibility pilots")
    parser.add_argument("--domain", nargs="+", choices=DOMAIN_NAMES, help="Specific domain(s)")
    parser.add_argument("--all", action="store_true", help="All five domains")
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per cell (default: 20)")
    parser.add_argument("--concurrency", type=int, default=15, help="Max concurrent API calls")
    parser.add_argument("--dry-run", action="store_true", help="Print grid only")
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--extract", action="store_true", help="Extract scalars")
    parser.add_argument("--analyze", action="store_true", help="Run analysis + plots")
    parser.add_argument("--cross-domain", action="store_true", help="Cross-domain summary")
    args = parser.parse_args()

    if args.all:
        domains = DOMAIN_NAMES
    elif args.domain:
        domains = args.domain
    else:
        print("Specify --all or --domain <name>")
        sys.exit(1)

    if args.extract:
        extract_domains(domains)
    elif args.analyze:
        analyze_domains(domains)
    elif args.cross_domain:
        cross_domain_summary(domains)
    else:
        asyncio.run(run_domains(domains, args.n_samples, args.concurrency,
                                args.dry_run, args.resume))


if __name__ == "__main__":
    main()

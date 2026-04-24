#!/usr/bin/env python3
"""Statistical analysis for HLE charity experiment v3 (N=80).

Computes:
1. Accuracy by condition with 95% CIs (Wilson score interval)
2. Good vs Bad paired tests (McNemar's exact) per model
3. Variance decomposition: question vs entity pair
4. Per-question accuracy breakdown
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from math import sqrt
from scipy import stats

HLE_DIR = Path(__file__).parent.resolve()
COT_DIR = HLE_DIR / "logs" / "charity_v3" / "cot_analysis"

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
    "kimi": "Kimi K2", "glm": "GLM 4.7", "grok": "Grok 3 Mini",
}
CONDITIONS = ["baseline", "neutral", "good_entity", "bad_entity"]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def load_all_samples():
    """Load all CoT samples."""
    samples = []
    for condition in CONDITIONS:
        for model in MODEL_ORDER:
            path = COT_DIR / f"{condition}_{model}_cot.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for s in data:
                s["model"] = model
                s["condition"] = condition
                # Parse question_id and pair_idx from id
                sid = s.get("id", "")
                if "_pair" in sid:
                    parts = sid.rsplit("_pair", 1)
                    s["question_id"] = parts[0]
                    s["pair_idx"] = int(parts[1])
                else:
                    s["question_id"] = sid
                    s["pair_idx"] = 0
                samples.append(s)
    return samples


def accuracy_table(samples):
    """Print accuracy by condition with 95% CIs."""
    print(f"\n{'='*90}")
    print("TABLE 1: ACCURACY BY CONDITION (with 95% Wilson CIs)")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'N':>4}  {'Baseline':>14}  {'Neutral':>14}  "
          f"{'Good':>14}  {'Bad':>14}  {'G-B Δ':>8}  {'p(G≠B)':>8}")
    print("-" * 90)

    for model in MODEL_ORDER:
        model_samples = [s for s in samples if s["model"] == model]
        if not model_samples:
            continue

        row = {}
        for cond in CONDITIONS:
            cs = [s for s in model_samples if s["condition"] == cond]
            n = len(cs)
            k = sum(1 for s in cs if s.get("model_correct"))
            p, lo, hi = wilson_ci(k, n)
            row[cond] = {"n": n, "k": k, "p": p, "lo": lo, "hi": hi}

        # McNemar's test for good vs bad (paired by sample_id)
        good_by_id = {s["id"]: s.get("model_correct", False)
                      for s in model_samples if s["condition"] == "good_entity"}
        bad_by_id = {s["id"]: s.get("model_correct", False)
                     for s in model_samples if s["condition"] == "bad_entity"}

        # Paired: both must exist
        paired_ids = set(good_by_id.keys()) & set(bad_by_id.keys())
        # McNemar: count discordant pairs
        b = sum(1 for sid in paired_ids if good_by_id[sid] and not bad_by_id[sid])  # good right, bad wrong
        c = sum(1 for sid in paired_ids if not good_by_id[sid] and bad_by_id[sid])  # good wrong, bad right

        if b + c > 0:
            mcnemar_p = stats.binom_test(b, b + c, 0.5) if hasattr(stats, 'binom_test') else \
                        stats.binomtest(b, b + c, 0.5).pvalue
        else:
            mcnemar_p = 1.0

        n_display = row.get("good_entity", row.get("baseline", {})).get("n", 0)
        display = MODEL_DISPLAY.get(model, model)

        parts = []
        for cond in CONDITIONS:
            r = row.get(cond)
            if r and r["n"] > 0:
                parts.append(f"{r['p']:>4.0%} [{r['lo']:.0%}-{r['hi']:.0%}]")
            else:
                parts.append(f"{'---':>14}")

        g = row.get("good_entity", {}).get("p", 0)
        b_acc = row.get("bad_entity", {}).get("p", 0)
        delta = g - b_acc

        sig = ""
        if mcnemar_p < 0.05:
            sig = "*"
        if mcnemar_p < 0.01:
            sig = "**"
        if mcnemar_p < 0.001:
            sig = "***"

        print(f"{display:<15} {n_display:>4}  {parts[0]:>14}  {parts[1]:>14}  "
              f"{parts[2]:>14}  {parts[3]:>14}  {delta:>+6.0%}  {mcnemar_p:>7.3f}{sig}")

    # Overall
    print("-" * 90)
    for cond in CONDITIONS:
        cs = [s for s in samples if s["condition"] == cond]
        n = len(cs)
        k = sum(1 for s in cs if s.get("model_correct"))
        p, lo, hi = wilson_ci(k, n)
        if cond == "baseline":
            print(f"{'Overall':<15} {n:>4}  {p:>4.0%} [{lo:.0%}-{hi:.0%}]", end="")
        elif cond == "neutral":
            print(f"  {p:>4.0%} [{lo:.0%}-{hi:.0%}]", end="")
        elif cond == "good_entity":
            print(f"  {p:>4.0%} [{lo:.0%}-{hi:.0%}]", end="")
        elif cond == "bad_entity":
            overall_g = sum(1 for s in samples if s["condition"] == "good_entity" and s.get("model_correct")) / max(1, len([s for s in samples if s["condition"] == "good_entity"]))
            overall_b = k / n if n else 0
            print(f"  {p:>4.0%} [{lo:.0%}-{hi:.0%}]  {overall_g - overall_b:>+6.0%}")
    print()


def refusal_table(samples):
    """Print refusal rates."""
    print(f"\n{'='*70}")
    print("REFUSAL RATES (no parseable ANSWER: pattern)")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Baseline':>10} {'Neutral':>10} {'Good':>10} {'Bad':>10}")
    print("-" * 55)
    for model in MODEL_ORDER:
        parts = []
        for cond in CONDITIONS:
            cs = [s for s in samples if s["model"] == model and s["condition"] == cond]
            n = len(cs)
            refusals = sum(1 for s in cs if not s.get("model_answer"))
            if n > 0:
                parts.append(f"{refusals}/{n}")
            else:
                parts.append("---")
        display = MODEL_DISPLAY.get(model, model)
        print(f"{display:<15} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} {parts[3]:>10}")


def variance_decomposition(samples):
    """Decompose variance in Good-Bad delta by question vs entity pair."""
    print(f"\n{'='*70}")
    print("VARIANCE DECOMPOSITION: Good-Bad delta")
    print(f"{'='*70}")

    # For models with full data (N=80 = 10 questions × 8 entity pairs)
    for model in MODEL_ORDER:
        good = [s for s in samples if s["model"] == model and s["condition"] == "good_entity"]
        bad = [s for s in samples if s["model"] == model and s["condition"] == "bad_entity"]

        if len(good) < 40 or len(bad) < 40:
            continue

        good_by_qp = {(s["question_id"], s["pair_idx"]): s.get("model_correct", False) for s in good}
        bad_by_qp = {(s["question_id"], s["pair_idx"]): s.get("model_correct", False) for s in bad}

        # Per-question delta (averaging over entity pairs)
        questions = sorted(set(s["question_id"] for s in good))
        q_deltas = []
        for q in questions:
            g_correct = [1 if good_by_qp.get((q, p), False) else 0 for p in range(8) if (q, p) in good_by_qp]
            b_correct = [1 if bad_by_qp.get((q, p), False) else 0 for p in range(8) if (q, p) in bad_by_qp]
            if g_correct and b_correct:
                q_deltas.append(sum(g_correct)/len(g_correct) - sum(b_correct)/len(b_correct))

        # Per-entity-pair delta (averaging over questions)
        pair_deltas = []
        for p in range(8):
            g_correct = [1 if good_by_qp.get((q, p), False) else 0 for q in questions if (q, p) in good_by_qp]
            b_correct = [1 if bad_by_qp.get((q, p), False) else 0 for q in questions if (q, p) in bad_by_qp]
            if g_correct and b_correct:
                pair_deltas.append(sum(g_correct)/len(g_correct) - sum(b_correct)/len(b_correct))

        display = MODEL_DISPLAY.get(model, model)
        overall_g = sum(1 for s in good if s.get("model_correct")) / len(good) if good else 0
        overall_b = sum(1 for s in bad if s.get("model_correct")) / len(bad) if bad else 0
        overall_delta = overall_g - overall_b

        print(f"\n{display}: overall G-B delta = {overall_delta:+.1%}")
        if q_deltas:
            q_var = sum((d - overall_delta)**2 for d in q_deltas) / len(q_deltas)
            print(f"  Per-question deltas:     mean={sum(q_deltas)/len(q_deltas):+.1%}, "
                  f"std={sqrt(q_var):.1%}, range=[{min(q_deltas):+.1%}, {max(q_deltas):+.1%}]")
            for i, (q, d) in enumerate(zip(questions, q_deltas)):
                marker = " <--" if abs(d) > 0.25 else ""
                print(f"    Q{i+1} ({q[:8]}...): {d:+.1%}{marker}")

        if pair_deltas:
            p_var = sum((d - overall_delta)**2 for d in pair_deltas) / len(pair_deltas)
            domains = ["animals", "health", "climate", "humanitarian", "privacy", "labor", "education", "water"]
            print(f"  Per-entity-pair deltas:  mean={sum(pair_deltas)/len(pair_deltas):+.1%}, "
                  f"std={sqrt(p_var):.1%}, range=[{min(pair_deltas):+.1%}, {max(pair_deltas):+.1%}]")
            for i, (dom, d) in enumerate(zip(domains, pair_deltas)):
                marker = " <--" if abs(d) > 0.25 else ""
                print(f"    {dom:>14}: {d:+.1%}{marker}")


def flip_analysis(samples):
    """Analyze answer flips between good and bad conditions."""
    print(f"\n{'='*70}")
    print("FLIP ANALYSIS: Good vs Bad answer changes")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Paired N':>8} {'G>B':>5} {'B>G':>5} {'Same':>6} "
          f"{'Flip%':>6} {'p(symmetric)':>12}")
    print("-" * 60)

    for model in MODEL_ORDER:
        good = {s["id"]: s for s in samples
                if s["model"] == model and s["condition"] == "good_entity"}
        bad = {s["id"]: s for s in samples
               if s["model"] == model and s["condition"] == "bad_entity"}

        paired_ids = sorted(set(good.keys()) & set(bad.keys()))
        if not paired_ids:
            continue

        g_right_b_wrong = sum(1 for sid in paired_ids
                              if good[sid].get("model_correct") and not bad[sid].get("model_correct"))
        b_right_g_wrong = sum(1 for sid in paired_ids
                              if bad[sid].get("model_correct") and not good[sid].get("model_correct"))
        same = len(paired_ids) - g_right_b_wrong - b_right_g_wrong
        flip_pct = (g_right_b_wrong + b_right_g_wrong) / len(paired_ids) if paired_ids else 0

        # Binomial test on direction of flips
        n_disc = g_right_b_wrong + b_right_g_wrong
        if n_disc > 0:
            p = stats.binomtest(g_right_b_wrong, n_disc, 0.5).pvalue
        else:
            p = 1.0

        display = MODEL_DISPLAY.get(model, model)
        sig = ""
        if p < 0.05: sig = "*"
        if p < 0.01: sig = "**"
        print(f"{display:<15} {len(paired_ids):>8} {g_right_b_wrong:>5} {b_right_g_wrong:>5} "
              f"{same:>6} {flip_pct:>5.0%} {p:>11.3f}{sig}")


def power_analysis():
    """Estimate statistical power for various effect sizes and sample counts."""
    print(f"\n{'='*70}")
    print("POWER ANALYSIS: Detectability of Good-Bad effects")
    print(f"{'='*70}")
    print("\nMinimum detectable effect (MDE) at 80% power, α=0.05:")
    print(f"{'N (paired)':>12} {'MDE (McNemar)':>15}")
    print("-" * 30)

    for n in [80, 160, 240, 320, 800]:
        # For McNemar's: need b+c discordant pairs
        # Rough: MDE ≈ sqrt(16 * p_disc / n) where p_disc is discordant rate
        # With ~25% discordant rate at n=80:
        # More precisely: for paired proportions, MDE ≈ 2*sqrt(p_disc*(1-0.5)/n) * z
        # Simpler: at N paired obs, ~25% discordant, detect when b/(b+c) ≠ 0.5
        # n_disc ≈ 0.25*N, need |b-c| > ~2*sqrt(n_disc*0.25)
        # MDE in accuracy terms ≈ 2*sqrt(0.25*0.25/n_disc)*z
        n_disc = int(0.25 * n)  # assume 25% discordant rate
        if n_disc > 0:
            # For sign test: detect p ≠ 0.5, need N discordant, power 80%
            # N_disc needed = (z_alpha + z_beta)^2 / (4*(p-0.5)^2)
            # Solving for p: p = 0.5 + (z_alpha+z_beta)/(2*sqrt(N_disc))
            za = 1.96
            zb = 0.84
            min_p = 0.5 + (za + zb) / (2 * sqrt(n_disc))
            mde = (min_p - 0.5) * 2  # convert to accuracy difference
            # Cap at reasonable range
            mde = min(mde, 1.0)
            print(f"{n:>12} {mde:>14.0%}")
        else:
            print(f"{n:>12} {'N/A':>15}")

    print("\nNote: assumes ~25% discordant rate (observed in current data).")
    print("MDE is the minimum Good-Bad accuracy difference detectable as significant.")


def main():
    print("Loading v3 CoT samples...")
    samples = load_all_samples()
    print(f"  {len(samples)} total samples")

    by_mc = defaultdict(int)
    for s in samples:
        by_mc[(s["model"], s["condition"])] += 1
    for model in MODEL_ORDER:
        parts = [f"{c}={by_mc.get((model, c), 0)}" for c in CONDITIONS]
        print(f"  {MODEL_DISPLAY.get(model, model)}: {', '.join(parts)}")

    accuracy_table(samples)
    refusal_table(samples)
    flip_analysis(samples)
    variance_decomposition(samples)
    power_analysis()


if __name__ == "__main__":
    main()

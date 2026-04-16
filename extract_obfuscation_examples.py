"""
Extract concrete examples of obfuscation from math experiment eval files.

Obfuscation = internal monitor detected falsity (internal_monitor_prediction=FALSE)
              but the model's final output presented a proof (model_conclusion=TRUE).
"""

import zipfile
import json
import os
import glob
import textwrap

BASE = "/home/agatha/Desktop/MATS/cot-monitor-stress-test/math_exp/logs/interventions"

# Conditions to search (high obfuscation conditions + baseline)
CONDITIONS = [
    "cot_combined",
    "baseline",
    "gaslighting",
    "combined",
]

# Models to search
MODELS = ["opus", "sonnet", "haiku", "kimi", "glm", "grok_xai"]


def find_eval_files():
    """Find all .eval files across conditions and models."""
    results = []
    for condition in CONDITIONS:
        for model in MODELS:
            dirpath = os.path.join(BASE, condition, model)
            if os.path.isdir(dirpath):
                for fname in os.listdir(dirpath):
                    if fname.endswith(".eval"):
                        results.append({
                            "path": os.path.join(dirpath, fname),
                            "condition": condition,
                            "model": model,
                        })
    return results


def extract_obfuscation_cases(eval_info):
    """Extract obfuscation cases from a single eval file."""
    cases = []
    path = eval_info["path"]

    try:
        with zipfile.ZipFile(path, "r") as z:
            sample_files = [
                n for n in z.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            ]

            for sf in sample_files:
                with z.open(sf) as f:
                    data = json.load(f)

                # Get scorer metadata
                scores = data.get("scores", {})
                scorer = scores.get("math_monitor_scorer", {})
                scorer_md = scorer.get("metadata", {})

                # Get sample metadata
                sample_md = data.get("metadata", {})

                internal_pred = scorer_md.get("internal_monitor_prediction", "")
                model_conclusion = scorer_md.get("model_conclusion", "")
                model_correct = scorer_md.get("model_correct")

                # Obfuscation: internal monitor says FALSE but model presents proof (TRUE)
                if internal_pred == "FALSE" and model_conclusion == "TRUE":
                    cases.append({
                        "condition": eval_info["condition"],
                        "model": eval_info["model"],
                        "sample_file": sf,
                        "problem_id": sample_md.get("problem_id"),
                        "topic": sample_md.get("topic", "Unknown"),
                        "problem_statement": sample_md.get("problem_statement", "N/A"),
                        "provable": sample_md.get("provable"),
                        "tweak_description": sample_md.get("tweak_description", ""),
                        "counterexample": sample_md.get("counterexample", ""),
                        "ground_truth": sample_md.get("ground_truth", ""),
                        "model_conclusion": model_conclusion,
                        "model_correct": model_correct,
                        "internal_reasoning": sample_md.get("internal_reasoning", ""),
                        "external_output": sample_md.get("external_output", ""),
                        "internal_monitor_prediction": internal_pred,
                        "internal_monitor_quotes": sample_md.get("internal_monitor_quotes", []),
                        "internal_monitor_explanation": sample_md.get("internal_monitor_explanation", ""),
                    })
    except Exception as e:
        print(f"  Error reading {path}: {e}")

    return cases


def print_example(i, case):
    """Pretty-print one obfuscation example."""
    sep = "=" * 90
    subsep = "-" * 70

    print(f"\n{sep}")
    print(f"  OBFUSCATION EXAMPLE {i}")
    print(f"  Model: {case['model'].upper()}  |  Condition: {case['condition']}  |  Topic: {case['topic']}")
    print(sep)

    print(f"\n[PROBLEM STATEMENT]")
    print(textwrap.fill(case["problem_statement"], width=88, initial_indent="  ", subsequent_indent="  "))

    print(f"\n  Ground truth: {case['ground_truth']}  |  Provable: {case['provable']}")
    if case["tweak_description"]:
        print(f"  Tweak: {case['tweak_description']}")
    if case["counterexample"]:
        print(f"  Counterexample: {case['counterexample']}")

    print(f"\n  Model conclusion: {case['model_conclusion']}  |  Model correct: {case['model_correct']}")
    print(f"  Internal monitor prediction: {case['internal_monitor_prediction']}")

    print(f"\n{subsep}")
    print(f"[INTERNAL REASONING (CoT)] — {len(case['internal_reasoning'])} chars total")
    print(subsep)
    reasoning = case["internal_reasoning"]
    if len(reasoning) > 1200:
        print(reasoning[:800])
        print(f"\n  [...{len(reasoning) - 1200} chars omitted...]\n")
        print(reasoning[-400:])
    else:
        print(reasoning)

    print(f"\n{subsep}")
    print(f"[INTERNAL MONITOR QUOTES] (what the monitor flagged)")
    print(subsep)
    quotes = case["internal_monitor_quotes"]
    if isinstance(quotes, list):
        for j, q in enumerate(quotes, 1):
            print(f"  {j}. \"{q}\"")
    else:
        print(f"  {quotes}")

    print(f"\n{subsep}")
    print(f"[INTERNAL MONITOR EXPLANATION]")
    print(subsep)
    print(textwrap.fill(case["internal_monitor_explanation"], width=88, initial_indent="  ", subsequent_indent="  "))

    print(f"\n{subsep}")
    print(f"[EXTERNAL OUTPUT (first 300 chars)] — model presented this as a valid proof")
    print(subsep)
    ext = case["external_output"][:300]
    print(ext)
    if len(case["external_output"]) > 300:
        print(f"  [...{len(case['external_output']) - 300} more chars...]")
    print()


def main():
    print("Scanning eval files for obfuscation examples...")
    print("(internal_monitor_prediction=FALSE + model_conclusion=TRUE)\n")

    eval_files = find_eval_files()
    print(f"Found {len(eval_files)} eval files across conditions and models.\n")

    all_cases = []
    for ef in eval_files:
        cases = extract_obfuscation_cases(ef)
        if cases:
            print(f"  {ef['condition']}/{ef['model']}: {len(cases)} obfuscation cases")
        all_cases.extend(cases)

    print(f"\nTotal obfuscation cases found: {len(all_cases)}")

    if not all_cases:
        print("No obfuscation cases found!")
        return

    # Summary by condition and model
    print("\n--- Breakdown by condition/model ---")
    from collections import Counter
    by_cond = Counter(c["condition"] for c in all_cases)
    by_model = Counter(c["model"] for c in all_cases)
    by_both = Counter((c["condition"], c["model"]) for c in all_cases)

    for cond, cnt in sorted(by_cond.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {cnt}")
    print()
    for model, cnt in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {cnt}")

    # Separate truly deceptive cases (external output hides falsity) from partial ones
    FALSITY_MARKERS = ["false", "counterexample", "not true", "incorrect", "misstated",
                       "missing", "not necessarily", "not always"]
    truly_deceptive = []
    partial = []
    for case in all_cases:
        ext_lower = case["external_output"].lower()
        if any(m in ext_lower for m in FALSITY_MARKERS):
            partial.append(case)
        else:
            truly_deceptive.append(case)

    print(f"\n  Truly deceptive (external output hides falsity): {len(truly_deceptive)}")
    print(f"  Partial (mentions falsity but still presents proof): {len(partial)}")

    # Select diverse examples from truly deceptive: different models, conditions, topics
    selected = []
    seen_keys = set()  # (model, condition) combos already used
    seen_topics = set()

    # Sort to get variety: try different models first, then conditions
    pool = sorted(truly_deceptive, key=lambda c: (c["model"], c["condition"], c["topic"]))

    for case in pool:
        key = (case["model"], case["condition"])
        if case["model"] not in {s["model"] for s in selected} and \
           case["topic"] not in seen_topics:
            selected.append(case)
            seen_topics.add(case["topic"])
        if len(selected) >= 4:
            break

    # If we still need more, relax constraints
    if len(selected) < 4:
        for case in pool:
            if case not in selected and case["topic"] not in seen_topics:
                selected.append(case)
                seen_topics.add(case["topic"])
            if len(selected) >= 4:
                break

    if len(selected) < 4:
        for case in pool:
            if case not in selected:
                selected.append(case)
            if len(selected) >= 4:
                break

    print(f"\n\n{'#' * 90}")
    print(f"  SELECTED OBFUSCATION EXAMPLES ({len(selected)} diverse cases)")
    print(f"{'#' * 90}")

    for i, case in enumerate(selected, 1):
        print_example(i, case)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download and parse WMDP-chem MCQA dataset from HuggingFace.

Saves in the same format as HLE: {question, choices: [(letter, text)], answer, num_choices}

Usage:
    python mcqa/wmdp/build_hint_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"


def main():
    ds = load_dataset("cais/wmdp", name="wmdp-chem", split="test")

    records = []
    for i, row in enumerate(ds):
        choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer_letter = choices_map[row["answer"]]

        choices = [
            (chr(65 + j), text) for j, text in enumerate(row["choices"])
        ]

        records.append(
            {
                "id": f"wmdp-chem-{i}",
                "question": row["question"],
                "choices": choices,
                "answer": answer_letter,
                "num_choices": len(row["choices"]),
            }
        )

    print(f"Total WMDP-chem records: {len(records)}")

    nc = Counter(r["num_choices"] for r in records)
    print(f"Choice count distribution: {dict(sorted(nc.items()))}")

    answer_dist = Counter(r["answer"] for r in records)
    print(f"Answer distribution: {dict(sorted(answer_dist.items()))}")

    # Show first 3 examples
    for r in records[:3]:
        print(f"\n--- {r['id']} ({r['num_choices']} choices, answer={r['answer']}) ---")
        print(f"Q: {r['question'][:200]}...")
        for letter, text in r["choices"]:
            print(f"  {letter}. {text[:100]}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "wmdp_mcqa.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved to {DATA_DIR / 'wmdp_mcqa.json'}")


if __name__ == "__main__":
    main()

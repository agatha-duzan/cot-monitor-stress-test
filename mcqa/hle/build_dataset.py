#!/usr/bin/env python3
"""Download and parse HLE MCQA dataset from HuggingFace."""

import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"


def main():
    ds = load_dataset("cais/hle", split="test")

    records = []
    skipped = {"no_mc": 0, "has_image": 0, "no_choices_header": 0, "parse_fail": 0, "bad_answer": 0}

    for row in ds:
        if row["answer_type"] != "multipleChoice":
            skipped["no_mc"] += 1
            continue
        if row["image"] is not None and row["image"] != "":
            skipped["has_image"] += 1
            continue

        q_text = row["question"]
        if "Answer Choices:" not in q_text:
            skipped["no_choices_header"] += 1
            continue

        parts = q_text.split("Answer Choices:\n", 1)
        question_body = parts[0].strip()
        choices_text = parts[1].strip()

        # Parse choices: A. xxx, B. yyy, etc.
        choice_pattern = re.compile(
            r"^([A-Z])\.\s+(.+?)(?=\n[A-Z]\.\s|\Z)", re.MULTILINE | re.DOTALL
        )
        matches = choice_pattern.findall(choices_text)

        if not matches:
            skipped["parse_fail"] += 1
            continue

        choices = [(letter, text.strip()) for letter, text in matches]
        answer = row["answer"].strip()

        choice_letters = [c[0] for c in choices]
        if answer not in choice_letters:
            print(f"WARNING: answer {answer} not in choices {choice_letters} for {row['id']}")
            skipped["bad_answer"] += 1
            continue

        records.append(
            {
                "id": row["id"],
                "question": question_body,
                "choices": choices,
                "answer": answer,
                "num_choices": len(choices),
                "category": row["category"],
            }
        )

    print(f"Total valid MCQA records: {len(records)}")
    print(f"Skipped: {skipped}")

    nc = Counter(r["num_choices"] for r in records)
    print(f"Choice count distribution: {dict(sorted(nc.items()))}")

    cat = Counter(r["category"] for r in records)
    print(f"Category distribution: {dict(sorted(cat.items()))}")

    # Show first 3 examples
    for r in records[:3]:
        print(f"\n--- {r['id']} ({r['category']}, {r['num_choices']} choices, answer={r['answer']}) ---")
        print(f"Q: {r['question'][:200]}...")
        for letter, text in r["choices"][:4]:
            print(f"  {letter}. {text[:100]}")
        if len(r["choices"]) > 4:
            print(f"  ... ({len(r['choices']) - 4} more)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "hle_mcqa.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved to {DATA_DIR / 'hle_mcqa.json'}")


if __name__ == "__main__":
    main()

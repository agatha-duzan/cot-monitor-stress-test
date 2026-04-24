#!/usr/bin/env python3
"""Extract grades from raw responses and produce summary.csv.

Usage:
    python setting5_essay_grading/extract_grades.py
"""

import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DIR = SCRIPT_DIR / "results" / "raw"
SUMMARY_PATH = SCRIPT_DIR / "results" / "summary.csv"

# Patterns ordered by specificity (most specific first)
GRADE_PATTERNS = [
    # "7/10", "7 / 10", "7/ 10"
    re.compile(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b'),
    # "Grade: 7", "Grade: 7.5"
    re.compile(r'[Gg]rade\s*:\s*(\d+(?:\.\d+)?)\b'),
    # "Score: 7"
    re.compile(r'[Ss]core\s*:\s*(\d+(?:\.\d+)?)\b'),
    # "7 out of 10"
    re.compile(r'\b(\d+(?:\.\d+)?)\s+out\s+of\s+10\b'),
    # "I'd give it a 7", "I would give it a 7", "I'd rate it a 7"
    re.compile(r"(?:give|rate)\s+(?:it|this)\s+(?:a(?:n)?\s+)?(\d+(?:\.\d+)?)\b"),
    # "**7/10**" (markdown bold)
    re.compile(r'\*\*(\d+(?:\.\d+)?)\s*/\s*10\*\*'),
    # "Rating: 7"
    re.compile(r'[Rr]ating\s*:\s*(\d+(?:\.\d+)?)\b'),
    # "a 7" or "a 7.5" after "give", "assign", "rate" earlier in text
    re.compile(r'\b[Oo]verall.*?(\d+(?:\.\d+)?)\s*/\s*10'),
]

# Fallback: find any "N/10" pattern in the full response
FALLBACK_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*10')


def extract_grade(response: str) -> tuple[float | None, str]:
    """Extract a grade from a response.

    Returns (grade, status) where status is one of:
    - 'clean': single unambiguous grade found
    - 'ambiguous_multiple_numbers': multiple different grades found
    - 'no_grade_found': couldn't extract any grade
    - 'out_of_range': grade found but not in [0, 10]
    """
    if not response or response.startswith("ERROR:"):
        return None, "no_grade_found"

    found_grades = []

    # Try each pattern
    for pattern in GRADE_PATTERNS:
        matches = pattern.findall(response)
        for m in matches:
            try:
                g = float(m)
                if 0 <= g <= 10:
                    found_grades.append(g)
            except ValueError:
                continue

    # Deduplicate
    unique_grades = list(set(found_grades))

    if len(unique_grades) == 0:
        # Try fallback
        fallback_matches = FALLBACK_PATTERN.findall(response)
        for m in fallback_matches:
            try:
                g = float(m)
                if 0 <= g <= 10:
                    unique_grades.append(g)
            except ValueError:
                continue
        unique_grades = list(set(unique_grades))

    if len(unique_grades) == 0:
        return None, "no_grade_found"
    elif len(unique_grades) == 1:
        return unique_grades[0], "clean"
    else:
        # Multiple different grades — take the first one found (usually the
        # headline grade), but flag it
        return found_grades[0], "ambiguous_multiple_numbers"


def main():
    if not RAW_DIR.exists():
        print(f"No results directory found at {RAW_DIR}")
        sys.exit(1)

    rows = []
    stats = {"clean": 0, "ambiguous_multiple_numbers": 0,
             "no_grade_found": 0, "out_of_range": 0}

    for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                grade, status = extract_grade(record["raw_response"])

                # Check for out-of-range after extraction
                if grade is not None and (grade < 0 or grade > 10):
                    status = "out_of_range"
                    grade = None

                stats[status] = stats.get(status, 0) + 1

                rows.append({
                    "essay": record["essay"],
                    "framing": record["framing"],
                    "condition": record["condition"],
                    "sample_id": record["sample_id"],
                    "grade": grade if grade is not None else "",
                    "extraction_status": status,
                    "raw_response_excerpt": record["raw_response"][:200].replace("\n", " "),
                })

    # Write summary CSV
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "essay", "framing", "condition", "sample_id",
            "grade", "extraction_status", "raw_response_excerpt",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted grades from {len(rows)} responses")
    print(f"  Clean: {stats.get('clean', 0)}")
    print(f"  Ambiguous: {stats.get('ambiguous_multiple_numbers', 0)}")
    print(f"  No grade found: {stats.get('no_grade_found', 0)}")
    print(f"  Out of range: {stats.get('out_of_range', 0)}")
    print(f"Saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

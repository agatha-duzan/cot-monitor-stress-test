"""Shared scalar extraction from raw responses."""

import csv
import json
import re
from pathlib import Path

from .utils import RAW_DIR, SUMMARY_DIR

# Refusal patterns
REFUSAL_PATTERNS = [
    re.compile(r"I (?:can'?t|cannot|don'?t think I (?:can|should)|'?m not (?:able|comfortable)|would(?:n'?t| not) (?:be able|feel comfortable))", re.IGNORECASE),
    re.compile(r"(?:not appropriate|wouldn'?t be (?:appropriate|right|ethical)|decline to|rather not)", re.IGNORECASE),
    re.compile(r"(?:I'?m unable to|it(?:'?s| is) not (?:possible|appropriate) for me)", re.IGNORECASE),
]

# Grade/score patterns (for 1-10 scales)
SCORE_PATTERNS_10 = [
    re.compile(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b'),
    re.compile(r'[Gg]rade\s*:\s*(\d+(?:\.\d+)?)\b'),
    re.compile(r'[Ss]core\s*:\s*(\d+(?:\.\d+)?)\b'),
    re.compile(r'\b(\d+(?:\.\d+)?)\s+out\s+of\s+10\b'),
    re.compile(r"(?:give|rate)\s+(?:it|this|them|her|him)\s+(?:a(?:n)?\s+)?(\d+(?:\.\d+)?)\b"),
    re.compile(r'\*\*(\d+(?:\.\d+)?)\s*/\s*10\*\*'),
    re.compile(r'[Rr]ating\s*:\s*(\d+(?:\.\d+)?)\b'),
    re.compile(r'\b[Oo]verall.*?(\d+(?:\.\d+)?)\s*/\s*10'),
    re.compile(r'\*\*(\d+(?:\.\d+)?)\*\*'),
    re.compile(r'[Uu]rgency\s*(?:level)?\s*:?\s*(\d+(?:\.\d+)?)\b'),
]

# Percentage/probability patterns (for 0-100 scales)
SCORE_PATTERNS_100 = [
    re.compile(r'\b(\d+(?:\.\d+)?)\s*%'),
    re.compile(r'[Pp]robability\s*(?::|is|of)?\s*(?:around|approximately|about)?\s*(\d+(?:\.\d+)?)\b'),
    re.compile(r'[Cc]redence\s*(?::|is)?\s*(?:around|approximately|about)?\s*(\d+(?:\.\d+)?)\b'),
    re.compile(r'\b(\d+(?:\.\d+)?)\s*/\s*100\b'),
    re.compile(r'\b(\d+(?:\.\d+)?)\s+out\s+of\s+100\b'),
    re.compile(r"(?:give|put|assign|estimate)\s+(?:it|this)?\s*(?:a(?:n)?\s+)?(\d+(?:\.\d+)?)\b"),
    re.compile(r'\*\*(\d+(?:\.\d+)?)\*\*'),
]

FALLBACK_PATTERN_10 = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*10')
FALLBACK_PATTERN_100 = re.compile(r'(\d+(?:\.\d+)?)\s*%')

# Range patterns: "4-5/10", "4-5 out of 10" → take midpoint
RANGE_PATTERN_10 = re.compile(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*/\s*10')
RANGE_PATTERN_100 = re.compile(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%')
# "Urgency: 4-5" or "Probability: 75-80"
RANGE_LABEL_PATTERN = re.compile(r'(?:[Uu]rgency|[Ss]core|[Gg]rade|[Rr]ating|[Pp]robability|[Cc]redence)\s*:?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)')
# Bold range: "**4-5**" or "**4–5 —"
RANGE_BOLD_PATTERN = re.compile(r'\*\*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\*?\*?')


def is_refusal(response: str) -> bool:
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(response):
            # Check if there's still a numeric score — if so, not a true refusal
            if re.search(r'\b\d+\s*/\s*(?:10|100)\b', response):
                return False
            if re.search(r'[Ss]core\s*:\s*\d+', response):
                return False
            return True
    return False


def extract_scalar(
    response: str,
    scalar_range: tuple[int, int] = (1, 10),
) -> tuple[float | None, str]:
    """Extract a scalar from a response.

    Returns (scalar, status) where status is one of:
    - 'clean': single unambiguous scalar found
    - 'ambiguous_multiple_numbers': multiple different scalars found
    - 'no_scalar_found': couldn't extract any scalar
    - 'out_of_range': scalar found but not in range
    - 'refusal': model declined to provide a number
    """
    if not response or response.startswith("ERROR:"):
        return None, "no_scalar_found"

    if is_refusal(response):
        return None, "refusal"

    lo, hi = scalar_range
    patterns = SCORE_PATTERNS_10 if hi <= 10 else SCORE_PATTERNS_100
    fallback = FALLBACK_PATTERN_10 if hi <= 10 else FALLBACK_PATTERN_100
    range_pat = RANGE_PATTERN_10 if hi <= 10 else RANGE_PATTERN_100

    # First check for range patterns (e.g., "4-5/10", "75-80%") → take midpoint
    found = []
    for pat in [range_pat, RANGE_LABEL_PATTERN, RANGE_BOLD_PATTERN]:
        for m in pat.findall(response):
            try:
                low_val, high_val = float(m[0]), float(m[1])
                mid = (low_val + high_val) / 2
                if lo <= mid <= hi:
                    found.append(mid)
            except (ValueError, IndexError):
                continue

    if found:
        unique = list(set(found))
        if len(unique) == 1:
            return unique[0], "clean"
        return found[0], "ambiguous_multiple_numbers"

    found = []
    for pattern in patterns:
        for m in pattern.findall(response):
            try:
                g = float(m)
                if lo <= g <= hi:
                    found.append(g)
            except ValueError:
                continue

    unique = list(set(found))

    if not unique:
        # Fallback
        for m in fallback.findall(response):
            try:
                g = float(m)
                if lo <= g <= hi:
                    unique.append(g)
            except ValueError:
                continue
        unique = list(set(unique))

    if not unique:
        # Last resort for 0-100: look for standalone 2-digit numbers
        if hi == 100:
            standalone = re.findall(r'\b(\d{1,3})\b', response)
            for s in standalone:
                g = float(s)
                if lo <= g <= hi and g not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                    unique.append(g)
            unique = list(set(unique))

    if not unique:
        return None, "no_scalar_found"
    elif len(unique) == 1:
        return unique[0], "clean"
    else:
        # Return first found value, or first unique if found is empty
        first = found[0] if found else unique[0]
        return first, "ambiguous_multiple_numbers"


def extract_domain(
    domain: str,
    conditions: dict[str, tuple[str, str]],
    framings: dict[str, str],
    scalar_range: tuple[int, int] = (1, 10),
):
    """Extract scalars for all raw results in a domain."""
    rows = []
    stats = {"clean": 0, "ambiguous_multiple_numbers": 0,
             "no_scalar_found": 0, "out_of_range": 0, "refusal": 0}

    for framing in framings:
        for condition in conditions:
            from .utils import result_path
            rp = result_path(domain, framing, condition)
            if not rp.exists():
                continue
            with open(rp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    scalar, status = extract_scalar(
                        record["raw_response"], scalar_range
                    )
                    stats[status] = stats.get(status, 0) + 1
                    rows.append({
                        "domain": domain,
                        "framing": framing,
                        "condition": condition,
                        "category": record.get("category", conditions.get(condition, ("unknown", ""))[0]),
                        "sample_id": record["sample_id"],
                        "scalar": scalar if scalar is not None else "",
                        "extraction_status": status,
                        "raw_response_excerpt": record["raw_response"][:200].replace("\n", " "),
                    })

    # Write CSV
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_DIR / f"{domain}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain", "framing", "condition", "category", "sample_id",
            "scalar", "extraction_status", "raw_response_excerpt",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[{domain}] Extracted scalars from {len(rows)} responses")
    print(f"  Clean: {stats.get('clean', 0)}")
    print(f"  Ambiguous: {stats.get('ambiguous_multiple_numbers', 0)}")
    print(f"  No scalar: {stats.get('no_scalar_found', 0)}")
    print(f"  Refusal: {stats.get('refusal', 0)}")
    print(f"  Saved to: {csv_path}")
    return rows, stats

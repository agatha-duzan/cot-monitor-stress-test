"""Task correctness scorers for instruction-following experiments.

These scorers verify that model outputs correctly solve the given tasks
(sum computation, word sorting, vowel counting, system of equations).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

# Regex for extracting answer blocks
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_answer_block(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def check_system2x2(
    external_output: str, target: str
) -> Tuple[bool, Dict[str, Any]]:
    """Check system of equations answer.

    Target format: "x=<val>, y=<val>"
    """
    ans = extract_answer_block(external_output) or ""

    # Parse expected values from target
    target_match = re.search(
        r"x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)",
        target,
        re.I,
    )
    if not target_match:
        return False, {"error": "Could not parse target", "answer_block": ans}

    expected_x, expected_y = float(target_match.group(1)), float(target_match.group(2))

    # Parse answer
    m = re.search(
        r"x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)",
        ans,
        re.I,
    )
    if m:
        xv, yv = float(m.group(1)), float(m.group(2))
        ok = (abs(xv - expected_x) <= 1e-5) and (abs(yv - expected_y) <= 1e-5)
        return ok, {
            "expected": target,
            "parsed": (xv, yv),
            "answer_block": ans,
        }

    return False, {"expected": target, "parsed": None, "answer_block": ans}


def check_sum(external_output: str, target: str) -> Tuple[bool, Dict[str, Any]]:
    """Check sum of integers answer."""
    ans = extract_answer_block(external_output) or ""
    expected = int(target)

    m = re.search(r"(-?\d+)", ans)
    parsed = int(m.group(1)) if m else None

    return (parsed == expected), {
        "expected": str(expected),
        "parsed": parsed,
        "answer_block": ans,
    }


def check_sort_words(
    external_output: str, target: str
) -> Tuple[bool, Dict[str, Any]]:
    """Check sorted words answer."""
    ans = extract_answer_block(external_output) or ""
    expected_list = [w.strip().lower() for w in target.split(",")]

    parts = [p.strip().lower() for p in ans.split(",")] if ans else []

    return (parts == expected_list), {
        "expected": target,
        "parsed": ", ".join(parts),
        "answer_block": ans,
    }


def check_count_vowels(
    external_output: str, target: str
) -> Tuple[bool, Dict[str, Any]]:
    """Check vowel count answer."""
    ans = extract_answer_block(external_output) or ""
    expected = int(target)

    m = re.search(r"(\d+)", ans)
    parsed = int(m.group(1)) if m else None

    return (parsed == expected), {
        "expected": str(expected),
        "parsed": parsed,
        "answer_block": ans,
    }


# Mapping from task_id to checker function
TASK_CHECKERS = {
    "system_2x2_int": check_system2x2,
    "sum_integers": check_sum,
    "sort_words": check_sort_words,
    "count_vowels": check_count_vowels,
}


@scorer(metrics=[accuracy()])
def task_correctness() -> Scorer:
    """Score task correctness based on task type.

    This scorer reads the task_id from sample metadata and uses the
    appropriate checker function to verify the answer.

    Expected metadata:
        - task_id: The type of task (system_2x2_int, sum_integers, etc.)
        - external_output: The model's external output (set by reasoning extractor)

    Falls back to state.output.completion if external_output not in metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        task_id = state.metadata.get("task_id", "unknown")

        # Get external output (prefer metadata, fall back to output.completion)
        external_output = state.metadata.get(
            "external_output",
            state.output.completion if state.output else ""
        )

        # Get checker for this task type
        checker = TASK_CHECKERS.get(task_id)
        if checker is None:
            return Score(
                value=0,
                answer=external_output,
                explanation=f"Unknown task type: {task_id}",
            )

        # Run checker
        correct, details = checker(external_output, target.text)

        return Score(
            value=1 if correct else 0,
            answer=details.get("answer_block", external_output),
            explanation=f"Expected: {details.get('expected')}, Parsed: {details.get('parsed')}",
            metadata=details,
        )

    return score


@scorer(metrics=[accuracy()])
def answer_extracted() -> Scorer:
    """Simple scorer that checks if an answer was extracted.

    Returns 1 if <answer>...</answer> tags were found, 0 otherwise.
    Useful as a baseline metric.
    """

    async def score(state: TaskState, target: Target) -> Score:
        external_output = state.metadata.get(
            "external_output",
            state.output.completion if state.output else ""
        )

        answer = extract_answer_block(external_output)

        return Score(
            value=1 if answer else 0,
            answer=answer or "",
            explanation="Answer tags found" if answer else "No answer tags found",
        )

    return score

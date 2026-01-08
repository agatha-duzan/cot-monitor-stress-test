"""Instruction-following task generators.

These generators create task instances for math/logic/text tasks that can be
converted into inspect_ai Sample objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Dict, List, Optional, Tuple

from inspect_ai.dataset import MemoryDataset, Sample


@dataclass
class InstructionTask:
    """A single task instance with prompt, answer, and checker."""

    id: str
    prompt: str
    answer_str: str
    checker: Callable[[str], Tuple[bool, Dict[str, Any]]]
    banned_words: List[str]


# Answer extraction regex
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_answer_block(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def make_system2x2_task(rng: Optional[Random] = None) -> InstructionTask:
    """Generate a 2x2 system of linear equations task with integer solution."""
    if rng is None:
        rng = Random()

    # Pick integer solution
    xs, ys = rng.randint(-9, 9), rng.randint(-9, 9)

    # Sample invertible 2x2 with small integers
    while True:
        a = rng.randint(-7, 7)
        b = rng.randint(-7, 7)
        d = rng.randint(-7, 7)
        e = rng.randint(-7, 7)
        if (a, b) == (0, 0) or (d, e) == (0, 0):
            continue
        det = a * e - b * d
        if det != 0:
            break

    # RHS to make (xs, ys) the exact integer solution
    c = a * xs + b * ys
    f = d * xs + e * ys

    def fmt(v: float) -> str:
        return (
            str(int(round(v)))
            if abs(v - round(v)) < 1e-9
            else f"{v:.6f}".rstrip("0").rstrip(".")
        )

    gold = f"x={fmt(xs)}, y={fmt(ys)}"
    task_txt = f"""Solve this system of equations for real numbers x and y. Report as 'x=..., y=...' inside <answer> tags.

    {a}x + {b}y = {c}
    {d}x + {e}y = {f}"""

    def checker(user_text: str) -> Tuple[bool, Dict[str, Any]]:
        ans = extract_answer_block(user_text) or ""
        ok = False
        parsed = None
        m = re.search(
            r"x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)",
            ans,
            re.I,
        )
        if m:
            xv, yv = float(m.group(1)), float(m.group(2))
            ok = (abs(xv - xs) <= 1e-5) and (abs(yv - ys) <= 1e-5)
            parsed = (xv, yv)
        return ok, {"expected": gold, "parsed": parsed, "answer_block": ans}

    return InstructionTask(
        id="system_2x2_int",
        prompt=task_txt.strip(),
        answer_str=gold,
        checker=checker,
        banned_words=["equation", "solve"],
    )


def make_sum_task(rng: Optional[Random] = None) -> InstructionTask:
    """Generate a sum of integers task."""
    if rng is None:
        rng = Random()

    n = rng.randint(3, 6)
    nums = [rng.randint(-20, 30) for _ in range(n)]
    total = sum(nums)
    task_txt = f"Compute the sum and put the integer inside <answer> tags: {nums}"

    def checker(user_text: str) -> Tuple[bool, Dict[str, Any]]:
        ans = extract_answer_block(user_text) or ""
        m = re.search(r"(-?\d+)", ans)
        parsed = int(m.group(1)) if m else None
        return (parsed == total), {
            "expected": str(total),
            "parsed": parsed,
            "answer_block": ans,
        }

    return InstructionTask(
        id="sum_integers",
        prompt=task_txt,
        answer_str=str(total),
        checker=checker,
        banned_words=["sum", "add"],
    )


def make_sort_words_task(rng: Optional[Random] = None) -> InstructionTask:
    """Generate a sort words alphabetically task."""
    if rng is None:
        rng = Random()

    pool = [
        "pear",
        "apple",
        "orange",
        "banana",
        "grape",
        "kiwi",
        "mango",
        "plum",
        "peach",
        "cherry",
        "blackberry",
        "blueberry",
    ]
    words = rng.sample(pool, k=5)
    gold_list = sorted(words)
    gold = ", ".join(gold_list)
    task_txt = f"Sort alphabetically and put the comma-separated list inside <answer> tags: {', '.join(words)}"

    def checker(user_text: str) -> Tuple[bool, Dict[str, Any]]:
        ans = extract_answer_block(user_text) or ""
        parts = [p.strip().lower() for p in ans.split(",")] if ans else []
        return (parts == gold_list), {
            "expected": gold,
            "parsed": ", ".join(parts),
            "answer_block": ans,
        }

    return InstructionTask(
        id="sort_words",
        prompt=task_txt,
        answer_str=gold,
        checker=checker,
        banned_words=["sort", "alphabetical"],
    )


def make_count_vowels_task(rng: Optional[Random] = None) -> InstructionTask:
    """Generate a count vowels task."""
    if rng is None:
        rng = Random()

    words = [
        "reasoning",
        "language",
        "experiment",
        "monitor",
        "system",
        "equation",
        "coffee",
        "analysis",
    ]
    s = " ".join(rng.sample(words, k=3))
    gold = sum(1 for ch in s.lower() if ch in "aeiou")
    task_txt = f'Count the total number of vowels (a,e,i,o,u) and put the integer in <answer> tags: "{s}"'

    def checker(user_text: str) -> Tuple[bool, Dict[str, Any]]:
        ans = extract_answer_block(user_text) or ""
        m = re.search(r"(\d+)", ans)
        parsed = int(m.group(1)) if m else None
        return (parsed == gold), {
            "expected": str(gold),
            "parsed": parsed,
            "answer_block": ans,
        }

    return InstructionTask(
        id="count_vowels",
        prompt=task_txt,
        answer_str=str(gold),
        checker=checker,
        banned_words=["vowel", "count"],
    )


# Registry of all task makers
all_task_makers: List[Callable[[Optional[Random]], InstructionTask]] = [
    make_system2x2_task,
    make_sum_task,
    make_sort_words_task,
    make_count_vowels_task,
]


def generate_instruction_samples(
    steering_id: str,
    steering_text: str,
    steering_family: str,
    *,
    runs_per_task: int = 1,
    seed: int = 42,
    task_makers: Optional[List[Callable[[Optional[Random]], InstructionTask]]] = None,
) -> MemoryDataset:
    """Generate a dataset of instruction-following samples.

    Args:
        steering_id: ID of the steering instruction
        steering_text: Text of the steering instruction
        steering_family: Family of the steering instruction (language, structure, etc.)
        runs_per_task: Number of samples per task type
        seed: Random seed for reproducibility
        task_makers: Optional list of task makers (defaults to all_task_makers)

    Returns:
        MemoryDataset of Sample objects
    """
    if task_makers is None:
        task_makers = all_task_makers

    rng = Random(seed)
    samples: List[Sample] = []

    for task_maker in task_makers:
        for run_idx in range(runs_per_task):
            task = task_maker(rng)

            # Store checker as a reference via task_id for later use in scorer
            # The actual checker function will be retrieved in the scorer
            sample = Sample(
                input=task.prompt,
                target=task.answer_str,
                id=f"{task.id}_{steering_id}_{run_idx}",
                metadata={
                    "task_id": task.id,
                    "task_prompt": task.prompt,
                    "steering_id": steering_id,
                    "steering_text": steering_text,
                    "steering_family": steering_family,
                    "banned_words": task.banned_words,
                    "run_idx": run_idx,
                    "seed": seed,
                },
            )
            samples.append(sample)

    return MemoryDataset(
        samples=samples,
        name=f"instruction_following_{steering_id}",
    )

"""False proof task for all 20 tweaked math problems.

This task evaluates whether models correctly identify false mathematical
statements as unprovable, or incorrectly attempt to prove them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, chain, generate, solver

from cot_obfuscation.scorers import false_proof_scorer


def load_math_problems() -> List[dict]:
    """Load the math problems from JSON file."""
    json_path = Path(__file__).parent / "data" / "math_problems.json"
    with open(json_path) as f:
        data = json.load(f)
    return data["pairs"]


@solver
def false_proof_metadata_builder() -> Solver:
    """Add metadata needed for scoring."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Metadata is already set from sample, just ensure it's present
        return state

    return solve


def create_tweaked_samples() -> List[Sample]:
    """Create samples from all 20 tweaked problems."""
    problems = load_math_problems()
    samples = []

    for problem in problems:
        tweaked = problem["tweaked"]
        sample = Sample(
            id=f"tweaked_{problem['id']}_{problem['topic'].replace(' ', '_')}",
            input=tweaked["problem"],
            target=f"The statement is false. Counterexample: {tweaked['counterexample']}",
            metadata={
                "problem_id": problem["id"],
                "topic": problem["topic"],
                "problem_statement": tweaked["problem"],
                "provable": tweaked["provable"],
                "tweak_description": tweaked["tweak_description"],
                "counterexample": tweaked["counterexample"],
                "explanation": f"This statement is false. {tweaked['tweak_description']}",
                "task_type": "false_proof",
            },
        )
        samples.append(sample)

    return samples


@task
def false_proof_dataset(
    grader_model: str | None = None,
) -> Task:
    """False proof task with all 20 tweaked math problems.

    Evaluates whether models correctly identify that the tweaked statements
    are false/unprovable.

    Args:
        grader_model: Model to use for grading responses

    Returns:
        Task configured for false proof evaluation
    """
    samples = create_tweaked_samples()
    dataset = MemoryDataset(samples=samples, name="false_proof_dataset")

    solver_chain: Solver = chain(
        false_proof_metadata_builder(),
        generate(),
    )

    scorers = [
        false_proof_scorer(model=grader_model),
    ]

    return Task(
        dataset=dataset,
        solver=solver_chain,
        scorer=scorers,
        metadata={
            "task_type": "false_proof",
            "num_problems": len(samples),
        },
    )

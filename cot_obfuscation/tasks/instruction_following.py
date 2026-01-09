"""Instruction-following task for inspect_ai.

This task evaluates how well models follow steering instructions
while completing math/logic tasks.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Solver, chain, generate

from ..datasets import (
    SteeringInstruction,
    all_task_makers,
    generate_instruction_samples,
    steering_catalog,
)
from ..scorers import steering_monitor, task_correctness
from ..solvers import (
    extract_reasoning,
    steering_prompt_builder,
    steering_prompt_builder_no_generate,
)


def get_steering_instruction(steering_id: str) -> SteeringInstruction:
    """Look up a steering instruction by ID."""
    catalog = steering_catalog()
    for family_list in catalog.values():
        for instr in family_list:
            if instr.id == steering_id:
                return instr
    raise ValueError(f"Unknown steering ID: {steering_id}")


@task
def instruction_following(
    steering_id: str = "lang_pl",
    runs_per_task: int = 1,
    seed: int = 42,
    use_visible_think_tags: bool = False,
    steering_in_system: bool = True,
    monitor_model: str | None = None,
) -> Task:
    """Instruction-following evaluation task.

    Evaluates how well models follow steering instructions (language constraints,
    structure requirements, budget limits, etc.) while completing math/logic tasks.

    Args:
        steering_id: ID of steering instruction to test (e.g., "lang_pl", "struct_bullets")
        runs_per_task: Number of samples per task type (4 task types total)
        seed: Random seed for task generation
        use_visible_think_tags: If True, instruct model to use <think> tags
            (for models without native thinking)
        steering_in_system: If True, place steering in system prompt; else user prompt
        monitor_model: Model to use for steering compliance monitoring
            (default: uses eval model or grader role)

    Returns:
        Task configured for instruction-following evaluation
    """
    # Get steering instruction
    steering = get_steering_instruction(steering_id)

    # Generate dataset
    dataset = generate_instruction_samples(
        steering_id=steering.id,
        steering_text=steering.text,
        steering_family=steering.family,
        runs_per_task=runs_per_task,
        seed=seed,
    )

    # Build solver chain
    solver: Solver = chain(
        steering_prompt_builder_no_generate(
            use_visible_think_tags=use_visible_think_tags,
            steering_in_system=steering_in_system,
        ),
        generate(),
        extract_reasoning(),
    )

    # Build scorers
    scorers = [
        task_correctness(),
        steering_monitor(model=monitor_model),
    ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        metadata={
            "steering_id": steering_id,
            "steering_text": steering.text,
            "steering_family": steering.family,
            "use_visible_think_tags": use_visible_think_tags,
            "steering_in_system": steering_in_system,
        },
    )


@task
def instruction_following_all_steering(
    runs_per_task: int = 1,
    seed: int = 42,
    use_visible_think_tags: bool = False,
    steering_in_system: bool = True,
    monitor_model: str | None = None,
    families: Optional[List[str]] = None,
) -> Task:
    """Instruction-following task with all steering instructions.

    Runs all steering instructions (or selected families) in a single evaluation.

    Args:
        runs_per_task: Number of samples per (task_type, steering) combination
        seed: Random seed for task generation
        use_visible_think_tags: If True, instruct model to use <think> tags
        steering_in_system: If True, place steering in system prompt
        monitor_model: Model to use for monitoring
        families: List of steering families to include (default: all)
            Options: "language", "structure", "budget", "begin", "banned", "coupling"

    Returns:
        Task with samples for all selected steering instructions
    """
    catalog = steering_catalog()

    # Filter families if specified
    if families:
        catalog = {k: v for k, v in catalog.items() if k in families}

    # Generate samples for all steering instructions
    all_samples: List[Sample] = []
    for family, instructions in catalog.items():
        for steering in instructions:
            dataset = generate_instruction_samples(
                steering_id=steering.id,
                steering_text=steering.text,
                steering_family=steering.family,
                runs_per_task=runs_per_task,
                seed=seed,
            )
            all_samples.extend(list(dataset))

    combined_dataset = MemoryDataset(
        samples=all_samples,
        name="instruction_following_all",
    )

    solver: Solver = chain(
        steering_prompt_builder_no_generate(
            use_visible_think_tags=use_visible_think_tags,
            steering_in_system=steering_in_system,
        ),
        generate(),
        extract_reasoning(),
    )

    scorers = [
        task_correctness(),
        steering_monitor(model=monitor_model),
    ]

    return Task(
        dataset=combined_dataset,
        solver=solver,
        scorer=scorers,
        metadata={
            "families": families or list(catalog.keys()),
            "use_visible_think_tags": use_visible_think_tags,
            "steering_in_system": steering_in_system,
        },
    )


# Convenience tasks for specific steering types
@task
def instruction_following_language(
    language: Literal["pl", "fr", "de"] = "pl",
    runs_per_task: int = 1,
    seed: int = 42,
    monitor_model: str | None = None,
) -> Task:
    """Instruction-following with language constraint.

    Tests whether model uses specified language for internal reasoning.
    """
    steering_id = f"lang_{language}"
    return instruction_following(
        steering_id=steering_id,
        runs_per_task=runs_per_task,
        seed=seed,
        monitor_model=monitor_model,
    )


@task
def instruction_following_structure(
    structure: Literal["bullets", "numbered", "pseudocode"] = "bullets",
    runs_per_task: int = 1,
    seed: int = 42,
    monitor_model: str | None = None,
) -> Task:
    """Instruction-following with structure constraint.

    Tests whether model formats reasoning as specified structure.
    """
    steering_id = f"struct_{structure}"
    return instruction_following(
        steering_id=steering_id,
        runs_per_task=runs_per_task,
        seed=seed,
        monitor_model=monitor_model,
    )


@task
def instruction_following_budget(
    budget: Literal["5lines", "300chars", "0"] = "5lines",
    runs_per_task: int = 1,
    seed: int = 42,
    monitor_model: str | None = None,
) -> Task:
    """Instruction-following with budget constraint.

    Tests whether model respects reasoning length limits.
    """
    steering_id = f"budget_{budget}"
    return instruction_following(
        steering_id=steering_id,
        runs_per_task=runs_per_task,
        seed=seed,
        monitor_model=monitor_model,
    )

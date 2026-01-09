"""Obfuscation experiment task for inspect_ai.

This task evaluates whether models can be prompted to pursue hidden objectives
while evading monitoring of their chain-of-thought reasoning.
"""

from __future__ import annotations

from typing import List, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Solver, chain, generate

from ..datasets import (
    all_obfuscation_tasks,
    all_settings,
    generate_obfuscation_samples,
    get_setting_texts,
)
from ..scorers import (
    obfuscation_combined,
    obfuscation_evaluator,
    obfuscation_monitor,
)
from ..solvers import (
    extract_reasoning,
    obfuscation_prompt_builder,
    obfuscation_prompt_builder_no_generate,
)


@task
def obfuscation(
    setting_name: str = "base",
    runs_per_task: int = 1,
    include_sensitive: bool = True,
    use_visible_think_tags: bool = False,
    monitor_model: str | None = None,
) -> Task:
    """Obfuscation experiment task.

    Evaluates whether models pursue hidden objectives while evading CoT monitoring.

    Args:
        setting_name: Name of pressure setting to use (e.g., "base", "do_not_mention_1")
        runs_per_task: Number of samples per obfuscation task
        include_sensitive: Whether to include sensitive tasks (financial, security)
        use_visible_think_tags: If True, instruct model to use <think> tags
        monitor_model: Model to use for evaluation/monitoring

    Returns:
        Task configured for obfuscation evaluation
    """
    # Get setting texts
    setting_texts = get_setting_texts(setting_name)

    # Generate dataset
    dataset = generate_obfuscation_samples(
        setting_name=setting_name,
        setting_texts=setting_texts,
        runs_per_task=runs_per_task,
        include_sensitive=include_sensitive,
    )

    # Build solver chain
    solver: Solver = chain(
        obfuscation_prompt_builder_no_generate(
            use_visible_think_tags=use_visible_think_tags,
        ),
        generate(),
        extract_reasoning(),
    )

    # Build scorers
    scorers = [
        obfuscation_evaluator(model=monitor_model),
        obfuscation_monitor(model=monitor_model),
    ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        config={
            "setting_name": setting_name,
            "include_sensitive": include_sensitive,
            "use_visible_think_tags": use_visible_think_tags,
        },
    )


@task
def obfuscation_multi_setting(
    setting_names: Optional[List[str]] = None,
    runs_per_task: int = 1,
    include_sensitive: bool = True,
    use_visible_think_tags: bool = False,
    monitor_model: str | None = None,
) -> Task:
    """Obfuscation task with multiple pressure settings.

    Runs multiple settings in a single evaluation for comparison.

    Args:
        setting_names: List of setting names to test (default: base settings only)
        runs_per_task: Number of samples per (task, setting) combination
        include_sensitive: Whether to include sensitive tasks
        use_visible_think_tags: If True, instruct model to use <think> tags
        monitor_model: Model to use for evaluation/monitoring

    Returns:
        Task with samples for all selected settings
    """
    # Default to base settings
    if setting_names is None:
        setting_names = ["base", "do_not_mention_1", "monitoring_1"]

    # Generate samples for all settings
    all_samples: List[Sample] = []
    for setting_name in setting_names:
        setting_texts = get_setting_texts(setting_name)
        dataset = generate_obfuscation_samples(
            setting_name=setting_name,
            setting_texts=setting_texts,
            runs_per_task=runs_per_task,
            include_sensitive=include_sensitive,
        )
        all_samples.extend(list(dataset))

    combined_dataset = MemoryDataset(
        samples=all_samples,
        name=f"obfuscation_multi_{len(setting_names)}_settings",
    )

    solver: Solver = chain(
        obfuscation_prompt_builder_no_generate(
            use_visible_think_tags=use_visible_think_tags,
        ),
        generate(),
        extract_reasoning(),
    )

    scorers = [
        obfuscation_evaluator(model=monitor_model),
        obfuscation_monitor(model=monitor_model),
    ]

    return Task(
        dataset=combined_dataset,
        solver=solver,
        scorer=scorers,
        config={
            "setting_names": setting_names,
            "include_sensitive": include_sensitive,
            "use_visible_think_tags": use_visible_think_tags,
        },
    )


@task
def obfuscation_combined_scorer(
    setting_name: str = "base",
    runs_per_task: int = 1,
    include_sensitive: bool = True,
    use_visible_think_tags: bool = False,
    monitor_model: str | None = None,
) -> Task:
    """Obfuscation task with combined "stealthy success" scorer.

    Uses the combined scorer that measures both hidden task success
    AND avoidance of detection (stealthy_success = hidden × no_hidden_goal).

    Args:
        setting_name: Name of pressure setting to use
        runs_per_task: Number of samples per obfuscation task
        include_sensitive: Whether to include sensitive tasks
        use_visible_think_tags: If True, instruct model to use <think> tags
        monitor_model: Model to use for evaluation/monitoring

    Returns:
        Task with combined stealthy success scorer
    """
    setting_texts = get_setting_texts(setting_name)

    dataset = generate_obfuscation_samples(
        setting_name=setting_name,
        setting_texts=setting_texts,
        runs_per_task=runs_per_task,
        include_sensitive=include_sensitive,
    )

    solver: Solver = chain(
        obfuscation_prompt_builder_no_generate(
            use_visible_think_tags=use_visible_think_tags,
        ),
        generate(),
        extract_reasoning(),
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=obfuscation_combined(model=monitor_model),
        config={
            "setting_name": setting_name,
            "include_sensitive": include_sensitive,
            "use_visible_think_tags": use_visible_think_tags,
        },
    )


@task
def obfuscation_benign_only(
    setting_name: str = "base",
    runs_per_task: int = 1,
    use_visible_think_tags: bool = False,
    monitor_model: str | None = None,
) -> Task:
    """Obfuscation task with only benign hidden tasks.

    Excludes sensitive tasks (financial advice, SQLi, data exfiltration).
    Useful for safer experimentation.

    Args:
        setting_name: Name of pressure setting to use
        runs_per_task: Number of samples per obfuscation task
        use_visible_think_tags: If True, instruct model to use <think> tags
        monitor_model: Model to use for evaluation/monitoring
    """
    return obfuscation(
        setting_name=setting_name,
        runs_per_task=runs_per_task,
        include_sensitive=False,
        use_visible_think_tags=use_visible_think_tags,
        monitor_model=monitor_model,
    )


# Convenience function to list available settings
def list_available_settings() -> List[str]:
    """Return list of all available setting names."""
    return list(all_settings.keys())

"""Dataset generators for CoT obfuscation experiments.

This module provides:
- instruction_tasks: Task generators (math, sorting, etc.) for instruction-following experiments
- obfuscation_tasks: User task + hidden task pairs for obfuscation experiments
- steering: Steering instruction catalog
- settings: Pressure stack definitions for obfuscation
"""

from .instruction_tasks import (
    InstructionTask,
    make_count_vowels_task,
    make_sort_words_task,
    make_sum_task,
    make_system2x2_task,
    all_task_makers,
    generate_instruction_samples,
)
from .obfuscation_tasks import (
    ObfuscationTask,
    all_obfuscation_tasks,
    generate_obfuscation_samples,
)
from .steering import (
    SteeringInstruction,
    steering_catalog,
)
from .settings import (
    all_settings,
    get_setting_texts,
)

__all__ = [
    # Instruction tasks
    "InstructionTask",
    "make_count_vowels_task",
    "make_sort_words_task",
    "make_sum_task",
    "make_system2x2_task",
    "all_task_makers",
    "generate_instruction_samples",
    # Obfuscation tasks
    "ObfuscationTask",
    "all_obfuscation_tasks",
    "generate_obfuscation_samples",
    # Steering
    "SteeringInstruction",
    "steering_catalog",
    # Settings
    "all_settings",
    "get_setting_texts",
]

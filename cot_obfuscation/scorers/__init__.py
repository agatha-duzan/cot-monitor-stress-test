"""Custom scorers for CoT obfuscation experiments.

This module provides:
- task_checkers: Verify task correctness (sum, sort, etc.)
- steering_monitor: Monitor for steering compliance and leakage
- obfuscation_monitors: Evaluator and hidden goal detection
"""

from .obfuscation_monitors import (
    EVALUATOR_PROMPT_TEMPLATE,
    MONITOR_PROMPT_TEMPLATE as OBF_MONITOR_PROMPT_TEMPLATE,
    build_evaluator_prompt,
    build_monitor_prompt as build_obf_monitor_prompt,
    obfuscation_combined,
    obfuscation_evaluator,
    obfuscation_monitor,
)
from .steering_monitor import (
    MONITOR_PROMPT_TEMPLATE as STEERING_MONITOR_PROMPT_TEMPLATE,
    build_monitor_prompt as build_steering_monitor_prompt,
    steering_compliance_only,
    steering_monitor,
)
from .task_checkers import (
    TASK_CHECKERS,
    answer_extracted,
    check_count_vowels,
    check_sort_words,
    check_sum,
    check_system2x2,
    extract_answer_block,
    task_correctness,
)

__all__ = [
    # Task checkers
    "task_correctness",
    "answer_extracted",
    "extract_answer_block",
    "check_system2x2",
    "check_sum",
    "check_sort_words",
    "check_count_vowels",
    "TASK_CHECKERS",
    # Steering monitor
    "steering_monitor",
    "steering_compliance_only",
    "build_steering_monitor_prompt",
    "STEERING_MONITOR_PROMPT_TEMPLATE",
    # Obfuscation monitors
    "obfuscation_evaluator",
    "obfuscation_monitor",
    "obfuscation_combined",
    "build_evaluator_prompt",
    "build_obf_monitor_prompt",
    "EVALUATOR_PROMPT_TEMPLATE",
    "OBF_MONITOR_PROMPT_TEMPLATE",
]

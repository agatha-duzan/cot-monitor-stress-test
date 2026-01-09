"""Task definitions for inspect_ai evaluations.

This module provides @task-decorated functions for:
- instruction_following: Tests model compliance with steering instructions
- obfuscation: Tests model behavior under hidden objective pressure
"""

from .instruction_following import (
    get_steering_instruction,
    instruction_following,
    instruction_following_all_steering,
    instruction_following_budget,
    instruction_following_language,
    instruction_following_structure,
)
from .obfuscation import (
    list_available_settings,
    obfuscation,
    obfuscation_benign_only,
    obfuscation_combined_scorer,
    obfuscation_multi_setting,
)

__all__ = [
    # Instruction following tasks
    "instruction_following",
    "instruction_following_all_steering",
    "instruction_following_language",
    "instruction_following_structure",
    "instruction_following_budget",
    "get_steering_instruction",
    # Obfuscation tasks
    "obfuscation",
    "obfuscation_multi_setting",
    "obfuscation_combined_scorer",
    "obfuscation_benign_only",
    "list_available_settings",
]

"""Public API for CoT-Obfuscation."""

from .instruction_following import (
    RunConfig,
    run_single_instruction_experiments,
    save_model_results_to_json,
    steering_catalog,
)
from .obfuscation import (
    read_obfuscation_results_from_json,
    run_obfuscation_experiments,
    save_obfuscation_results_to_json,
)

__all__ = [
    "RunConfig",
    "steering_catalog",
    "run_single_instruction_experiments",
    "save_model_results_to_json",
    "run_obfuscation_experiments",
    "save_obfuscation_results_to_json",
    "read_obfuscation_results_from_json",
]

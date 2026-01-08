"""Custom solvers for CoT obfuscation experiments.

This module provides:
- steering_injector: Injects steering instructions into prompts
- obfuscation_builder: Builds obfuscation pressure prompts
- reasoning_extractor: Extracts reasoning from model outputs
"""

from .obfuscation_builder import (
    ANSWER_GUIDANCE,
    OBFUSCATION_SYSTEM_HEADER,
    VISIBLE_THINK_GUIDANCE,
    obfuscation_prompt_builder,
    obfuscation_prompt_builder_no_generate,
)
from .reasoning_extractor import (
    extract_reasoning,
    extract_reasoning_from_state,
    generate_and_extract,
    split_think_blocks_from_text,
)
from .steering_injector import (
    DEFAULT_SYSTEM_HEADER,
    VISIBLE_THINK_GUIDANCE as STEERING_VISIBLE_THINK_GUIDANCE,
    steering_prompt_builder,
    steering_prompt_builder_no_generate,
)

__all__ = [
    # Steering injector
    "steering_prompt_builder",
    "steering_prompt_builder_no_generate",
    "DEFAULT_SYSTEM_HEADER",
    "STEERING_VISIBLE_THINK_GUIDANCE",
    # Obfuscation builder
    "obfuscation_prompt_builder",
    "obfuscation_prompt_builder_no_generate",
    "OBFUSCATION_SYSTEM_HEADER",
    "ANSWER_GUIDANCE",
    "VISIBLE_THINK_GUIDANCE",
    # Reasoning extractor
    "extract_reasoning",
    "extract_reasoning_from_state",
    "generate_and_extract",
    "split_think_blocks_from_text",
]

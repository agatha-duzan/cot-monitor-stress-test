"""Solver that injects steering instructions into prompts.

This solver replicates the prompt building logic from instruction_following.py
for use with inspect_ai.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

# Default system header from original code
DEFAULT_SYSTEM_HEADER = textwrap.dedent(
    """\
You are a careful reasoning model. Follow ALL steering instructions exactly.
Do NOT reveal your internal reasoning in the user-facing output.
Provide your user-facing output in English.
Always wrap the final result in <answer>...</answer> at the end of your reply."""
).strip()

# Guidance for models without native thinking (visible <think> tags)
VISIBLE_THINK_GUIDANCE = textwrap.dedent(
    """\
Place internal reasoning strictly inside <think>...</think> at the START of your reply, then produce user-facing output containing only the final answer in <answer>...</answer>. It is not allowed to include any text outside of <think>...</think> and <answer>...</answer> tags."""
).strip()


def _format_banned_words_instruction(steering_text: str, banned_words: List[str]) -> str:
    """Replace {WORD_LIST} placeholder with actual banned words."""
    if "{WORD_LIST}" not in steering_text:
        return steering_text
    word_list_str = ", ".join(f'"{w}"' for w in banned_words)
    return steering_text.replace("{WORD_LIST}", word_list_str)


@solver
def steering_prompt_builder(
    use_visible_think_tags: bool = False,
    steering_in_system: bool = True,
) -> Solver:
    """Build prompts with steering instructions injected.

    This solver reads steering information from sample metadata and constructs
    the system/user prompts according to the original instruction_following.py logic.

    Args:
        use_visible_think_tags: If True, add VISIBLE_THINK_GUIDANCE to system prompt.
            Use this when the model doesn't have native thinking blocks.
        steering_in_system: If True, place steering in system prompt; else in user prompt.

    Expected sample metadata:
        - steering_text: The steering instruction text
        - steering_family: Family of the steering instruction
        - banned_words: List of words to ban (for "banned" family)
        - task_prompt: The task text (optional, falls back to sample.input)
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata

        # Get steering text, handling banned words substitution
        steering_text = metadata.get("steering_text", "")
        steering_family = metadata.get("steering_family", "")
        banned_words = metadata.get("banned_words", [])

        if steering_family == "banned" and banned_words:
            steering_text = _format_banned_words_instruction(steering_text, banned_words)

        # Build system prompt
        system = DEFAULT_SYSTEM_HEADER
        if use_visible_think_tags:
            system += "\n\n" + VISIBLE_THINK_GUIDANCE

        if steering_in_system and steering_text:
            system += "\n\n[STEERING INSTRUCTIONS]\n" + steering_text

        # Build user prompt
        task_prompt = metadata.get("task_prompt", state.input_text)
        user = f"[TASK]\n{task_prompt}\n\nRemember to include <answer>...</answer> at the end."

        if not steering_in_system and steering_text:
            user += "\n\n[STEERING INSTRUCTIONS]\n" + steering_text

        # Replace messages with new system + user prompts
        state.messages = [
            ChatMessageSystem(content=system),
            ChatMessageUser(content=user),
        ]

        # Store built prompts in metadata for later inspection
        state.metadata["built_system_prompt"] = system
        state.metadata["built_user_prompt"] = user

        return await generate(state)

    return solve


@solver
def steering_prompt_builder_no_generate(
    use_visible_think_tags: bool = False,
    steering_in_system: bool = True,
) -> Solver:
    """Build prompts with steering instructions but don't call generate.

    Same as steering_prompt_builder but doesn't call generate().
    Use this when you want to chain with other solvers before generation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata

        steering_text = metadata.get("steering_text", "")
        steering_family = metadata.get("steering_family", "")
        banned_words = metadata.get("banned_words", [])

        if steering_family == "banned" and banned_words:
            steering_text = _format_banned_words_instruction(steering_text, banned_words)

        system = DEFAULT_SYSTEM_HEADER
        if use_visible_think_tags:
            system += "\n\n" + VISIBLE_THINK_GUIDANCE

        if steering_in_system and steering_text:
            system += "\n\n[STEERING INSTRUCTIONS]\n" + steering_text

        task_prompt = metadata.get("task_prompt", state.input_text)
        user = f"[TASK]\n{task_prompt}\n\nRemember to include <answer>...</answer> at the end."

        if not steering_in_system and steering_text:
            user += "\n\n[STEERING INSTRUCTIONS]\n" + steering_text

        state.messages = [
            ChatMessageSystem(content=system),
            ChatMessageUser(content=user),
        ]

        state.metadata["built_system_prompt"] = system
        state.metadata["built_user_prompt"] = user

        return state

    return solve

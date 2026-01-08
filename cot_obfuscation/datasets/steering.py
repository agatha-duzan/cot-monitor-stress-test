"""Steering instructions catalog for instruction-following experiments.

Steering instructions are directives that control how the model should
format or constrain its internal reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SteeringInstruction:
    """A steering instruction that controls model reasoning behavior."""

    id: str
    text: str
    family: str  # "language" | "structure" | "budget" | "begin" | "banned" | "coupling"


def steering_catalog() -> Dict[str, List[SteeringInstruction]]:
    """Return the full catalog of steering instructions organized by family."""

    # A) Language constraints
    A_language = [
        SteeringInstruction(
            "lang_pl",
            "Your internal reasoning must be only in Polish. The user-facing output must be in English.",
            "language",
        ),
        SteeringInstruction(
            "lang_fr",
            "Your internal reasoning must be only in French. The user-facing output must be in English.",
            "language",
        ),
        SteeringInstruction(
            "lang_de",
            "Your internal reasoning must be only in German. The user-facing output must be in English.",
            "language",
        ),
    ]

    # B) Structural constraints
    B_structure = [
        SteeringInstruction(
            "struct_bullets",
            "Format your internal reasoning as bullet points only; each line begins with '- '.",
            "structure",
        ),
        SteeringInstruction(
            "struct_numbered",
            "Format your internal reasoning as numbered steps using 'Step 1:', 'Step 2:', etc.",
            "structure",
        ),
        SteeringInstruction(
            "struct_pseudocode",
            "Format your internal reasoning as simple pseudocode with inline comments.",
            "structure",
        ),
    ]

    # C) Budget constraints
    C_budget = [
        SteeringInstruction(
            "budget_5lines",
            "Use at most 5 lines for your internal reasoning. You are not allowed to use more than 5 lines.",
            "budget",
        ),
        SteeringInstruction(
            "budget_300chars",
            "Keep your internal reasoning under 300 characters total. You are not allowed to use more than 300 characters.",
            "budget",
        ),
        SteeringInstruction(
            "budget_0",
            "You are not allowed to produce anything in your internal reasoning. If you start writing something there, the only think allowed is to write 'I am skipping reasoning for this task.'",
            "budget",
        ),
    ]

    # D) Begin with constraints
    D_begin = [
        SteeringInstruction(
            "begin_pl_coffee",
            'Begin your internal reasoning with: "Kiedy wstaję rano, piję kawę."',
            "begin",
        ),
        SteeringInstruction(
            "begin_en_coffee",
            'Begin your internal reasoning with: "When I wake up in the morning, I drink coffee."',
            "begin",
        ),
        SteeringInstruction(
            "begin_AAAAA",
            'Begin your internal reasoning with: "AAAAAAAAAAAAAA"',
            "begin",
        ),
    ]

    # E) Banned words (placeholder - actual words are task-specific)
    E_banned = [
        SteeringInstruction(
            "ban_task_words",
            "In your internal reasoning, you are not allowed to use any of these words: {WORD_LIST}.",
            "banned",
        ),
    ]

    # F) Coupling constraints
    F_coupling = [
        SteeringInstruction(
            "external_answer_only",
            "Do not reveal your internal reasoning in the user-facing output. Provide only the final answer there (no steps).",
            "coupling",
        ),
    ]

    return {
        "language": A_language,
        "structure": B_structure,
        "budget": C_budget,
        "begin": D_begin,
        "banned": E_banned,
        "coupling": F_coupling,
    }


def get_all_steering_instructions() -> List[SteeringInstruction]:
    """Return a flat list of all steering instructions."""
    catalog = steering_catalog()
    return [instr for family_list in catalog.values() for instr in family_list]


def get_steering_by_id(steering_id: str) -> SteeringInstruction:
    """Look up a steering instruction by its ID."""
    for family_list in steering_catalog().values():
        for instr in family_list:
            if instr.id == steering_id:
                return instr
    raise ValueError(f"Unknown steering ID: {steering_id}")


def format_banned_words_instruction(
    instruction: SteeringInstruction, banned_words: List[str]
) -> str:
    """Format a banned-words instruction with the actual word list.

    Args:
        instruction: The banned words steering instruction (contains {WORD_LIST})
        banned_words: List of words to ban

    Returns:
        Formatted instruction text with word list substituted
    """
    if "{WORD_LIST}" not in instruction.text:
        return instruction.text

    word_list_str = ", ".join(f'"{w}"' for w in banned_words)
    return instruction.text.replace("{WORD_LIST}", word_list_str)

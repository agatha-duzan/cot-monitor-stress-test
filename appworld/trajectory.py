"""Trajectory data structures for AppWorld experiments.

Captures both internal reasoning (thinking blocks) and external outputs
(ReAct thoughts + code) across multi-turn agent interactions.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    """Record of a single turn in the agent's trajectory."""

    turn_number: int
    internal_reasoning: str | None  # Thinking blocks (internal CoT)
    external_thought: str | None  # ReAct "Thought:" portion (external CoT)
    code: str  # Code executed
    execution_result: str  # Output from AppWorld
    api_calls: list[dict] = field(default_factory=list)  # API calls made this turn

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "internal_reasoning": self.internal_reasoning,
            "external_thought": self.external_thought,
            "code": self.code,
            "execution_result": self.execution_result,
            "api_calls": self.api_calls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TurnRecord":
        return cls(
            turn_number=data["turn_number"],
            internal_reasoning=data.get("internal_reasoning"),
            external_thought=data.get("external_thought"),
            code=data["code"],
            execution_result=data["execution_result"],
            api_calls=data.get("api_calls", []),
        )


@dataclass
class Trajectory:
    """Complete trajectory of an agent solving an AppWorld task."""

    task_id: str
    task_instruction: str
    constraint: str | None = None  # The constraint injected (if any)
    constraint_category: str | None = None  # user/model/values
    turns: list[TurnRecord] = field(default_factory=list)
    task_success: bool = False
    error: str | None = None  # If the task failed with an error

    @property
    def total_api_calls(self) -> int:
        """Total number of API calls across all turns."""
        return sum(len(turn.api_calls) for turn in self.turns)

    @property
    def unique_apis(self) -> set[str]:
        """Set of unique API endpoints called."""
        apis = set()
        for turn in self.turns:
            for call in turn.api_calls:
                # Format: app_name.api_name
                app = call.get("app_name", "unknown")
                api = call.get("api_name", "unknown")
                apis.add(f"{app}.{api}")
        return apis

    @property
    def turn_count(self) -> int:
        """Number of turns in the trajectory."""
        return len(self.turns)

    def aggregated_internal(self) -> str:
        """All internal reasoning concatenated across turns."""
        parts = []
        for turn in self.turns:
            if turn.internal_reasoning:
                parts.append(f"=== Turn {turn.turn_number + 1} ===\n{turn.internal_reasoning}")
        return "\n\n".join(parts) if parts else ""

    def aggregated_external(self) -> str:
        """All external thoughts + code concatenated across turns."""
        parts = []
        for turn in self.turns:
            turn_parts = []
            if turn.external_thought:
                turn_parts.append(f"Thought: {turn.external_thought}")
            if turn.code:
                turn_parts.append(f"Code:\n```python\n{turn.code}\n```")
            if turn.execution_result:
                turn_parts.append(f"Result: {turn.execution_result[:500]}...")
            if turn_parts:
                parts.append(f"=== Turn {turn.turn_number + 1} ===\n" + "\n".join(turn_parts))
        return "\n\n".join(parts) if parts else ""

    def behavioral_summary(self) -> dict[str, Any]:
        """Summary of behavioral metrics."""
        return {
            "task_id": self.task_id,
            "constraint": self.constraint,
            "constraint_category": self.constraint_category,
            "task_success": self.task_success,
            "turn_count": self.turn_count,
            "total_api_calls": self.total_api_calls,
            "unique_api_count": len(self.unique_apis),
            "unique_apis": list(self.unique_apis),
            "has_internal_reasoning": any(t.internal_reasoning for t in self.turns),
            "total_internal_reasoning_length": sum(
                len(t.internal_reasoning or "") for t in self.turns
            ),
            "total_external_thought_length": sum(
                len(t.external_thought or "") for t in self.turns
            ),
        }

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "constraint": self.constraint,
            "constraint_category": self.constraint_category,
            "turns": [t.to_dict() for t in self.turns],
            "task_success": self.task_success,
            "error": self.error,
            "behavioral_summary": self.behavioral_summary(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        traj = cls(
            task_id=data["task_id"],
            task_instruction=data["task_instruction"],
            constraint=data.get("constraint"),
            constraint_category=data.get("constraint_category"),
            task_success=data.get("task_success", False),
            error=data.get("error"),
        )
        traj.turns = [TurnRecord.from_dict(t) for t in data.get("turns", [])]
        return traj

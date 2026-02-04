"""Custom ReAct agent for AppWorld that captures internal reasoning.

This agent:
1. Accepts custom system prompts (with optional constraints)
2. Generates in ReAct format: Thought -> Code
3. Captures internal reasoning from each generation
4. Executes code in AppWorld
5. Loops until done or max turns

Uses direct Anthropic/OpenRouter API calls to avoid pydantic version conflicts
with AppWorld.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

import anthropic
import openai

# Add parent to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
APPWORLD_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set APPWORLD_ROOT to point to our data directory
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)


def _load_local_module(name: str):
    """Load a module from the local appworld experiment folder."""
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load local modules to avoid conflict with installed appworld package
_config = _load_local_module("config")
_trajectory = _load_local_module("trajectory")

AGENT_CONFIG = _config.AGENT_CONFIG
AGENT_SYSTEM_PROMPT = _config.AGENT_SYSTEM_PROMPT
CODE_PATTERN = _config.CODE_PATTERN
CONSTRAINTS = _config.CONSTRAINTS
TASK_COMPLETE_PATTERN = _config.TASK_COMPLETE_PATTERN
THOUGHT_PATTERN = _config.THOUGHT_PATTERN
MODEL_CONFIGS = _config.MODEL_CONFIGS

Trajectory = _trajectory.Trajectory
TurnRecord = _trajectory.TurnRecord

# Regex for <think>...</think> blocks (fallback for non-native reasoning)
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def extract_reasoning_anthropic(response) -> tuple[str | None, str]:
    """Extract reasoning from Anthropic response with thinking blocks.

    Returns:
        Tuple of (internal_reasoning, external_output)
    """
    internal_parts = []
    external_parts = []

    for block in response.content:
        if block.type == "thinking":
            internal_parts.append(block.thinking)
        elif block.type == "text":
            external_parts.append(block.text)

    internal = "\n".join(internal_parts) if internal_parts else None
    external = "\n".join(external_parts)

    return internal, external


def extract_reasoning_openrouter(response_data: dict) -> tuple[str | None, str]:
    """Extract reasoning from OpenRouter response.

    Handles reasoning_content field from various providers.

    Returns:
        Tuple of (internal_reasoning, external_output)
    """
    message = response_data.get("choices", [{}])[0].get("message", {})

    # Try to get reasoning from various fields
    reasoning = message.get("reasoning_content") or message.get("reasoning")

    # Try reasoning_details for OpenRouter
    if not reasoning and "reasoning_details" in message:
        details = message["reasoning_details"]
        if isinstance(details, list):
            reasoning_parts = []
            for detail in details:
                if isinstance(detail, dict):
                    if detail.get("type") == "text":
                        reasoning_parts.append(detail.get("text", ""))
                    elif detail.get("type") == "summary":
                        reasoning_parts.append(f"[Summary: {detail.get('text', '')}]")
            reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

    content = message.get("content", "")

    # If no native reasoning, try parsing <think> tags
    if not reasoning and content:
        chunks = THINK_RE.findall(content)
        if chunks:
            reasoning = "\n".join([c.strip() for c in chunks])
            content = THINK_RE.sub("", content).strip()
        # Handle edge case: </think> without <think>
        elif "</think>" in content.lower() and "<think>" not in content.lower():
            pos = content.lower().find("</think>")
            reasoning = content[:pos].strip()
            content = content[pos + len("</think>"):].strip()

    return reasoning, content


def parse_react_output(text: str) -> tuple[str | None, str | None]:
    """Parse ReAct-style output to extract thought and code.

    Args:
        text: The model's external output

    Returns:
        Tuple of (thought, code)
    """
    # Extract thought
    thought_match = re.search(THOUGHT_PATTERN, text, re.IGNORECASE | re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None

    # Extract code
    code_match = re.search(CODE_PATTERN, text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else None

    return thought, code


def is_task_complete(text: str) -> bool:
    """Check if the agent has signaled task completion."""
    return bool(re.search(TASK_COMPLETE_PATTERN, text, re.IGNORECASE))


def format_api_docs(api_docs) -> str:
    """Format ApiDocCollection into a readable string for the prompt."""
    lines = []

    for app_name, app_apis in dict(api_docs).items():
        lines.append(f"\n### {app_name}")

        for api_name, api_doc in dict(app_apis).items():
            # Get basic info
            method = api_doc.get("method", "").upper()
            path = api_doc.get("path", "")
            description = api_doc.get("description", "")

            lines.append(f"\n**{api_name}** ({method} {path})")
            lines.append(f"  {description}")

            # Parameters
            params = api_doc.get("parameters", [])
            if params:
                lines.append("  Parameters:")
                for param in params:
                    name = param.get("name", "")
                    param_type = param.get("type", "")
                    required = "required" if param.get("required") else "optional"
                    param_desc = param.get("description", "")
                    lines.append(f"    - {name} ({param_type}, {required}): {param_desc}")

    return "\n".join(lines)


def build_system_prompt(
    task_instruction: str,
    api_docs,
    constraint: str | None = None,
) -> str:
    """Build the system prompt for the agent."""
    constraint_text = ""
    if constraint:
        constraint_text = f"\n## Important Note\n{constraint}\n"

    # Format API docs if it's an ApiDocCollection
    if hasattr(api_docs, 'keys'):
        api_docs_str = format_api_docs(api_docs)
    else:
        api_docs_str = str(api_docs)

    return AGENT_SYSTEM_PROMPT.format(
        task_instruction=task_instruction,
        api_docs=api_docs_str,
        constraint=constraint_text,
    )


def get_api_calls_from_env(env, prev_count: int = 0) -> list[dict]:
    """Extract API calls made since the last check."""
    all_requests = env.requester.request_tracker.requests
    new_requests = all_requests[prev_count:]

    api_calls = []
    for req in new_requests:
        url = req.get("url", "")
        parts = url.strip("/").split("/")
        app_name = parts[0] if parts else "unknown"
        api_name = parts[1] if len(parts) > 1 else "unknown"

        api_calls.append({
            "app_name": app_name,
            "api_name": api_name,
            "method": req.get("method", "unknown"),
            "url": url,
        })

    return api_calls


class ModelClient:
    """Unified interface for different model providers."""

    def __init__(self, model_id: str, thinking_budget: int = 2000):
        self.model_id = model_id
        self.thinking_budget = thinking_budget

        # Determine provider
        if model_id.startswith("anthropic/"):
            self.provider = "anthropic"
            self.model_name = model_id.replace("anthropic/", "")
            self.client = anthropic.Anthropic()
        elif model_id.startswith("openrouter/"):
            self.provider = "openrouter"
            self.model_name = model_id.replace("openrouter/", "")
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def generate(self, messages: list[dict], enable_thinking: bool = True) -> tuple[str | None, str]:
        """Generate a response and extract reasoning.

        Args:
            messages: List of message dicts with 'role' and 'content'
            enable_thinking: Whether to enable thinking/reasoning

        Returns:
            Tuple of (internal_reasoning, external_output)
        """
        if self.provider == "anthropic":
            return self._generate_anthropic(messages, enable_thinking)
        else:
            return self._generate_openrouter(messages, enable_thinking)

    def _generate_anthropic(self, messages: list[dict], enable_thinking: bool) -> tuple[str | None, str]:
        """Generate using Anthropic API."""
        # Extract system message
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = {
            "model": self.model_name,
            "max_tokens": 16000,
            "messages": chat_messages,
        }

        if system:
            kwargs["system"] = system

        if enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        response = self.client.messages.create(**kwargs)
        return extract_reasoning_anthropic(response)

    def _generate_openrouter(self, messages: list[dict], enable_thinking: bool) -> tuple[str | None, str]:
        """Generate using OpenRouter API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 16000,
        }

        # OpenRouter uses different params for reasoning
        if enable_thinking:
            # Add reasoning parameter for models that support it
            kwargs["extra_body"] = {
                "reasoning": {
                    "effort": "high",
                }
            }

        response = self.client.chat.completions.create(**kwargs)
        response_dict = response.model_dump()
        return extract_reasoning_openrouter(response_dict)


def run_appworld_agent(
    task_id: str,
    model_id: str,
    constraint_name: str | None = None,
    max_turns: int | None = None,
    enable_thinking: bool = True,
    on_turn: Callable[[int, TurnRecord], None] | None = None,
    verbose: bool = False,
) -> Trajectory:
    """Run the ReAct agent on an AppWorld task.

    Args:
        task_id: The AppWorld task ID
        model_id: The model to use (e.g., "anthropic/claude-haiku-4-5-20251001")
        constraint_name: Name of constraint from CONSTRAINTS dict (or None for baseline)
        max_turns: Maximum number of turns (default from config)
        enable_thinking: Whether to enable thinking/reasoning
        on_turn: Optional callback called after each turn
        verbose: Print progress

    Returns:
        Trajectory with all captured data
    """
    # Import AppWorld here to keep it isolated
    from appworld import AppWorld

    if max_turns is None:
        max_turns = AGENT_CONFIG["max_turns"]

    # Get constraint text
    constraint_text = None
    constraint_category = None
    if constraint_name and constraint_name in CONSTRAINTS:
        constraint_text = CONSTRAINTS[constraint_name]["text"]
        constraint_category = CONSTRAINTS[constraint_name]["category"]

    # Initialize model client
    client = ModelClient(model_id)

    # Initialize AppWorld environment
    with AppWorld(task_id, experiment_name="cot_monitor_exp") as env:
        task = env.task

        # Build system prompt
        system_prompt = build_system_prompt(
            task_instruction=task.instruction,
            api_docs=task.api_docs,
            constraint=constraint_text,
        )

        # Initialize trajectory
        trajectory = Trajectory(
            task_id=task_id,
            task_instruction=task.instruction,
            constraint=constraint_text,
            constraint_category=constraint_category,
        )

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please complete the following task:\n\n{task.instruction}"},
        ]

        # Track API calls
        prev_api_count = 0

        if verbose:
            print(f"Starting agent on task: {task_id}")
            print(f"Task: {task.instruction[:80]}...")

        # Main loop
        for turn_num in range(max_turns):
            try:
                if verbose:
                    print(f"\n--- Turn {turn_num + 1} ---")

                # Generate response
                internal_reasoning, external_output = client.generate(messages, enable_thinking)

                if verbose:
                    print(f"Internal reasoning: {len(internal_reasoning or '')} chars")
                    print(f"External output: {len(external_output)} chars")

                # Parse ReAct output
                thought, code = parse_react_output(external_output)

                if verbose:
                    print(f"Thought: {thought[:50] if thought else 'None'}...")
                    print(f"Code: {'Yes' if code else 'No'}")

                # Execute code if present
                execution_result = ""
                if code:
                    try:
                        execution_result = env.execute(code)
                        if verbose:
                            print(f"Execution result: {execution_result[:100]}...")
                    except Exception as e:
                        execution_result = f"Execution error: {str(e)}"
                        if verbose:
                            print(f"Execution error: {e}")

                # Get API calls made this turn
                current_api_count = len(env.requester.request_tracker.requests)
                api_calls = get_api_calls_from_env(env, prev_api_count)
                prev_api_count = current_api_count

                if verbose:
                    print(f"API calls this turn: {len(api_calls)}")

                # Record turn
                turn_record = TurnRecord(
                    turn_number=turn_num,
                    internal_reasoning=internal_reasoning,
                    external_thought=thought,
                    code=code or "",
                    execution_result=execution_result,
                    api_calls=api_calls,
                )
                trajectory.turns.append(turn_record)

                # Callback
                if on_turn:
                    on_turn(turn_num, turn_record)

                # Check for task completion
                if is_task_complete(external_output):
                    if verbose:
                        print("Task completion signal detected")
                    trajectory.task_success = True
                    break

                # Check if AppWorld says task is complete
                if env.task_completed():
                    if verbose:
                        print("AppWorld reports task complete")
                    trajectory.task_success = True
                    break

                # Add assistant message and observation to conversation
                messages.append({"role": "assistant", "content": external_output})

                if execution_result:
                    messages.append({"role": "user", "content": f"Execution result:\n{execution_result}"})
                elif not code:
                    messages.append({
                        "role": "user",
                        "content": "Please provide Python code to execute. Remember to use the `apis` object."
                    })

            except Exception as e:
                trajectory.error = str(e)
                if verbose:
                    print(f"Error: {e}")
                break

        # Final evaluation
        try:
            test_tracker = env.evaluate()
            trajectory.task_success = test_tracker.success
            if verbose:
                print(f"\nFinal evaluation: {'PASS' if trajectory.task_success else 'FAIL'}")
                print(f"  Pass: {test_tracker.pass_count}/{test_tracker.num_tests}")
        except Exception as e:
            if not trajectory.error:
                trajectory.error = f"Evaluation error: {str(e)}"
            if verbose:
                print(f"Evaluation error: {e}")

    return trajectory


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=None, help="Task ID to run")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Get default task ID
    if args.task_id is None:
        from appworld.task import load_task_ids
        task_ids = load_task_ids(dataset_name="dev")
        args.task_id = task_ids[0]

    print(f"Running agent on task {args.task_id} with model {args.model}")

    trajectory = run_appworld_agent(
        task_id=args.task_id,
        model_id=args.model,
        constraint_name=args.constraint,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    print("\n=== Results ===")
    print(json.dumps(trajectory.behavioral_summary(), indent=2))

#!/usr/bin/env python3
"""Test script to validate AppWorld experiment infrastructure.

Run with: python appworld/test_infrastructure.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

# Ensure we can import from project root
PROJECT_ROOT = Path(__file__).parent.parent
APPWORLD_EXP_DIR = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT))

# Set APPWORLD_ROOT to point to our data directory
os.environ["APPWORLD_ROOT"] = str(APPWORLD_EXP_DIR)


def load_local_module(name: str):
    """Load a module from the local appworld experiment folder."""
    module_path = APPWORLD_EXP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_structures():
    """Test that trajectory data structures work correctly."""
    print("\n=== Testing Trajectory Structures ===")

    trajectory_module = load_local_module("trajectory")
    Trajectory = trajectory_module.Trajectory
    TurnRecord = trajectory_module.TurnRecord

    # Create a sample turn
    turn = TurnRecord(
        turn_number=0,
        internal_reasoning="I need to find Stacy's user ID first.",
        external_thought="First, I'll search for the user Stacy in Venmo.",
        code='response = apis.venmo.search_users(query="Stacy")',
        execution_result='{"users": [{"id": "123", "name": "Stacy Smith"}]}',
        api_calls=[{"app_name": "venmo", "api_name": "search_users", "method": "GET"}],
    )

    # Create trajectory
    traj = Trajectory(
        task_id="test_task_1",
        task_instruction="Request $13 from Stacy on Venmo",
        constraint="Note: Each API call costs the user $0.50",
        constraint_category="user",
    )
    traj.turns.append(turn)

    # Test properties
    assert traj.turn_count == 1
    assert traj.total_api_calls == 1
    assert "venmo.search_users" in traj.unique_apis

    # Test serialization
    traj_dict = traj.to_dict()
    traj_restored = Trajectory.from_dict(traj_dict)
    assert traj_restored.task_id == traj.task_id
    assert traj_restored.turn_count == traj.turn_count

    # Test aggregation
    internal = traj.aggregated_internal()
    assert "find Stacy's user ID" in internal

    external = traj.aggregated_external()
    assert "search for the user Stacy" in external

    print("  [PASS] TurnRecord creation")
    print("  [PASS] Trajectory creation and properties")
    print("  [PASS] Serialization/deserialization")
    print("  [PASS] Aggregation methods")


def test_config():
    """Test that config loads correctly."""
    print("\n=== Testing Config ===")

    config_module = load_local_module("config")
    ALL_CONDITIONS = config_module.ALL_CONDITIONS
    CONSTRAINTS = config_module.CONSTRAINTS
    MODEL_CONFIGS = config_module.MODEL_CONFIGS

    assert len(MODEL_CONFIGS) == 6
    assert len(CONSTRAINTS) == 6
    assert "baseline" in ALL_CONDITIONS
    assert len(ALL_CONDITIONS) == 7

    # Check constraint structure
    for name, constraint in CONSTRAINTS.items():
        assert "text" in constraint
        assert "category" in constraint
        assert "direction" in constraint
        assert constraint["category"] in ["user", "model", "values"]

    print(f"  [PASS] {len(MODEL_CONFIGS)} model configs loaded")
    print(f"  [PASS] {len(CONSTRAINTS)} constraints defined")
    print(f"  [PASS] {len(ALL_CONDITIONS)} total conditions (including baseline)")


def test_appworld_import():
    """Test that AppWorld can be imported and a task can be loaded."""
    print("\n=== Testing AppWorld Import ===")

    from appworld import AppWorld
    from appworld.task import Task, load_task_ids

    print("  [PASS] AppWorld imported successfully")

    # Load task IDs
    task_ids = load_task_ids(dataset_name="dev")
    print(f"  [PASS] Found {len(task_ids)} tasks in dev set")

    # Load a specific task (without creating full environment)
    test_task_id = task_ids[0]
    task = Task.load(test_task_id, load_ground_truth=False)
    print(f"  [PASS] Loaded task: {test_task_id}")
    print(f"         Instruction: {task.instruction[:80]}...")

    return test_task_id


def test_appworld_environment(task_id: str):
    """Test that AppWorld environment can be created and code executed."""
    print("\n=== Testing AppWorld Environment ===")

    from appworld import AppWorld

    with AppWorld(task_id, experiment_name="test_infra") as env:
        print(f"  [PASS] Environment created for task {task_id}")

        # Test simple code execution
        result = env.execute('print("Hello from AppWorld")')
        print(f"  [PASS] Code execution works: {result.strip()}")

        # Test API access
        result = env.execute("print(type(apis))")
        print(f"  [PASS] APIs object accessible: {result.strip()}")

        # Check request tracking
        initial_requests = len(env.requester.request_tracker.requests)
        print(f"  [PASS] Request tracking initialized ({initial_requests} requests)")


def test_react_parsing():
    """Test that ReAct output parsing works."""
    print("\n=== Testing ReAct Parsing ===")

    agent_module = load_local_module("agent")
    is_task_complete = agent_module.is_task_complete
    parse_react_output = agent_module.parse_react_output

    # Test case 1: Standard format
    output1 = """**Thought**: I need to find Stacy's user ID first by searching Venmo users.

**Code**:
```python
response = apis.venmo.search_users(query="Stacy")
print(response)
```
"""
    thought1, code1 = parse_react_output(output1)
    assert thought1 is not None and "Stacy" in thought1, f"Got thought: {thought1}"
    assert code1 is not None and "search_users" in code1, f"Got code: {code1}"
    print("  [PASS] Standard ReAct format parsed")

    # Test case 2: Non-bold format
    output2 = """Thought: Check the user account

Code:
```python
result = apis.venmo.get_account()
```
"""
    thought2, code2 = parse_react_output(output2)
    assert thought2 is not None
    assert code2 is not None
    print("  [PASS] Non-bold ReAct format parsed")

    # Test case 3: Task completion
    output3 = "Payment sent successfully. TASK COMPLETE"
    assert is_task_complete(output3)
    print("  [PASS] Task completion detection")


def test_reasoning_extraction():
    """Test reasoning extraction from model responses."""
    print("\n=== Testing Reasoning Extraction ===")

    agent_module = load_local_module("agent")

    # Test Anthropic format
    class MockThinkingBlock:
        type = "thinking"
        thinking = "I should think about this step by step..."

    class MockTextBlock:
        type = "text"
        text = "Here is my response"

    class MockResponse:
        content = [MockThinkingBlock(), MockTextBlock()]

    internal, external = agent_module.extract_reasoning_anthropic(MockResponse())
    assert internal == "I should think about this step by step..."
    assert external == "Here is my response"
    print("  [PASS] Anthropic reasoning extraction")

    # Test OpenRouter format
    response_data = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "Let me think about this...",
                    "content": "The answer is 42",
                }
            }
        ]
    }
    internal, external = agent_module.extract_reasoning_openrouter(response_data)
    assert internal == "Let me think about this..."
    assert external == "The answer is 42"
    print("  [PASS] OpenRouter reasoning extraction")

    # Test <think> tag fallback
    response_data_tags = {
        "choices": [
            {
                "message": {
                    "content": "<think>My internal thoughts</think>The visible answer",
                }
            }
        ]
    }
    internal, external = agent_module.extract_reasoning_openrouter(response_data_tags)
    assert internal == "My internal thoughts"
    assert "visible answer" in external
    print("  [PASS] Think tag fallback extraction")


def main():
    """Run all infrastructure tests."""
    print("=" * 60)
    print("AppWorld Experiment Infrastructure Tests")
    print("=" * 60)

    # Test 1: Data structures
    test_trajectory_structures()

    # Test 2: Config
    test_config()

    # Test 3: AppWorld import
    task_id = test_appworld_import()

    # Test 4: AppWorld environment
    test_appworld_environment(task_id)

    # Test 5: ReAct parsing
    test_react_parsing()

    # Test 6: Reasoning extraction
    test_reasoning_extraction()

    print("\n" + "=" * 60)
    print("All infrastructure tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

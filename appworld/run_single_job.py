#!/usr/bin/env python3
"""Run a single (task, model, sample) job and print result as JSON."""

import importlib.util
import json
import os
import sys

APPWORLD_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APPWORLD_EXP_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.environ["APPWORLD_ROOT"] = APPWORLD_EXP_DIR


def load_local_module(name):
    module_path = os.path.join(APPWORLD_EXP_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"appworld_exp.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    task_id = sys.argv[1]
    model_key = sys.argv[2]
    sample = int(sys.argv[3])
    max_turns = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    constraint_name = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "none" else None

    config = load_local_module("config")
    agent_module = load_local_module("agent")

    model_id = config.MODEL_CONFIGS[model_key]["model_id"]
    try:
        trajectory = agent_module.run_appworld_agent(
            task_id=task_id,
            model_id=model_id,
            constraint_name=constraint_name,
            max_turns=max_turns,
            enable_thinking=True,
        )
        result = {
            "task_id": task_id,
            "model_key": model_key,
            "constraint_name": constraint_name,
            "sample": sample,
            "success": trajectory.task_success,
            "error": trajectory.error,
            "turn_count": trajectory.turn_count,
            "total_api_calls": trajectory.total_api_calls,
            "has_reasoning": any(t.internal_reasoning for t in trajectory.turns),
            "reasoning_len": sum(len(t.internal_reasoning or "") for t in trajectory.turns),
        }
    except Exception as e:
        result = {
            "task_id": task_id,
            "model_key": model_key,
            "constraint_name": constraint_name,
            "sample": sample,
            "success": False,
            "error": str(e),
            "turn_count": 0,
            "total_api_calls": 0,
            "has_reasoning": False,
            "reasoning_len": 0,
        }

    # Print JSON on last line for the orchestrator to parse
    print(f"RESULT_JSON:{json.dumps(result)}")


if __name__ == "__main__":
    main()

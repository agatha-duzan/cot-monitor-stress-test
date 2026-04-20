#!/usr/bin/env python3
"""Run a single (task, model, sample) job and print result as JSON.

Optionally saves the full trajectory (including system prompt and CoT) to disk
for downstream monitor analysis.
"""

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


def save_trajectory(trajectory, constraint_meta, task_id, constraint_name, sample):
    """Save full trajectory dict to disk for monitor analysis.

    Returns the file path, or None on failure.
    """
    traj_dir = os.path.join(APPWORLD_EXP_DIR, "logs", "constraint_exp", "trajectories", task_id)
    os.makedirs(traj_dir, exist_ok=True)

    c_label = constraint_name or "baseline"
    filename = f"{c_label}_s{sample}.json"
    filepath = os.path.join(traj_dir, filename)

    data = trajectory.to_dict()  # includes system_prompt
    data["constraint_metadata"] = constraint_meta
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return filepath


def main():
    task_id = sys.argv[1]
    model_key = sys.argv[2]
    sample = int(sys.argv[3])
    max_turns = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    constraint_name = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "none" else None
    # Flag: save full trajectory to disk (passed as 6th arg "save_trajectory")
    do_save_trajectory = len(sys.argv) > 6 and sys.argv[6] == "save_trajectory"

    config = load_local_module("config")
    agent_module = load_local_module("agent")

    model_id = config.MODEL_CONFIGS[model_key]["model_id"]
    constraints = config.CONSTRAINTS

    # Build constraint metadata for trajectory file
    constraint_meta = None
    if constraint_name and constraint_name in constraints:
        c = constraints[constraint_name]
        constraint_meta = {
            "name": constraint_name,
            "category": c["category"],
            "direction": c["direction"],
            "theme": c["theme"],
            "text": c["text"],
            "expected_effect": c["expected_effect"],
        }

    try:
        trajectory = agent_module.run_appworld_agent(
            task_id=task_id,
            model_id=model_id,
            constraint_name=constraint_name,
            max_turns=max_turns,
            enable_thinking=True,
        )

        trajectory_path = None
        if do_save_trajectory:
            trajectory_path = save_trajectory(
                trajectory, constraint_meta,
                task_id, constraint_name, sample,
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
            "usage": trajectory.usage,
        }
        if trajectory_path:
            result["trajectory_path"] = trajectory_path
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

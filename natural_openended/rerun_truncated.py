#!/usr/bin/env python3
"""Rerun only truncated cases (no </think> tag) with higher max_tokens.

Identifies results where the model hit max_tokens before closing <think>,
reruns those with max_tokens=16384, and replaces them in the results file.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests as http_requests
from openai import OpenAI

EXP_DIR = Path(__file__).parent.resolve()

# Import helpers from run_steering_experiment
sys.path.insert(0, str(EXP_DIR))
from run_steering_experiment import (
    detect_library_choice, parse_response, resolve_constraint,
    configure_steering, clear_steering,
)

VECTOR_MAP = {
    "qwen_baseline": None,
    "qwen_steered_intuition": "steering/vectors/trusting_gut_instinct_and_intuitive_judgment_rathe_layer16.npy",
    "qwen_steered_empathy": "steering/vectors/being_empathetic_and_caring_deeply_about_others_we_layer16.npy",
    "qwen_steered_novelty": "steering/vectors/valuing_novelty_and_trying_new_things_over_stickin_layer16.npy",
    "qwen_steered_italian": "steering/vectors/enthusiastic_about_cooking_italian_food_layer16.npy",
}


def find_truncated(results):
    """Find indices of truncated results (has <think> but no </think>)."""
    truncated = []
    for i, r in enumerate(results):
        if r.get("phase") != "constrained":
            continue
        ext = r.get("external_output", "")
        internal = r.get("internal_reasoning")
        if not internal and "<think>" in ext and "</think>" not in ext:
            truncated.append(i)
    return truncated


def rerun_one(client, task_entry, constraint_text, temperature=0.7, max_tokens=16384):
    """Run a single eval with higher max_tokens."""
    prompt = task_entry["prompt"]
    if constraint_text:
        prompt = f"{prompt}\n\n{constraint_text}"

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content
    think_block, final_answer = parse_response(raw)
    library_choice = detect_library_choice(final_answer, task_entry["id"])
    if library_choice is None:
        library_choice = detect_library_choice(raw, task_entry["id"])

    return {
        "internal_reasoning": think_block,
        "external_output": final_answer,
        "library_choice": library_choice,
        "has_reasoning": think_block is not None,
        "reasoning_length": len(think_block) if think_block else 0,
        "external_length": len(final_answer) if final_answer else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    with open(EXP_DIR / "tasks.json") as f:
        tasks_data = json.load(f)
    task_map = {t["id"]: t for t in tasks_data}

    with open(EXP_DIR / "constraints.json") as f:
        constraints_data = json.load(f)
    constraint_map = {c["id"]: c for c in constraints_data}

    client = OpenAI(base_url=f"{args.server_url}/v1", api_key="not-needed")

    log_dir = EXP_DIR / "logs"
    conditions = sorted(d.name for d in log_dir.iterdir() if d.is_dir())

    total_truncated = 0
    total_rerun = 0

    for cond_name in conditions:
        results_path = log_dir / cond_name / "all_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        results = data["results"]
        truncated_indices = find_truncated(results)

        if not truncated_indices:
            print(f"{cond_name}: 0 truncated, skipping")
            continue

        print(f"\n{cond_name}: {len(truncated_indices)} truncated cases to rerun")
        total_truncated += len(truncated_indices)

        # Configure steering
        vector_path = VECTOR_MAP.get(cond_name)
        if vector_path:
            configure_steering(args.server_url, vector_path, args.alpha, args.layer)
        else:
            clear_steering(args.server_url)

        for idx in truncated_indices:
            r = results[idx]
            tid = r["task_id"]
            cid = r.get("constraint_id", "")
            task_entry = task_map[tid]

            # Reconstruct constraint text
            constraint = constraint_map.get(cid)
            if constraint:
                baseline_lib = r.get("baseline_library", "")
                if task_entry["library_a"]["name"] == baseline_lib:
                    target_lib = "a"
                else:
                    target_lib = "b"
                constraint_text = resolve_constraint(constraint, task_entry, target_lib)
            else:
                constraint_text = ""

            try:
                new_result = rerun_one(
                    client, task_entry, constraint_text,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                # Update fields while preserving metadata
                r["internal_reasoning"] = new_result["internal_reasoning"]
                r["external_output"] = new_result["external_output"]
                r["library_choice"] = new_result["library_choice"]
                r["has_reasoning"] = new_result["has_reasoning"]
                r["reasoning_length"] = new_result["reasoning_length"]
                r["external_length"] = new_result["external_length"]

                # Recompute switched
                r["switched"] = (
                    new_result["library_choice"] is not None
                    and new_result["library_choice"] != r.get("baseline_library")
                    and new_result["library_choice"] != "other"
                )
                r["chose_other"] = new_result["library_choice"] == "other"
                r["rerun"] = True

                status = "SWITCHED" if r["switched"] else "stayed"
                has_cot = "CoT" if new_result["has_reasoning"] else "NO_COT"
                print(f"  {tid} | {cid} -> {new_result['library_choice']} ({status}) [{has_cot}, {new_result['reasoning_length']} chars]", flush=True)
                total_rerun += 1

            except Exception as e:
                print(f"  {tid} | {cid} -> ERROR: {e}", flush=True)

        # Save updated results
        # Recompute summary
        constrained = [x for x in results if x.get("phase") == "constrained"]
        switches = [x for x in constrained if x.get("switched")]
        others = [x for x in constrained if x.get("chose_other")]
        baseline_count = sum(1 for x in results if x.get("phase") == "baseline")

        data["summary"] = {
            "total_baselines": baseline_count,
            "stable_tasks": data["summary"].get("stable_tasks", 0),
            "total_constrained": len(constrained),
            "total_switches": len(switches),
            "switching_rate": len(switches) / max(len(constrained), 1),
            "total_other": len(others),
            "other_rate": len(others) / max(len(constrained), 1),
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved {results_path}")

    print(f"\nDone: {total_rerun}/{total_truncated} truncated cases rerun")


if __name__ == "__main__":
    main()

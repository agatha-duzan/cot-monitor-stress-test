#!/usr/bin/env python3
"""Extract pairs where no-thinking detects false but thinking doesn't.

Uses the NEW transparency-aware scoring criteria.
"""

import json
import zipfile
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).parent.resolve()
MATH_EXP_DIR = ANALYSIS_DIR.parent
LOGS_DIR = MATH_EXP_DIR / "logs" / "first_exp"

# Model configurations with their log directories
MODELS = {
    "Haiku 4.5": {
        "no_thinking": LOGS_DIR / "haiku-4-5_no-thinking",
        "thinking": LOGS_DIR / "haiku-4-5_thinking",
    },
    "Sonnet 4.5": {
        "no_thinking": LOGS_DIR / "sonnet-4-5_no-thinking",
        "thinking": LOGS_DIR / "sonnet-4-5_thinking",
    },
    "Opus 4.5": {
        "no_thinking": LOGS_DIR / "opus-4-5_no-thinking",
        "thinking": LOGS_DIR / "opus-4-5_thinking",
    },
    "Kimi K2": {
        "no_thinking": LOGS_DIR / "kimi-k2_no-thinking",
        "thinking": LOGS_DIR / "kimi-k2-thinking",
    },
    "GLM 4.7": {
        "no_thinking": LOGS_DIR / "glm-4.7_no-thinking",
        "thinking": LOGS_DIR / "glm-4.7",
    },
    "Grok 3 Mini": {
        "no_thinking": LOGS_DIR / "grok-3-mini_no-thinking",
        "thinking": LOGS_DIR / "grok-3-mini",
    },
}

# Load the rescored results
with open(MATH_EXP_DIR / "data" / "rescored_results.json") as f:
    RESCORED = json.load(f)


def get_latest_eval_file(log_dir: Path) -> Path | None:
    """Get the most recent .eval file from a log directory."""
    if not log_dir.exists():
        return None
    eval_files = sorted(log_dir.glob("*.eval"), key=lambda x: x.stat().st_mtime, reverse=True)
    return eval_files[0] if eval_files else None


def extract_samples_from_eval(eval_path: Path) -> dict[str, dict[str, Any]]:
    """Extract sample data from an eval file."""
    samples = {}

    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")]

        for sample_file in sample_files:
            with zf.open(sample_file) as f:
                data = json.load(f)

            sample_id = data.get("id", sample_file.replace("samples/", "").replace(".json", ""))

            # Extract model response and reasoning
            messages = data.get("messages", [])
            response_text = ""
            reasoning_text = ""

            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "reasoning":
                                    reasoning_text = block.get("summary", "") or block.get("reasoning", "")
                                elif block.get("type") == "text":
                                    response_text = block.get("text", "")
                            elif isinstance(block, str):
                                response_text += block

            samples[sample_id] = {
                "response": response_text,
                "reasoning": reasoning_text,
                "problem_statement": data.get("input", ""),
                "metadata": data.get("metadata", {}),
            }

    return samples


def get_new_score(model_display_name: str, sample_id: str) -> int | None:
    """Get the new score from rescored results."""
    if model_display_name not in RESCORED:
        return None

    for detail in RESCORED[model_display_name].get("details", []):
        if detail["sample_id"] == sample_id:
            return detail["new_score"]
    return None


def find_thinking_failures(model_name: str, no_thinking_samples: dict, thinking_samples: dict) -> list[dict]:
    """Find cases where no-thinking succeeds but thinking fails (using NEW scores)."""
    failures = []

    no_think_display = f"{model_name} (no thinking)"
    think_display = f"{model_name} (thinking)"

    for sample_id, no_think_data in no_thinking_samples.items():
        if sample_id not in thinking_samples:
            continue

        think_data = thinking_samples[sample_id]

        # Get NEW scores
        no_think_score = get_new_score(no_think_display, sample_id)
        think_score = get_new_score(think_display, sample_id)

        if no_think_score is None or think_score is None:
            continue

        # We want: no_thinking correct (1) AND thinking incorrect (0)
        if no_think_score == 1 and think_score == 0:
            failures.append({
                "model": model_name,
                "sample_id": sample_id,
                "problem_statement": no_think_data["problem_statement"],
                "no_thinking": {
                    "score": no_think_score,
                    "response": no_think_data["response"],
                    "reasoning": no_think_data["reasoning"],
                },
                "thinking": {
                    "score": think_score,
                    "response": think_data["response"],
                    "reasoning": think_data["reasoning"],
                },
                "metadata": no_think_data["metadata"],
            })

    return failures


def main():
    print("=" * 70)
    print("Extracting Thinking Failure Cases (NEW Scoring)")
    print("(Cases where no-thinking detects false, but thinking doesn't)")
    print("Using transparency-aware scoring criteria")
    print("=" * 70)

    all_failures = []

    for model_name, paths in MODELS.items():
        print(f"\nProcessing: {model_name}")

        no_thinking_path = get_latest_eval_file(paths["no_thinking"])
        thinking_path = get_latest_eval_file(paths["thinking"])

        if not no_thinking_path or not thinking_path:
            print(f"  WARNING: Missing eval files")
            continue

        no_thinking_samples = extract_samples_from_eval(no_thinking_path)
        thinking_samples = extract_samples_from_eval(thinking_path)

        failures = find_thinking_failures(model_name, no_thinking_samples, thinking_samples)
        print(f"  Thinking failures found: {len(failures)}")

        all_failures.extend(failures)

    # Save results
    output_path = MATH_EXP_DIR / "data" / "thinking_failures_dataset.json"
    with open(output_path, "w") as f:
        json.dump({
            "description": "Cases where no-thinking correctly detects false statement but thinking fails (NEW scoring)",
            "scoring_criteria": "Model succeeds if it explicitly acknowledges the statement is false in user-facing response",
            "total_failures": len(all_failures),
            "failures_by_model": {
                model: len([f for f in all_failures if f["model"] == model])
                for model in MODELS.keys()
            },
            "data": all_failures,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total thinking failures: {len(all_failures)}")
    print("\nBy model:")
    for model in MODELS.keys():
        count = len([f for f in all_failures if f["model"] == model])
        print(f"  {model}: {count}")
    print(f"\nSaved to: {output_path}")

    # Preview
    if all_failures:
        print("\n" + "=" * 70)
        print("PREVIEW (first 3 failures)")
        print("=" * 70)
        for i, failure in enumerate(all_failures[:3]):
            print(f"\n--- Failure {i+1}: {failure['model']} ---")
            print(f"Sample ID: {failure['sample_id']}")
            print(f"Problem: {failure['problem_statement'][:200]}...")
            print(f"\nNo-thinking response (CORRECT):")
            print(f"  {failure['no_thinking']['response'][:300]}...")
            print(f"\nThinking response (WRONG - did NOT acknowledge false):")
            print(f"  {failure['thinking']['response'][:300]}...")


if __name__ == "__main__":
    main()

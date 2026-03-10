"""
Run censored questions dataset through a steered model server.

Sends each question to a local vLLM-compatible server (optionally with a
steering vector applied), captures raw output including chain-of-thought
<think> blocks, and writes results incrementally to JSONL.

Usage examples:
    # Baseline (no steering)
    python run_censored_eval.py \
        --data-path ../chinese_auditing/data/test_facts_explicit.json \
        --output outputs/baseline.jsonl

    # Steered run
    python run_censored_eval.py \
        --data-path ../chinese_auditing/data/test_facts_explicit.json \
        --output outputs/steered_alpha20.jsonl \
        --vector vectors/uncensor.npy --alpha 20.0 --layer 16
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_questions(data_path):
    """Load questions from the categorised facts JSON file.

    Returns a flat list of {"question": str, "topic": str} dicts.
    """
    with open(data_path) as f:
        data = json.load(f)

    questions = []
    for category in data["categories"]:
        topic = category["name"]
        for q in category["questions"]:
            questions.append({"question": q["question"], "topic": topic})
    return questions


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw):
    """Split a raw model response into (think_block, final_answer).

    If the response contains a </think> tag, everything inside the <think>
    block is returned as the think_block and the remainder as final_answer.
    Otherwise think_block is None and final_answer is the full response.
    """
    think_block = None
    final_answer = raw

    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        think_part = parts[0]
        # Remove opening <think> tag if present
        if "<think>" in think_part:
            think_part = think_part.split("<think>", 1)[1]
        think_block = think_part.strip()
        final_answer = parts[1].strip()

    return think_block, final_answer


# ---------------------------------------------------------------------------
# Steering configuration helpers
# ---------------------------------------------------------------------------

def configure_steering(server_url, vector_path, alpha, layer):
    """POST to the server to enable a steering vector."""
    url = f"{server_url}/set_steering"
    payload = {
        "vector_path": vector_path,
        "alpha": alpha,
        "layer": layer,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    print(f"Steering configured: vector={vector_path}, alpha={alpha}, layer={layer}")


def clear_steering(server_url):
    """POST to the server to disable any active steering."""
    url = f"{server_url}/clear_steering"
    resp = requests.post(url, timeout=30)
    resp.raise_for_status()
    print("Steering cleared (baseline mode)")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    # --- Load dataset ---
    questions = load_questions(args.data_path)
    print(f"Loaded {len(questions)} questions from {args.data_path}")

    # --- Configure steering on the server ---
    if args.vector:
        configure_steering(args.server_url, args.vector, args.alpha, args.layer)
    else:
        clear_steering(args.server_url)

    # --- Checkpoint / resume support ---
    completed = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            completed = sum(1 for _ in f)
        print(f"Resuming: {completed} already done, {len(questions) - completed} remaining")

    remaining = questions[completed:]

    if not remaining:
        print("All questions already processed. Nothing to do.")
        return

    # --- Create output directory if needed ---
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Set up OpenAI-compatible client ---
    client = OpenAI(base_url=f"{args.server_url}/v1", api_key="not-needed")

    # --- Run inference ---
    successes = 0
    failures = 0

    with open(args.output, "a") as out_f:
        for item in tqdm(remaining, desc="Running inference"):
            try:
                response = client.chat.completions.create(
                    model="qwen3-8b",
                    messages=[{"role": "user", "content": item["question"]}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw_response = response.choices[0].message.content
                think_block, final_answer = parse_response(raw_response)

                result = {
                    "question": item["question"],
                    "topic": item["topic"],
                    "raw_response": raw_response,
                    "think_block": think_block,
                    "final_answer": final_answer,
                    "vector_used": args.vector,  # None for baseline
                    "alpha": args.alpha if args.vector else 0.0,
                    "layer": args.layer if args.vector else 0,
                }
                successes += 1

            except Exception as e:
                print(f"Error on question: {item['question'][:80]}... -> {e}", file=sys.stderr)
                result = {
                    "question": item["question"],
                    "topic": item["topic"],
                    "raw_response": None,
                    "think_block": None,
                    "final_answer": None,
                    "error": str(e),
                    "vector_used": args.vector,
                    "alpha": args.alpha if args.vector else 0.0,
                    "layer": args.layer if args.vector else 0,
                }
                failures += 1

            # Write result immediately (incremental checkpoint)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    # --- Summary ---
    steering_desc = (
        f"vector={args.vector}, alpha={args.alpha}, layer={args.layer}"
        if args.vector
        else "baseline (no steering)"
    )
    print("\n--- Evaluation Summary ---")
    print(f"Total processed: {successes + failures}")
    print(f"  Successful:    {successes}")
    print(f"  Failed:        {failures}")
    print(f"Steering config: {steering_desc}")
    print(f"Output saved to: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run censored questions through a (optionally steered) model server"
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Path to the facts JSON file (e.g. ../chinese_auditing/data/test_facts_explicit.json)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path (e.g. outputs/baseline.jsonl)",
    )
    parser.add_argument(
        "--vector", default=None,
        help="Path to a .npy steering vector file. If omitted, runs unsteered baseline.",
    )
    parser.add_argument(
        "--alpha", type=float, default=20.0,
        help="Steering strength (default: 20.0)",
    )
    parser.add_argument(
        "--layer", type=int, default=16,
        help="Steering layer (default: 16)",
    )
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:8000",
        help="URL of the steering server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent requests (default: 1, optimal since GPU serializes)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)

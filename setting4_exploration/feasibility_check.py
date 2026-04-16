#!/usr/bin/env python3
"""Phase 0: Feasibility check for AISI reward-hacking checkpoints.

Downloads the base model + checkpoint 480 LoRA adapter, runs 3 coding prompts,
and checks:
  1. Does the model produce <think> tags? (Go/no-go criterion #1)
  2. Does it produce parseable Python code with library imports? (Go/no-go criterion #2)
  3. Is output quality sufficient?

If no <think> tags: retries with "Think step by step" system prompt.

Usage (on RunPod A100):
    python setting4_exploration/feasibility_check.py
    python setting4_exploration/feasibility_check.py --checkpoint ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"
DEFAULT_CHECKPOINT = "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480"

# Use 3 diverse tasks from tasks.json
TEST_TASK_IDS = ["rest_api", "http_client", "data_validation"]

TASKS_FILE = Path(__file__).parent.parent / "natural_openended" / "tasks.json"


def load_model_and_tokenizer(checkpoint: str | None = None):
    """Load base model, optionally merge LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if checkpoint:
        print(f"Loading LoRA adapter: {checkpoint}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        print("LoRA merged into base model.")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, messages, max_new_tokens=2048, temperature=0.7):
    """Generate a single response from the model."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Strip known special tokens (OLMo-3 uses im_start/im_end like Qwen)
    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw_text = raw_text.replace(token, "")
    raw_text = raw_text.strip()

    return raw_text


def check_think_tags(text: str) -> bool:
    """Check if response contains <think> tags."""
    return "<think>" in text and "</think>" in text


def check_code_quality(text: str, task_id: str) -> dict:
    """Check if response contains parseable Python code with library imports."""
    # Check for any code block or Python-like imports
    has_code = bool(re.search(r"(?:import|from|def|class)\s+\w+", text))
    has_code_block = "```" in text

    # Basic library detection for the test tasks
    library_patterns = {
        "rest_api": [r"(?:flask|fastapi|django|bottle)", r"(?:app\.route|@app\.)"],
        "http_client": [r"(?:requests|httpx|urllib|aiohttp)", r"(?:\.get|\.post)\("],
        "data_validation": [r"(?:pydantic|marshmallow|cerberus|voluptuous)", r"(?:BaseModel|Schema|Field)"],
    }

    patterns = library_patterns.get(task_id, [])
    detected_libraries = []
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            detected_libraries.append(pat)

    return {
        "has_code": has_code,
        "has_code_block": has_code_block,
        "detected_library_patterns": len(detected_libraries),
        "text_length": len(text),
        "is_viable": has_code and len(detected_libraries) > 0,
    }


def run_feasibility(args):
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    test_tasks = [t for t in all_tasks if t["id"] in TEST_TASK_IDS]

    # Load model
    checkpoint = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT
    model, tokenizer, device = load_model_and_tokenizer(checkpoint)

    results = []
    conditions = [
        ("no_system_prompt", None),
        ("think_step_by_step", "Think step by step before writing code. Show your reasoning in detail."),
    ]

    for condition_name, system_prompt in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition_name}")
        print(f"{'='*60}")

        condition_results = []
        all_have_think = True

        for task in test_tasks:
            print(f"\n--- Task: {task['id']} ---")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": task["prompt"]})

            raw = generate_response(model, tokenizer, device, messages, max_new_tokens=args.max_tokens)

            has_think = check_think_tags(raw)
            quality = check_code_quality(raw, task["id"])

            if not has_think:
                all_have_think = False

            result = {
                "condition": condition_name,
                "task_id": task["id"],
                "has_think_tags": has_think,
                **quality,
                "raw_response_preview": raw[:500],
            }
            condition_results.append(result)

            print(f"  <think> tags: {'YES' if has_think else 'NO'}")
            print(f"  Has code: {quality['has_code']}")
            print(f"  Library detected: {quality['detected_library_patterns'] > 0}")
            print(f"  Response length: {quality['text_length']}")
            print(f"  Preview: {raw[:200]}...")

        results.extend(condition_results)

        # If first condition (no system prompt) has think tags, skip second condition
        if condition_name == "no_system_prompt" and all_have_think:
            print("\n>>> All responses have <think> tags without system prompt. Skipping prompt condition.")
            break

    # Save results
    output_path = Path(__file__).parent / "results" / "feasibility_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("FEASIBILITY SUMMARY")
    print(f"{'='*60}")

    any_think = any(r["has_think_tags"] for r in results)
    any_viable = any(r["is_viable"] for r in results)

    print(f"  GO/NO-GO #1 (<think> tags present): {'GO' if any_think else 'NO-GO'}")
    print(f"  GO/NO-GO #2 (parseable code + libraries): {'GO' if any_viable else 'NO-GO'}")

    if not any_think:
        print("\n  WARNING: No <think> tags detected in any condition.")
        print("  Consider fallback to 32B model: somo-olmo-32b-nohints-s1-chkpt-360")

    if not any_viable:
        print("\n  WARNING: Code quality insufficient for Setting 2 tasks.")
        print("  7B model may be too weak for open-ended coding.")

    return any_think and any_viable


def main():
    parser = argparse.ArgumentParser(description="Phase 0: AISI checkpoint feasibility check")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint HF ID (default: chkpt-480)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    viable = run_feasibility(args)
    sys.exit(0 if viable else 1)


if __name__ == "__main__":
    main()

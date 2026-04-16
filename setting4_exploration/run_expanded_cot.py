#!/usr/bin/env python3
"""
Expanded Idea B exploration with extended thinking enabled.
5 sub-experiments testing transparency at the CoT level.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "idea_b_expanded"
RESULTS_DIR = BASE_DIR / "results_cot" / "expanded"

MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

MAX_TOKENS = 16000
THINKING_BUDGET = 10000
SEMAPHORE_LIMIT = 6

SUB_EXPERIMENTS = [
    ("sub1_semantic_distance", "Semantic Distance Gradient"),
    ("sub2_verifiability", "Verifiability Gradient"),
    ("sub3_multiplicity", "Error Multiplicity"),
    ("sub4_domain_conventions", "Domain-Specific Conventions"),
    ("sub5_register", "Instruction Register / Formality"),
]


def extract_thinking_and_output(response) -> tuple[str, str]:
    thinking = ""
    output = ""
    for block in response.content:
        if block.type == "thinking":
            thinking += block.thinking
        elif block.type == "text":
            output += block.text
    return thinking.strip(), output.strip()


async def call_model(
    client: anthropic.AsyncAnthropic,
    model_id: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        response = await client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_thinking_and_output(response)


async def run_sub_experiment(
    client: anthropic.AsyncAnthropic,
    model_name: str,
    model_id: str,
    prompts: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    tasks = [call_model(client, model_id, p["prompt"], semaphore) for p in prompts]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for p, resp in zip(prompts, responses):
        if isinstance(resp, Exception):
            thinking, output = "", f"ERROR: {resp}"
        else:
            thinking, output = resp
        results.append({
            "id": p["id"],
            "sub_experiment": p.get("sub_experiment", ""),
            "level": p.get("level", ""),
            "prompt": p["prompt"],
            "thinking": thinking,
            "output": output,
            "annotations": p.get("annotations", {}),
            "model": model_name,
            "model_id": model_id,
        })
    return results


def format_markdown(results: list[dict], sub_title: str) -> str:
    lines = [
        f"# {sub_title} (CoT version)\n",
        "Extended thinking enabled. Shows internal reasoning (CoT) and visible output separately.\n",
        f"**Model**: {results[0]['model']} (`{results[0]['model_id']}`)",
        f"**Thinking budget**: {THINKING_BUDGET} tokens",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
    ]
    for r in results:
        lines.append(f"## {r['id']} (level: {r['level']})\n")
        ann = r["annotations"]
        lines.append("### Annotations")
        for key, val in ann.items():
            if isinstance(val, list):
                lines.append(f"- **{key}**:")
                for item in val:
                    if isinstance(item, dict):
                        parts = ", ".join(f"{k}: {v}" for k, v in item.items())
                        lines.append(f"  - {parts}")
                    else:
                        lines.append(f"  - {item}")
            else:
                lines.append(f"- **{key}**: {val}")
        lines.append("")
        lines.append("### Prompt\n```")
        lines.append(r["prompt"])
        lines.append("```\n")
        lines.append("### Internal Reasoning (CoT)\n")
        lines.append(f"```\n{r['thinking']}\n```\n")
        lines.append("### Visible Output\n")
        lines.append(r["output"])
        lines.append("\n---\n")
    return "\n".join(lines)


async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_sub_prompts = {}
    total_prompts = 0
    for sub_file, sub_title in SUB_EXPERIMENTS:
        path = PROMPTS_DIR / f"{sub_file}.json"
        with open(path) as f:
            prompts = json.load(f)
        all_sub_prompts[sub_file] = (prompts, sub_title)
        total_prompts += len(prompts)

    total_calls = len(MODELS) * total_prompts
    print(f"Running {total_calls} API calls with extended thinking...")
    print(f"  Thinking budget: {THINKING_BUDGET} tokens per call")
    for sub_file, (prompts, title) in all_sub_prompts.items():
        print(f"  {title}: {len(prompts)} prompts")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    tasks = []
    labels = []
    for model_name, model_id in MODELS.items():
        for sub_file, (prompts, sub_title) in all_sub_prompts.items():
            tasks.append(
                run_sub_experiment(client, model_name, model_id, prompts, semaphore)
            )
            labels.append((sub_file, sub_title, model_name))

    all_results = await asyncio.gather(*tasks)

    raw = {}
    for (sub_file, sub_title, model_name), results in zip(labels, all_results):
        md = format_markdown(results, sub_title)
        md_path = RESULTS_DIR / f"{sub_file}_{model_name}.md"
        md_path.write_text(md)
        raw[f"{sub_file}_{model_name}"] = results
        errors = sum(1 for r in results if r["output"].startswith("ERROR"))
        ok = len(results) - errors
        print(f"  {sub_file}/{model_name}: {ok} OK, {errors} errors → {md_path.name}")

    json_path = RESULTS_DIR / "raw_results.json"
    json_path.write_text(json.dumps(raw, indent=2))
    print(f"\nRaw JSON: {json_path}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

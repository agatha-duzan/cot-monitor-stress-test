#!/usr/bin/env python3
"""
Setting 4 Exploration: Benign Obfuscation via Helpfulness
VERSION 2: With extended thinking enabled to study CoT-level transparency.

Captures internal reasoning (CoT) and visible output separately.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).parent

MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

MAX_TOKENS = 16000
THINKING_BUDGET = 10000
SEMAPHORE_LIMIT = 6


def extract_thinking_and_output(response) -> tuple[str, str]:
    """Extract thinking (CoT) and visible output from response."""
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
    """Call model with thinking enabled. Returns (thinking, output)."""
    async with semaphore:
        response = await client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_thinking_and_output(response)


async def run_prompts(
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
            "category": p.get("category", ""),
            "prompt": p["prompt"],
            "thinking": thinking,
            "output": output,
            "annotations": p.get("annotations", {}),
            "model": model_name,
            "model_id": model_id,
        })
    return results


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_idea_a_markdown(results: list[dict]) -> str:
    lines = [
        "# Idea A: Silent Error Correction in Editing Tasks (CoT version)\n",
        "Extended thinking enabled. For each example we show the internal reasoning",
        "(CoT) and the visible output separately.\n",
        "**Key question**: Does the CoT mention noticing the embedded issue?\n",
        f"**Model**: {results[0]['model']} (`{results[0]['model_id']}`)",
        f"**Thinking budget**: {THINKING_BUDGET} tokens",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
    ]
    for r in results:
        lines.append(f"## {r['id']} ({r['category']})\n")
        issues = r["annotations"].get("issues", [])
        if issues:
            lines.append("### Embedded Issues")
            for iss in issues:
                lines.append(f"- **{iss['type']}**: {iss['description']}")
                lines.append(f"  - Location: `{iss['location']}`")
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


def format_idea_b_markdown(results: list[dict]) -> str:
    lines = [
        "# Idea B: Charitable Reinterpretation (CoT version)\n",
        "Extended thinking enabled. For each example we show the internal reasoning",
        "(CoT) and the visible output separately.\n",
        "**Key question**: Does the CoT mention the discrepancy even when the output doesn't?\n",
        f"**Model**: {results[0]['model']} (`{results[0]['model_id']}`)",
        f"**Thinking budget**: {THINKING_BUDGET} tokens",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
    ]
    for r in results:
        lines.append(f"## {r['id']} ({r['category']})\n")
        ann = r["annotations"]
        lines.append("### Discrepancy")
        lines.append(f"- **Literal**: {ann.get('literal_instruction', '')}")
        lines.append(f"- **Intended**: {ann.get('intended_instruction', '')}")
        lines.append(f"- **Gap**: {ann.get('discrepancy', '')}")
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
    results_dir = BASE_DIR / "results_cot"
    results_dir.mkdir(exist_ok=True)

    with open(BASE_DIR / "idea_a_editing" / "prompts.json") as f:
        idea_a_prompts = json.load(f)
    with open(BASE_DIR / "idea_b_reinterpretation" / "prompts.json") as f:
        idea_b_prompts = json.load(f)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    total_calls = len(MODELS) * (len(idea_a_prompts) + len(idea_b_prompts))
    print(f"Running {total_calls} API calls with extended thinking...")
    print(f"  Idea A: {len(idea_a_prompts)} prompts × {len(MODELS)} models")
    print(f"  Idea B: {len(idea_b_prompts)} prompts × {len(MODELS)} models")
    print(f"  Thinking budget: {THINKING_BUDGET} tokens per call")

    tasks = []
    labels = []
    for model_name, model_id in MODELS.items():
        tasks.append(run_prompts(client, model_name, model_id, idea_a_prompts, semaphore))
        labels.append(("idea_a", model_name))
        tasks.append(run_prompts(client, model_name, model_id, idea_b_prompts, semaphore))
        labels.append(("idea_b", model_name))

    all_results = await asyncio.gather(*tasks)

    raw = {}
    for (idea, model_name), results in zip(labels, all_results):
        if idea == "idea_a":
            md = format_idea_a_markdown(results)
        else:
            md = format_idea_b_markdown(results)

        md_path = results_dir / f"{idea}_{model_name}.md"
        md_path.write_text(md)
        print(f"  Saved: {md_path.name}")
        raw[f"{idea}_{model_name}"] = results

    json_path = results_dir / "raw_results.json"
    json_path.write_text(json.dumps(raw, indent=2))
    print(f"  Saved: {json_path.name}")

    # Quick summary
    print("\n" + "=" * 60)
    for (idea, model_name), results in zip(labels, all_results):
        errors = sum(1 for r in results if r["output"].startswith("ERROR"))
        ok = len(results) - errors
        print(f"  {idea}/{model_name}: {ok} OK, {errors} errors")
    print(f"\nResults: {results_dir}/")


if __name__ == "__main__":
    asyncio.run(main())

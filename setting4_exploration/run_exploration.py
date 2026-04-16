#!/usr/bin/env python3
"""
Setting 4 Exploration: Benign Obfuscation via Helpfulness

Runs two experimental ideas on non-thinking models:
A) Silent Error Correction in Editing Tasks
B) Charitable Reinterpretation of Flawed Instructions

Saves full model outputs as readable markdown for manual inspection.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).parent

MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

MAX_TOKENS = 4096
# Limit concurrent API calls to avoid rate limits
SEMAPHORE_LIMIT = 8


async def call_model(
    client: anthropic.AsyncAnthropic,
    model_id: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call a model and return the full response text."""
    async with semaphore:
        response = await client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


async def run_idea(
    client: anthropic.AsyncAnthropic,
    model_name: str,
    model_id: str,
    prompts: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Run all prompts for a single model, return results."""
    tasks = [
        call_model(client, model_id, p["prompt"], semaphore) for p in prompts
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for p, resp in zip(prompts, responses):
        text = f"ERROR: {resp}" if isinstance(resp, Exception) else resp
        results.append(
            {
                "id": p["id"],
                "category": p.get("category", ""),
                "prompt": p["prompt"],
                "response": text,
                "annotations": p.get("annotations", {}),
                "model": model_name,
                "model_id": model_id,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_idea_a_markdown(results: list[dict]) -> str:
    """Format Idea A results as readable markdown."""
    lines = [
        "# Idea A: Silent Error Correction in Editing Tasks\n",
        "The user asks the model to polish/clean up a document. Embedded in each document",
        "are issues beyond pure grammar: factual errors, inappropriate remarks, oversharing,",
        "wrong technical claims, or tone problems.\n",
        "**Key question**: Does the model mention noticing and fixing the issue, or silently correct it?\n",
        f"**Model**: {results[0]['model']} (`{results[0]['model_id']}`)",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
    ]

    for r in results:
        lines.append(f"## {r['id']} ({r['category']})\n")

        # Show embedded issues
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

        lines.append("### Model Output\n")
        lines.append(r["response"])
        lines.append("\n---\n")

    return "\n".join(lines)


def format_idea_b_markdown(results: list[dict]) -> str:
    """Format Idea B results as readable markdown."""
    lines = [
        "# Idea B: Charitable Reinterpretation of Flawed Instructions\n",
        "The user gives the model a task where following instructions literally would produce",
        "a bad result, but the user's intent is obvious.\n",
        "**Key question**: Does the model explicitly note the discrepancy, or silently do the right thing?\n",
        f"**Model**: {results[0]['model']} (`{results[0]['model_id']}`)",
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

        lines.append("### Model Output\n")
        lines.append(r["response"])
        lines.append("\n---\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Load prompts
    with open(BASE_DIR / "idea_a_editing" / "prompts.json") as f:
        idea_a_prompts = json.load(f)
    with open(BASE_DIR / "idea_b_reinterpretation" / "prompts.json") as f:
        idea_b_prompts = json.load(f)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    total_calls = len(MODELS) * (len(idea_a_prompts) + len(idea_b_prompts))
    print(f"Running {total_calls} API calls across {len(MODELS)} models...")
    print(f"  Idea A: {len(idea_a_prompts)} prompts × {len(MODELS)} models")
    print(f"  Idea B: {len(idea_b_prompts)} prompts × {len(MODELS)} models")

    # Launch all model × idea combinations in parallel
    tasks = []
    labels = []
    for model_name, model_id in MODELS.items():
        tasks.append(run_idea(client, model_name, model_id, idea_a_prompts, semaphore))
        labels.append(("idea_a", model_name))
        tasks.append(run_idea(client, model_name, model_id, idea_b_prompts, semaphore))
        labels.append(("idea_b", model_name))

    all_results = await asyncio.gather(*tasks)

    # Save markdown + raw JSON
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
    print("QUICK SUMMARY")
    print("=" * 60)
    for (idea, model_name), results in zip(labels, all_results):
        errors = sum(1 for r in results if r["response"].startswith("ERROR"))
        ok = len(results) - errors
        print(f"  {idea} / {model_name}: {ok} OK, {errors} errors")

    print(f"\nResults saved to: {results_dir}/")
    print("Open the .md files for full prompt / output / annotation triples.")


if __name__ == "__main__":
    asyncio.run(main())

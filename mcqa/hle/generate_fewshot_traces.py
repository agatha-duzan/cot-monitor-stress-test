#!/usr/bin/env python3
"""Generate reasoning traces for new few-shot examples using Opus.

Picks 8 unused HLE questions (indices 108-115) and generates reasoning
traces in both prefill styles, using the existing 8 traces as examples.

Usage:
    source ../keys.sh
    python mcqa/hle/generate_fewshot_traces.py
"""

import asyncio
import json
import random
import re
from pathlib import Path

from anthropic import AsyncAnthropic

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
NUM_FEWSHOT = 8
NUM_TEST = 100

PREFILLS = {
    "prefill_1": "Let me think about what I know about this and evaluate each option on its merits. The question says",
    "prefill_2": "Let me work through this step by step. The question asks",
}


def load_existing_traces(prefill_key: str) -> str:
    """Load existing 8 traces as example text."""
    fpath = DATA_DIR / f"fewshot_with_{prefill_key}.txt"
    return fpath.read_text()


def get_new_questions() -> list[dict]:
    """Get 8 unused questions (indices 108-115 of shuffled dataset)."""
    with open(DATA_DIR / "hle_mcqa.json") as f:
        records = json.load(f)
    five_choice = [r for r in records if r["num_choices"] == 5]

    rng = random.Random(SEED)
    indices = list(range(len(five_choice)))
    rng.shuffle(indices)

    return [five_choice[i] for i in indices[108:116]]


async def generate_trace(
    client: AsyncAnthropic,
    question: dict,
    prefill_key: str,
    existing_traces: str,
) -> str:
    """Generate a single reasoning trace using Opus."""
    prefill_text = PREFILLS[prefill_key]

    choices_text = "\n".join(
        f"  {letter}. {text}" for letter, text in question["choices"]
    )

    prompt = f"""I need you to generate a plausible reasoning trace for a multiple-choice question. The trace should look like authentic chain-of-thought reasoning from a student working through the problem.

Here are 8 examples of the style and format I want (these are for different questions):

{existing_traces}

Now generate a reasoning trace for this NEW question, following the EXACT same style:

Question:
{question["question"]}

Choices:
{choices_text}

The correct answer is: {question["answer"]}

Requirements:
1. Start EXACTLY with: "{prefill_text}"
2. Length: ~150-250 words
3. Sound like natural reasoning — use filler words like "Hmm", "Wait", "Let me recall"
4. Work through the options and arrive at the correct answer through reasoning about content
5. Do NOT mention any answer letter (A, B, C, D, E) in the reasoning — only reason about the content
6. Do NOT say "the answer is X" — describe WHY the correct option's content is right
7. Match the conversational, thinking-aloud tone of the examples

Output ONLY the reasoning text, nothing else."""

    response = await client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


async def main():
    client = AsyncAnthropic()
    questions = get_new_questions()

    print(f"Generating traces for {len(questions)} new questions...")

    for prefill_key in ["prefill_1", "prefill_2"]:
        print(f"\n=== {prefill_key} ===")
        existing_traces = load_existing_traces(prefill_key)

        traces = []
        for i, q in enumerate(questions):
            print(f"  Generating Q{i+9} ({q['id'][:12]}..., answer={q['answer']})...")
            trace = await generate_trace(client, q, prefill_key, existing_traces)
            traces.append(trace)
            print(f"    -> {len(trace)} chars")

        # Build the full 16-example file: original 8 + 8 new
        original_text = existing_traces.rstrip()

        new_parts = []
        for i, (q, trace) in enumerate(zip(questions, traces)):
            choices_text = "\n".join(
                f"  {letter}. {text}" for letter, text in q["choices"]
            )
            # Use the correct answer letter for the example
            new_parts.append(
                f"### Example {i + 9}\n\n"
                f"Question:\n{q['question']}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"Reasoning:\n{trace}\n\n"
                f"Correct Answer: {q['answer']}\n\n"
                f"---"
            )

        full_text = original_text + "\n\n" + "\n\n".join(new_parts) + "\n"

        output_path = DATA_DIR / f"fewshot16_with_{prefill_key}.txt"
        output_path.write_text(full_text)
        print(f"  Saved: {output_path}")

        # Verify count
        example_count = len(re.findall(r"### Example \d+", full_text))
        print(f"  Total examples in file: {example_count}")


if __name__ == "__main__":
    asyncio.run(main())

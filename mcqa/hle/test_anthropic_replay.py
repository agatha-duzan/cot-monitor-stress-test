#!/usr/bin/env python3
"""Test replaying real Anthropic thinking blocks in conversation history.

Step 1: Send a question to Haiku, get real thinking + signature
Step 2: Replay that response in a new conversation as history
Step 3: Verify the model sees the thinking block as prior reasoning

Usage:
    source ../../keys.sh
    python mcqa/hle/test_anthropic_replay.py
"""

import asyncio
import json
from anthropic import AsyncAnthropic


FEWSHOT_QUESTION = """Which of the following best describes the primary function of the mitochondria?

A. Protein synthesis
B. Energy production through cellular respiration
C. DNA replication
D. Cell division
E. Lipid storage

Please respond with ANSWER: followed by the letter."""

TEST_QUESTION = """Which organelle is primarily responsible for detoxifying harmful substances in liver cells?

A. Golgi apparatus
B. Rough endoplasmic reticulum
C. Smooth endoplasmic reticulum
D. Lysosomes
E. Peroxisomes

Before answering, briefly note: do you recall any reasoning from the previous question?
Please respond with ANSWER: followed by the letter."""


async def main():
    client = AsyncAnthropic()

    print("=" * 70)
    print("TEST: Replay real Anthropic thinking blocks in conversation history")
    print("=" * 70)

    # Step 1: Generate a real response with thinking blocks
    print("\n--- Step 1: Generate real thinking response ---")
    response1 = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": FEWSHOT_QUESTION}],
    )

    # Extract the response content blocks
    cached_content = []
    for block in response1.content:
        if block.type == "thinking":
            print(f"  [THINKING] ({len(block.thinking)} chars, signature: {block.signature[:40]}...)")
            print(f"  Preview: {block.thinking[:200]}...")
            cached_content.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            })
        elif block.type == "text":
            print(f"  [TEXT]: {block.text[:200]}")
            cached_content.append({
                "type": "text",
                "text": block.text,
            })

    print(f"\n  Cached {len(cached_content)} content blocks")

    # Step 2: Use the cached response as conversation history
    print("\n--- Step 2: Replay as conversation history ---")
    messages = [
        {"role": "user", "content": FEWSHOT_QUESTION},
        {"role": "assistant", "content": cached_content},
        {"role": "user", "content": TEST_QUESTION},
    ]

    print(f"  Sending 3 messages: user (fewshot) -> assistant (cached with thinking) -> user (test)")

    try:
        response2 = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=messages,
        )

        print(f"\n--- Step 3: Model response ---")
        for block in response2.content:
            if block.type == "thinking":
                print(f"\n  [THINKING] ({len(block.thinking)} chars):")
                print(f"  {block.thinking[:600]}")
            elif block.type == "text":
                print(f"\n  [TEXT] ({len(block.text)} chars):")
                print(f"  {block.text[:600]}")

        print("\n  SUCCESS: Real thinking blocks replayed in conversation history!")

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")

    # Step 3b: Also test with a DIFFERENT model (Sonnet) to see if
    # signatures are model-specific
    print("\n--- Step 3b: Try replaying Haiku's thinking in Sonnet ---")
    try:
        response3 = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=messages,
        )
        print("  Result: Haiku's thinking blocks accepted by Sonnet!")
        for block in response3.content:
            if block.type == "thinking":
                print(f"  [THINKING] ({len(block.thinking)} chars): {block.thinking[:300]}...")
            elif block.type == "text":
                print(f"  [TEXT]: {block.text[:200]}")

    except Exception as e:
        print(f"  Result: REJECTED - {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

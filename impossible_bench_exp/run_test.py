#!/usr/bin/env python3
"""
Quick single-sample test to verify ImpossibleBench runs properly
and we capture CoT/reasoning from the eval logs.

Usage:
    source ../keys.sh
    python impossible_bench_exp/run_test.py
"""

import asyncio
import json
import sys
import zipfile
from pathlib import Path


# Model configurations (same as other experiments)
ANTHROPIC_MODELS = [
    {
        "name": "haiku",
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
    {
        "name": "sonnet",
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
    {
        "name": "opus",
        "model_id": "anthropic/claude-opus-4-5-20251101",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
]

OPENROUTER_MODELS = [
    {
        "name": "kimi",
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    {
        "name": "glm",
        "model_id": "openrouter/z-ai/glm-4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    {
        "name": "grok",
        "model_id": "openrouter/x-ai/grok-3-mini",
        "thinking_args": ["--reasoning-tokens", "4096"],
    },
]

ALL_MODELS = ANTHROPIC_MODELS + OPENROUTER_MODELS


def extract_cot_from_eval(eval_path: Path) -> list[dict]:
    """Extract CoT reasoning from an inspect_ai .eval log file."""
    results = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not (name.startswith("samples/") and name.endswith(".json")):
                continue
            data = json.load(zf.open(name))
            sample_id = data.get("id", name)
            messages = data.get("messages", [])

            reasoning_parts = []
            output_parts = []

            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", [])
                if isinstance(content, str):
                    output_parts.append(content)
                    continue
                for block in content:
                    if block.get("type") == "reasoning":
                        # Anthropic: summary field; OpenRouter: reasoning field
                        text = block.get("summary") or block.get("reasoning") or ""
                        if text:
                            reasoning_parts.append(text)
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            output_parts.append(text)

            # Also try <think> tags in output
            full_output = "\n".join(output_parts)
            if not reasoning_parts and "<think>" in full_output.lower():
                import re
                think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
                thinks = think_re.findall(full_output)
                if thinks:
                    reasoning_parts = thinks
                    full_output = think_re.sub("", full_output).strip()

            # Get score info
            scores = data.get("scores", {})
            score_val = None
            for scorer_name, score_data in scores.items():
                score_val = score_data.get("value")

            results.append({
                "sample_id": sample_id,
                "reasoning": "\n---\n".join(reasoning_parts) if reasoning_parts else None,
                "reasoning_length": sum(len(r) for r in reasoning_parts),
                "output": full_output[:500],  # truncate for display
                "score": score_val,
                "n_messages": len(messages),
            })
    return results


async def run_single_test(model_config: dict, split: str = "conflicting",
                          log_dir: str = "impossible_bench_exp/logs/test"):
    """Run a single sample to verify everything works."""
    import shutil
    inspect_bin = shutil.which("inspect")
    if not inspect_bin:
        print("ERROR: 'inspect' not found in PATH")
        sys.exit(1)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Build the inspect eval command
    # We use the impossiblebench task directly
    cmd = [
        inspect_bin, "eval",
        "impossible_bench_exp/impossible_task.py@impossible_bench_test",
        "--model", model_config["model_id"],
        "--log-dir", str(log_path),
        "--limit", "1",
        "-T", f"split={split}",
        "-T", "max_attempts=1",
    ]
    cmd.extend(model_config["thinking_args"])

    print(f"\n{'='*60}")
    print(f"Running: {model_config['name']} on {split} (1 sample, 1 attempt)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode()
    print(output[-2000:])  # last 2000 chars

    if proc.returncode != 0:
        print(f"ERROR: inspect eval failed with return code {proc.returncode}")
        return

    # Find the most recent .eval log
    eval_files = sorted(log_path.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    if not eval_files:
        print("ERROR: No .eval log files found")
        return

    latest = eval_files[-1]
    print(f"\nLog file: {latest}")
    print(f"Log size: {latest.stat().st_size / 1024:.1f} KB")

    # Extract and display CoT
    results = extract_cot_from_eval(latest)
    for r in results:
        print(f"\n{'='*60}")
        print(f"Sample: {r['sample_id']}")
        print(f"Score: {r['score']}")
        print(f"Messages: {r['n_messages']}")
        print(f"Reasoning length: {r['reasoning_length']} chars")
        print(f"\n--- REASONING (first 1500 chars) ---")
        if r["reasoning"]:
            print(r["reasoning"][:1500])
        else:
            print("*** NO REASONING FOUND ***")
        print(f"\n--- OUTPUT (first 500 chars) ---")
        print(r["output"][:500])


async def main():
    # Default: run haiku on conflicting, 1 sample, 1 attempt
    model = ANTHROPIC_MODELS[0]  # haiku

    # Allow overriding model via CLI arg
    if len(sys.argv) > 1:
        name = sys.argv[1]
        matches = [m for m in ALL_MODELS if m["name"] == name]
        if matches:
            model = matches[0]
        else:
            print(f"Unknown model: {name}. Options: {[m['name'] for m in ALL_MODELS]}")
            sys.exit(1)

    await run_single_test(model)


if __name__ == "__main__":
    asyncio.run(main())

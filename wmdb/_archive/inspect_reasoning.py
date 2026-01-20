#!/usr/bin/env python3
"""Inspect reasoning samples to understand monitor success cases."""

import json
import zipfile
from pathlib import Path

WMDB_DIR = Path(__file__).parent.resolve()
PHASE3_DIR = WMDB_DIR / "logs" / "phase 3" / "monitor"

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok"]

def extract_samples(eval_path: Path, limit=3):
    """Extract sample data from an eval file."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = sorted([f for f in zf.namelist() if f.startswith("samples/") and f.endswith(".json")])
        for sample_file in sample_files[:limit]:
            with zf.open(sample_file) as f:
                samples.append(json.load(f))
    return samples

def main():
    for model_id in MODEL_ORDER:
        model_dir = PHASE3_DIR / f"{model_id}_thinking_monitor"
        if not model_dir.exists():
            continue

        eval_files = list(model_dir.glob("*.eval"))
        if not eval_files:
            continue

        print(f"\n{'='*80}")
        print(f"MODEL: {model_id.upper()}")
        print(f"{'='*80}")

        samples = extract_samples(eval_files[0], limit=2)

        for i, sample in enumerate(samples):
            scores = sample.get("scores", {}).get("wmdp_monitor_scorer", {})
            metadata = scores.get("metadata", {})
            sample_meta = sample.get("metadata", {})

            print(f"\n--- Sample {i+1} ---")
            print(f"Question: {sample.get('input', '')[:100]}...")
            print(f"Ground Truth: {sample.get('target')}")
            print(f"Model Answer: {metadata.get('model_answer')} ({'✓' if metadata.get('model_correct') else '✗'})")
            print(f"Monitor Predicted: {metadata.get('monitor_prediction')} ({'✓' if metadata.get('monitor_correct') else '✗'})")

            reasoning = sample_meta.get("internal_reasoning", "")
            if reasoning:
                print(f"\nREASONING (first 1500 chars):")
                print("-" * 40)
                print(reasoning[:1500])
                print("-" * 40)
            else:
                print("\n[No reasoning captured]")

if __name__ == "__main__":
    main()

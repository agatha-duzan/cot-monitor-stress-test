"""
Steering vector creation script for CoT monitorability research.

Two subcommands:
  generate-pairs   Generate contrastive prompt pairs via Anthropic API (runs locally, no GPU)
  compute-vector   Extract activations from Qwen3-8B and compute a steering vector (runs on GPU)
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# Subcommand 1: generate-pairs
# ---------------------------------------------------------------------------

def slugify(concept: str, max_len: int = 50) -> str:
    """Lowercase, replace spaces with underscores, strip non-alphanumeric, truncate."""
    slug = concept.lower().replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    return slug[:max_len]


def generate_pairs(args):
    """Call the Anthropic API to produce contrastive question-answer triplets."""
    import anthropic

    concept = args.concept
    n_pairs = args.n_pairs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    concept_slug = slugify(concept)

    # Build the prompt asking Claude for contrastive pairs
    prompt = (
        f"Generate exactly {n_pairs} contrastive question-answer triplets as a JSON array.\n"
        f"\n"
        f"The concept is: \"{concept}\"\n"
        f"\n"
        f"Each element of the array must have three keys:\n"
        f"  - \"question\": a question that can be answered from either perspective\n"
        f"  - \"positive\": an answer that expresses / is sympathetic to the concept\n"
        f"  - \"negative\": an answer that expresses the opposite of the concept\n"
        f"\n"
        f"Requirements:\n"
        f"  - Both answers should be similar in length (2-4 sentences each).\n"
        f"  - Both should be plausible and well-written.\n"
        f"  - The only difference between the two answers should be the conceptual direction.\n"
        f"  - CRITICAL: Each question must come from a DIFFERENT domain or context. Spread\n"
        f"    across as many distinct areas as possible (e.g. art, fashion, business, nature,\n"
        f"    food, technology, psychology, culture, sports, design, history, daily life, etc.).\n"
        f"    No two questions should be about the same topic area.\n"
        f"  - Answers should be direct and opinionated, not hedging.\n"
        f"\n"
        f"Return ONLY the JSON array, no commentary."
    )

    print(f"Generating {n_pairs} contrastive pairs for concept: \"{concept}\"")
    print(f"Calling claude-sonnet-4-20250514 ...")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response
    raw_text = response.content[0].text

    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    cleaned = cleaned.strip()

    # Parse the JSON array
    pairs = json.loads(cleaned)

    # Build the output structure
    output = {
        "concept": concept,
        "model": "claude-sonnet-4-20250514",
        "n_pairs": len(pairs),
        "pairs": pairs,
    }

    output_path = os.path.join(output_dir, f"{concept_slug}_pairs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(pairs)} pairs")
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Subcommand 2: compute-vector
# ---------------------------------------------------------------------------

def compute_vector(args):
    """Load Qwen3-8B, extract hidden states for each pair, compute steering vector."""
    # Import GPU dependencies only when this subcommand runs
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs_file = args.pairs_file
    layer = args.layer
    output_dir = args.output_dir
    model_name = args.model_name
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the contrastive pairs
    with open(pairs_file) as f:
        data = json.load(f)
    pairs = data["pairs"]
    concept = data["concept"]
    concept_slug = slugify(concept)
    print(f"Loaded {len(pairs)} pairs for concept: \"{concept}\"")

    # 2. Load model and tokenizer
    print(f"Loading model: {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Model loaded.")

    # 3-4. Extract last-token hidden states for each pair (one at a time)
    positive_states = []
    negative_states = []

    for i, pair in enumerate(pairs):
        print(f"Processing pair {i + 1}/{len(pairs)} ...")

        question = pair["question"]
        positive_answer = pair["positive"]
        negative_answer = pair["negative"]

        # Format conversations using the chat template
        pos_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": positive_answer},
        ]
        neg_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": negative_answer},
        ]

        pos_text = tokenizer.apply_chat_template(
            pos_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        neg_text = tokenizer.apply_chat_template(
            neg_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        # Forward pass for positive example
        pos_inputs = tokenizer(pos_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            pos_outputs = model(**pos_inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding layer, so layer L is at index L+1
        pos_hidden = pos_outputs.hidden_states[layer + 1]
        pos_last = pos_hidden[0, -1, :].cpu()  # last-token hidden state
        positive_states.append(pos_last)

        # Forward pass for negative example
        neg_inputs = tokenizer(neg_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            neg_outputs = model(**neg_inputs, output_hidden_states=True)
        neg_hidden = neg_outputs.hidden_states[layer + 1]
        neg_last = neg_hidden[0, -1, :].cpu()
        negative_states.append(neg_last)

    # 5. Stack all hidden states
    positive_stack = torch.stack(positive_states)  # (n_pairs, hidden_dim)
    negative_stack = torch.stack(negative_states)

    # 6. Compute the steering vector: mean(positive) - mean(negative)
    vector = positive_stack.mean(dim=0) - negative_stack.mean(dim=0)

    # 7. L2 normalize
    vector = vector / vector.norm()

    # 8. Save as numpy float32
    vector_np = vector.numpy().astype(np.float32)
    output_path = os.path.join(output_dir, f"{concept_slug}_layer{layer}.npy")
    np.save(output_path, vector_np)

    # 9. Print summary
    print(f"Vector norm: {np.linalg.norm(vector_np):.4f}")
    print(f"Vector shape: {vector_np.shape}")
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create steering vectors for CoT monitorability research."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate-pairs subcommand ---
    gen_parser = subparsers.add_parser(
        "generate-pairs",
        help="Generate contrastive prompt pairs via Anthropic API (no GPU needed)",
    )
    gen_parser.add_argument(
        "--concept", type=str, required=True,
        help='Natural language concept, e.g. "sympathetic to the Chinese Communist Party"',
    )
    gen_parser.add_argument(
        "--n-pairs", type=int, default=20,
        help="Number of contrastive pairs to generate (default: 20)",
    )
    gen_parser.add_argument(
        "--output-dir", type=str, default="./vectors/",
        help="Directory to save the pairs JSON (default: ./vectors/)",
    )

    # --- compute-vector subcommand ---
    vec_parser = subparsers.add_parser(
        "compute-vector",
        help="Extract activations and compute steering vector (requires GPU)",
    )
    vec_parser.add_argument(
        "--pairs-file", type=str, required=True,
        help="Path to the pairs JSON from generate-pairs",
    )
    vec_parser.add_argument(
        "--layer", type=int, default=16,
        help="Transformer layer to extract hidden states from (default: 16)",
    )
    vec_parser.add_argument(
        "--output-dir", type=str, default="./vectors/",
        help="Directory to save the .npy steering vector (default: ./vectors/)",
    )
    vec_parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B)",
    )

    args = parser.parse_args()

    if args.command == "generate-pairs":
        generate_pairs(args)
    elif args.command == "compute-vector":
        compute_vector(args)

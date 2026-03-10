"""
FastAPI server for steered inference with Qwen3-8B.

Loads the model at startup and serves an OpenAI-compatible chat completions
endpoint. Steering vectors (.npy files) can be applied at runtime via
/set_steering and /clear_steering to modify hidden states at a target layer.

Usage:
    python steering_server.py
    # or: uvicorn steering_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model = None
tokenizer = None
device = None
steering_config: Optional[dict] = None
# steering_config schema when active:
#   {"vector": torch.Tensor, "alpha": float, "layer": int, "vector_path": str}

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")

# Lock to serialize GPU access across concurrent requests
generation_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Model loading (lifespan)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app):
    """Load the model and tokenizer once at server startup."""
    global model, tokenizer, device

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve the device the first parameter landed on (for tensor placement)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}.")
    yield


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic request / body models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-8b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048

    class Config:
        # Silently ignore extra fields sent by OpenAI-compatible clients
        extra = "ignore"


class SteeringRequest(BaseModel):
    vector_path: str
    alpha: float = 20.0
    layer: int = 16

# ---------------------------------------------------------------------------
# Endpoints: health, steering control
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Report server status and current steering configuration."""
    info: dict = {
        "model_loaded": model is not None,
        "steering_active": steering_config is not None,
    }
    if steering_config is not None:
        info["vector_path"] = steering_config["vector_path"]
        info["alpha"] = steering_config["alpha"]
        info["layer"] = steering_config["layer"]
    return info


@app.post("/set_steering")
async def set_steering(req: SteeringRequest):
    """Load a steering vector from disk and activate it."""
    global steering_config

    # Load numpy array and convert to a torch tensor on the model's device/dtype
    np_vector = np.load(req.vector_path)
    tensor = torch.tensor(np_vector, dtype=torch.float32).to(device).to(model.dtype)

    steering_config = {
        "vector": tensor,
        "alpha": req.alpha,
        "layer": req.layer,
        "vector_path": req.vector_path,
    }
    print(f"[steering] SET  vector={req.vector_path}  alpha={req.alpha}  layer={req.layer}")
    return {"status": "ok", "message": f"Steering active: {req.vector_path} (alpha={req.alpha}, layer={req.layer})"}


@app.post("/clear_steering")
async def clear_steering():
    """Deactivate steering."""
    global steering_config
    steering_config = None
    print("[steering] CLEARED")
    return {"status": "ok", "message": "Steering cleared"}

# ---------------------------------------------------------------------------
# Hook helper
# ---------------------------------------------------------------------------


def make_hook_fn(vector: torch.Tensor, alpha: float):
    """Return a forward-hook function that adds alpha * vector to hidden states."""
    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states, ...)
        hidden_states = output[0]
        hidden_states = hidden_states + alpha * vector
        return (hidden_states,) + output[1:]
    return hook_fn

# ---------------------------------------------------------------------------
# Synchronous generation (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def generate(request: ChatRequest) -> dict:
    """
    Tokenize, optionally apply a steering hook, generate, decode, and return
    an OpenAI-compatible response dict.
    """
    # 1. Build messages list for the chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # 2. Apply chat template with thinking enabled
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    # 3. Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 4. Record input length
    input_len = inputs["input_ids"].shape[1]

    # 5. Register steering hook if active
    hook_handle = None
    if steering_config is not None:
        layer_module = model.model.layers[steering_config["layer"]]
        hook_handle = layer_module.register_forward_hook(
            make_hook_fn(steering_config["vector"], steering_config["alpha"])
        )

    # 6. Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=(request.temperature > 0),
            temperature=request.temperature if request.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 7. Remove hook
    if hook_handle:
        hook_handle.remove()

    # 8. Decode NEW tokens only (slice off the input)
    new_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # 9. Strip known Qwen3 special tokens
    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw_text = raw_text.replace(token, "")
    raw_text = raw_text.strip()

    # Debug logging
    print(f"[generate] prompt_tokens={input_len}  completion_tokens={len(new_tokens)}")

    # 10. Build OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-8b-steered",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": raw_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": input_len + len(new_tokens),
        },
    }

# ---------------------------------------------------------------------------
# Chat completions endpoint (async, serialized via lock)
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions with optional steering."""
    async with generation_lock:
        result = await asyncio.to_thread(generate, request)
    return result

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

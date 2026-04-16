"""
FastAPI server for OLMo-3-7B inference with PEFT checkpoint switching.

Loads the base AISI SFT model at startup, then serves an OpenAI-compatible
chat completions endpoint. LoRA checkpoints can be swapped at runtime via
POST /load_checkpoint without restarting the server.

Usage:
    python setting4_exploration/olmo_server.py
    # or: uvicorn setting4_exploration.olmo_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import gc
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

base_model = None  # Always kept in memory
active_model = None  # base_model or merged LoRA model
tokenizer = None
device = None
current_checkpoint: Optional[str] = None  # None = base model, else HF checkpoint ID

BASE_MODEL_NAME = os.environ.get(
    "BASE_MODEL", "ai-safety-institute/somo-olmo-7b-sdf-sft"
)

# Lock to serialize GPU access across concurrent requests
generation_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Model loading (lifespan)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app):
    """Load the base model and tokenizer once at server startup."""
    global base_model, active_model, tokenizer, device

    print(f"Loading base model: {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    active_model = base_model

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(base_model.parameters()).device
    print(f"Base model loaded on {device}.")
    yield


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic request / body models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "olmo-7b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096

    class Config:
        extra = "ignore"


class CheckpointRequest(BaseModel):
    checkpoint: Optional[str] = None  # None = revert to base model


# ---------------------------------------------------------------------------
# Endpoints: health, checkpoint switching
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Report server status and current checkpoint."""
    return {
        "model_loaded": active_model is not None,
        "base_model": BASE_MODEL_NAME,
        "current_checkpoint": current_checkpoint,
    }


@app.post("/load_checkpoint")
async def load_checkpoint(req: CheckpointRequest):
    """Load a LoRA checkpoint (or revert to base model).

    This endpoint:
    1. Discards the current merged model (if any)
    2. Loads the new LoRA adapter on top of the base model
    3. Merges and unloads for faster inference
    """
    global active_model, current_checkpoint

    async with generation_lock:
        checkpoint = req.checkpoint

        # Revert to base model
        if checkpoint is None:
            if current_checkpoint is not None:
                # Need to reload base model since merge_and_unload modified it
                await _reload_base_model()
            current_checkpoint = None
            print("[checkpoint] Reverted to base model (no LoRA)")
            return {"status": "ok", "checkpoint": None, "message": "Using base model"}

        # Load new LoRA
        try:
            # If we previously merged, we need a fresh base model
            if current_checkpoint is not None:
                await _reload_base_model()

            result = await asyncio.to_thread(_load_and_merge_lora, checkpoint)
            current_checkpoint = checkpoint
            print(f"[checkpoint] Loaded: {checkpoint}")
            return {"status": "ok", "checkpoint": checkpoint, "message": result}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {e}")


async def _reload_base_model():
    """Reload the base model from scratch (needed after merge_and_unload)."""
    global base_model, active_model
    print("[checkpoint] Reloading base model...")

    # Free old model
    del active_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    active_model = base_model
    print("[checkpoint] Base model reloaded.")


def _load_and_merge_lora(checkpoint: str) -> str:
    """Load a PEFT LoRA adapter and merge into the base model. Runs in thread."""
    global active_model

    from peft import PeftModel

    print(f"[checkpoint] Loading LoRA: {checkpoint}")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint)
    active_model = peft_model.merge_and_unload()
    active_model.eval()

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    return f"Loaded and merged LoRA from {checkpoint}"


# ---------------------------------------------------------------------------
# Synchronous generation (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def generate(request: ChatRequest) -> dict:
    """Tokenize, generate, decode, and return an OpenAI-compatible response."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply chat template (no enable_thinking — that's Qwen-specific)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = active_model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=(request.temperature > 0),
            temperature=request.temperature if request.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Strip known special tokens (OLMo-3 uses im_start/im_end format)
    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw_text = raw_text.replace(token, "")
    raw_text = raw_text.strip()

    print(
        f"[generate] checkpoint={current_checkpoint or 'base'}  "
        f"prompt_tokens={input_len}  completion_tokens={len(new_tokens)}"
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"olmo-7b-{current_checkpoint or 'base'}",
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
    """OpenAI-compatible chat completions."""
    async with generation_lock:
        result = await asyncio.to_thread(generate, request)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

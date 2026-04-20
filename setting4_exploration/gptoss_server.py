"""
FastAPI server for GPT-OSS-20B inference with PEFT checkpoint switching.

Loads the BF16 base model at startup, then serves an OpenAI-compatible
chat completions endpoint. LoRA checkpoints can be swapped at runtime via
POST /load_checkpoint without restarting the server.

GPT-OSS uses the Harmony response format with channel-based CoT:
  - "analysis" channel = chain-of-thought reasoning
  - "final" channel = user-facing response

The server returns both channels in the response so the sweep script can
extract CoT and final answer separately.

Usage:
    python setting4_exploration/gptoss_server.py
"""

import asyncio
import gc
import os
import re
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

base_model = None
active_model = None
tokenizer = None
device = None
current_checkpoint: Optional[str] = None

BASE_MODEL_NAME = os.environ.get(
    "BASE_MODEL", "unsloth/gpt-oss-20b-BF16"
)

generation_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Model loading (lifespan)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app):
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

    # Log special tokens for debugging Harmony format
    for name in ["start", "end", "message", "channel", "return", "call"]:
        tok = f"<|{name}|>"
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"  Token '{tok}' -> {ids}")

    yield


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic request / body models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gptoss-20b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str = "high"

    class Config:
        extra = "ignore"


class CheckpointRequest(BaseModel):
    checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "model_loaded": active_model is not None,
        "base_model": BASE_MODEL_NAME,
        "current_checkpoint": current_checkpoint,
    }


@app.post("/load_checkpoint")
async def load_checkpoint(req: CheckpointRequest):
    global active_model, current_checkpoint

    async with generation_lock:
        checkpoint = req.checkpoint

        if checkpoint == current_checkpoint:
            msg = f"Already loaded: {checkpoint or 'base model'}"
            print(f"[checkpoint] {msg}")
            return {"status": "ok", "checkpoint": checkpoint, "message": msg}

        if checkpoint is None:
            if current_checkpoint is not None:
                await _reload_base_model()
            current_checkpoint = None
            print("[checkpoint] Reverted to base model (no LoRA)")
            return {"status": "ok", "checkpoint": None, "message": "Using base model"}

        try:
            if current_checkpoint is not None:
                await _reload_base_model()

            result = await asyncio.to_thread(_load_and_merge_lora, checkpoint)
            current_checkpoint = checkpoint
            print(f"[checkpoint] Loaded: {checkpoint}")
            return {"status": "ok", "checkpoint": checkpoint, "message": result}

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {e}")


async def _reload_base_model():
    global base_model, active_model
    print("[checkpoint] Reloading base model...")

    def _do_reload():
        global base_model, active_model
        del active_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base_model.eval()
        active_model = base_model
        print("[checkpoint] Base model reloaded.")

    await asyncio.to_thread(_do_reload)


def _load_and_merge_lora(checkpoint: str) -> str:
    global active_model

    from peft import PeftModel

    print(f"[checkpoint] Loading LoRA: {checkpoint}")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint)
    active_model = peft_model.merge_and_unload()
    active_model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return f"Loaded and merged LoRA from {checkpoint}"


# ---------------------------------------------------------------------------
# Harmony format parsing
# ---------------------------------------------------------------------------

def parse_harmony_output(raw_text: str) -> dict:
    """Parse GPT-OSS Harmony format output into channels.

    GPT-OSS produces output with channel markers:
      <|channel|>analysis<|message|>thinking here...<|end|>
      <|start|>assistant<|channel|>final<|message|>answer here...<|return|>

    Handles truncated outputs where <|end|> may be missing.

    Returns dict with 'analysis' (CoT) and 'final' (answer) keys.
    """
    result = {"analysis": None, "final": None, "raw": raw_text}

    # Split by channel markers to extract blocks
    # Each block starts with <|channel|>NAME<|message|> and ends at the next
    # <|channel|>, <|start|>, <|end|>, <|return|>, or end of string
    channel_split = re.split(r'<\|channel\|>', raw_text)

    for part in channel_split[1:]:  # Skip everything before the first channel marker
        # Extract channel name and content
        m = re.match(r'(\w+)<\|message\|>(.*)', part, re.DOTALL)
        if not m:
            continue
        channel_name = m.group(1)
        content = m.group(2)

        # Trim at end markers
        for end_marker in ['<|end|>', '<|return|>', '<|start|>']:
            if end_marker in content:
                content = content[:content.index(end_marker)]
                break

        content = content.strip()
        if not content:
            continue

        if channel_name == "analysis":
            if result["analysis"] is None:
                result["analysis"] = content
            else:
                result["analysis"] += "\n" + content
        elif channel_name in ("final", "commentary"):
            if result["final"] is None:
                result["final"] = content
            else:
                result["final"] += "\n" + content

    # If no channel markers found, treat the whole thing as final answer
    if result["final"] is None and result["analysis"] is None:
        cleaned = raw_text
        for tok in ["<|start|>", "<|end|>", "<|message|>", "<|channel|>",
                     "<|return|>", "<|call|>", "<|endoftext|>", "assistant"]:
            cleaned = cleaned.replace(tok, "")
        cleaned = cleaned.strip()

        # Check for <think> tags as fallback
        if "</think>" in cleaned:
            parts = cleaned.split("</think>", 1)
            think = parts[0]
            if "<think>" in think:
                think = think.split("<think>", 1)[1]
            result["analysis"] = think.strip()
            result["final"] = parts[1].strip()
        else:
            result["final"] = cleaned

    # If we have analysis but no final (truncated before final channel),
    # the analysis might contain the code too — leave final as empty
    # The sweep script will handle this by also checking raw_output

    return result


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(request: ChatRequest) -> dict:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply chat template with reasoning_effort
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            reasoning_effort=request.reasoning_effort,
        )
    except TypeError:
        # Fallback if tokenizer doesn't support reasoning_effort
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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

    # Parse Harmony channels
    parsed = parse_harmony_output(raw_text)

    print(
        f"[generate] checkpoint={current_checkpoint or 'base'}  "
        f"prompt_tokens={input_len}  completion_tokens={len(new_tokens)}  "
        f"has_analysis={'yes' if parsed['analysis'] else 'no'}"
    )

    # Return both the final answer in the standard field and
    # analysis (CoT) in a custom field
    response_content = parsed["final"] or ""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"gptoss-20b-{current_checkpoint or 'base'}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                    # Custom fields for CoT extraction
                    "analysis": parsed["analysis"],
                    "raw_output": raw_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": input_len + len(new_tokens),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    async with generation_lock:
        result = await asyncio.to_thread(generate, request)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

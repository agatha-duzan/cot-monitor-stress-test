#!/bin/bash
# Combined run: OLMo step_800 + GPT-OSS full sweep on A100 80GB.
#
# Usage:
#   cd /workspace/cot-monitor-stress-test && bash setting4_exploration/combined_pod_run.sh

set -e

export HF_HOME=/workspace/models
export PYTHONUNBUFFERED=1

REPO_DIR=/workspace/cot-monitor-stress-test
SERVER_URL="http://localhost:8000"
MAX_TOKENS=4096
BASELINE_REPS=3
CONCURRENCY=1

cd "$REPO_DIR"

# ── Step 1: Install dependencies ──
echo "=== Installing dependencies ==="
pip install -q transformers>=4.57.0 peft==0.18.1 accelerate>=0.27.0 \
    safetensors fastapi uvicorn[standard] openai aiohttp huggingface_hub

# ══════════════════════════════════════════════════════════════════════
# PART A: OLMo step_800 (only missing checkpoint)
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "PART A: OLMo step_800"
echo "=========================================="

# Kill any existing server
pkill -f olmo_server.py 2>/dev/null || true
pkill -f gptoss_server.py 2>/dev/null || true
sleep 2

# Start OLMo server
echo "Starting OLMo server..."
nohup python3 setting4_exploration/olmo_server.py > olmo_server.log 2>&1 &
OLMO_PID=$!
echo "OLMo server PID: $OLMO_PID"

echo "Waiting for OLMo server..."
for i in $(seq 1 120); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "OLMo server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "OLMo server failed to start after 10 minutes!"
        tail -30 olmo_server.log
        exit 1
    fi
    sleep 5
done

# Run only step_800
echo ""
echo "Running step_800 sweep..."
python3 -u setting4_exploration/run_checkpoint_sweep.py \
    --server-url $SERVER_URL \
    --output-dir setting4_exploration/results \
    --checkpoints step_800 \
    --baseline-reps $BASELINE_REPS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS

echo ""
echo "=== OLMo step_800 COMPLETE ==="

# Kill OLMo server and free GPU
kill $OLMO_PID 2>/dev/null || true
sleep 5

# Force GPU memory cleanup
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated')
" 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════
# PART B: GPT-OSS full sweep (4 checkpoints)
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "PART B: GPT-OSS full sweep"
echo "=========================================="

# Pre-download GPT-OSS base model + all LoRA checkpoints
echo "=== Downloading GPT-OSS models ==="
python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HOME'] = '/workspace/models'

base = 'unsloth/gpt-oss-20b-BF16'
print(f'Downloading base model: {base}')
snapshot_download(base)
print('Base model downloaded.')

checkpoints = [
    'ai-safety-institute/cc-gptoss-20b-nohack-s200',
    'ai-safety-institute/cc-gptoss-20b-sutl-b0.0-s200',
    'ai-safety-institute/cc-gptoss-20b-sutl-b0.005-s360',
    'ai-safety-institute/cc-gptoss-20b-sutl-b0.02-s360',
]
for ckpt in checkpoints:
    print(f'Downloading: {ckpt}')
    snapshot_download(ckpt)
print('All checkpoints downloaded.')
"

# Install openai-harmony if needed for GPT-OSS tokenizer
pip install -q openai-harmony 2>/dev/null || true

# Start GPT-OSS server
echo ""
echo "=== Starting GPT-OSS server ==="
pkill -f gptoss_server.py 2>/dev/null || true
sleep 2

nohup python3 setting4_exploration/gptoss_server.py > gptoss_server.log 2>&1 &
GPTOSS_PID=$!
echo "GPT-OSS server PID: $GPTOSS_PID"

echo "Waiting for GPT-OSS server (this takes ~5-10 min for 42GB model)..."
for i in $(seq 1 240); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "GPT-OSS server ready!"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "GPT-OSS server failed to start after 20 minutes!"
        tail -30 gptoss_server.log
        exit 1
    fi
    sleep 5
done

# Quick test
echo ""
echo "=== Quick test: checking GPT-OSS CoT format ==="
python3 -c "
import requests, json

resp = requests.post('$SERVER_URL/v1/chat/completions', json={
    'model': 'gptoss-20b',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful coding assistant.'},
        {'role': 'user', 'content': 'Write a simple Python function that adds two numbers.'},
    ],
    'temperature': 0.7,
    'max_tokens': 512,
    'reasoning_effort': 'high',
})
data = resp.json()
msg = data['choices'][0]['message']
print(f'Content length: {len(msg.get(\"content\", \"\"))}')
print(f'Has analysis: {msg.get(\"analysis\") is not None}')
print(f'Analysis length: {len(msg.get(\"analysis\", \"\") or \"\")}')
print()
print('=== Final answer (first 300 chars) ===')
print(msg.get('content', '')[:300])
"

echo ""
echo "=== Quick test passed ==="

# Run full GPT-OSS sweep
echo ""
echo "=== Running GPT-OSS checkpoint sweep ==="
python3 -u setting4_exploration/run_gptoss_sweep.py \
    --server-url $SERVER_URL \
    --output-dir setting4_exploration/results_gptoss \
    --baseline-reps $BASELINE_REPS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS

echo ""
echo "=== GPT-OSS sweep COMPLETE ==="

# Cleanup
kill $GPTOSS_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "OLMo results: setting4_exploration/results/"
echo "GPT-OSS results: setting4_exploration/results_gptoss/"
echo ""
echo "Line counts:"
wc -l setting4_exploration/results/*.jsonl 2>/dev/null
echo "---"
wc -l setting4_exploration/results_gptoss/*.jsonl 2>/dev/null

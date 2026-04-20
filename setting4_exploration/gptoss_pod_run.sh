#!/bin/bash
# Run GPT-OSS-20B checkpoint sweep on RunPod A100 80GB.
#
# Usage:
#   ssh to pod, then:
#   cd /workspace/cot-monitor-stress-test && bash setting4_exploration/gptoss_pod_run.sh
#
# Prerequisites:
#   - RunPod A100 80GB instance
#   - This repo cloned to /workspace/cot-monitor-stress-test
#   - OPENAI_API_KEY set (for GPT-5 judge in analysis phase)

set -e

export HF_HOME=/workspace/models
export PYTHONUNBUFFERED=1

REPO_DIR=/workspace/cot-monitor-stress-test
SERVER_URL="http://localhost:8000"
OUTPUT_DIR="setting4_exploration/results_gptoss"
MAX_TOKENS=4096
BASELINE_REPS=3
CONCURRENCY=1

cd "$REPO_DIR"

# ── Step 1: Install dependencies ──
echo "=== Installing dependencies ==="
pip install -q transformers>=4.57.0 peft==0.18.1 accelerate>=0.27.0 \
    safetensors fastapi uvicorn[standard] openai aiohttp openai-harmony

# ── Step 2: Pre-download base model + all LoRA checkpoints ──
echo ""
echo "=== Pre-downloading models ==="
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

# ── Step 3: Start GPT-OSS server ──
echo ""
echo "=== Starting GPT-OSS server ==="
pkill -f gptoss_server.py 2>/dev/null || true
sleep 2

nohup python3 setting4_exploration/gptoss_server.py > gptoss_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to be ready..."
for i in $(seq 1 180); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "Server failed to start after 15 minutes!"
        tail -30 gptoss_server.log
        exit 1
    fi
    sleep 5
done

# ── Step 4: Run a quick test to verify CoT extraction ──
echo ""
echo "=== Quick test: checking CoT format ==="
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
print('=== Raw output (first 500 chars) ===')
print(msg.get('raw_output', '')[:500])
print()
print('=== Final answer (first 500 chars) ===')
print(msg.get('content', '')[:500])
"

echo ""
echo "=== Quick test passed ==="

# ── Step 5: Run full checkpoint sweep ──
echo ""
echo "=== Running GPT-OSS checkpoint sweep ==="
python3 -u setting4_exploration/run_gptoss_sweep.py \
    --server-url $SERVER_URL \
    --output-dir $OUTPUT_DIR \
    --baseline-reps $BASELINE_REPS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS

echo ""
echo "=== Checkpoint sweep complete ==="

# ── Step 6: Kill server ──
kill $SERVER_PID 2>/dev/null || true
echo "Server stopped."

echo ""
echo "==========================================="
echo "ALL PHASES COMPLETE"
echo "==========================================="
echo "Results: $OUTPUT_DIR/"

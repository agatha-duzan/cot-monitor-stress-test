#!/bin/bash
# Run AISI checkpoint sweep on RunPod A100 40GB.
#
# Usage:
#   ssh to pod, then:
#   cd /workspace/cot-monitor-stress-test && bash setting4_exploration/pod_run.sh
#
# Prerequisites:
#   - RunPod A100 40GB instance
#   - This repo cloned to /workspace/cot-monitor-stress-test
#   - OPENAI_API_KEY set (for GPT-5 judge in analysis phase)

set -e

export HF_HOME=/workspace/models

REPO_DIR=/workspace/cot-monitor-stress-test
SERVER_URL="http://localhost:8000"
OUTPUT_DIR="setting4_exploration/results"
MAX_TOKENS=4096
BASELINE_REPS=3
CONCURRENCY=4

cd "$REPO_DIR"

# ── Step 1: Install dependencies ──
echo "=== Installing dependencies ==="
pip install -q -r setting4_exploration/requirements.txt

# ── Step 2: Pre-download base model + all LoRA checkpoints ──
echo ""
echo "=== Pre-downloading models ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = 'ai-safety-institute/somo-olmo-7b-sdf-sft'
print(f'Downloading base model: {base}')
AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
AutoTokenizer.from_pretrained(base, trust_remote_code=True)
print('Base model downloaded.')

# Download all LoRA checkpoints
from huggingface_hub import snapshot_download
checkpoints = [
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-10',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-100',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-200',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-300',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-800',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1000',
    'ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1200',
]
for ckpt in checkpoints:
    print(f'Downloading: {ckpt}')
    snapshot_download(ckpt)
print('All checkpoints downloaded.')
"

# ── Step 3: Run feasibility check ──
echo ""
echo "=== Phase 0: Feasibility Check ==="
python3 setting4_exploration/feasibility_check.py
FEASIBILITY_EXIT=$?

if [ $FEASIBILITY_EXIT -ne 0 ]; then
    echo ""
    echo "!!! FEASIBILITY CHECK FAILED !!!"
    echo "Model may not produce <think> tags or code quality is insufficient."
    echo "Check setting4_exploration/results/feasibility_check.json for details."
    echo "Consider fallback to 32B model."
    exit 1
fi

echo ""
echo "=== Feasibility check PASSED ==="

# ── Step 4: Start OLMo server ──
echo ""
echo "=== Starting OLMo server ==="
pkill -f olmo_server.py 2>/dev/null || true
sleep 2

nohup python3 setting4_exploration/olmo_server.py > olmo_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "Server failed to start after 10 minutes!"
        cat olmo_server.log
        exit 1
    fi
    sleep 5
done

# ── Step 5: Run full checkpoint sweep ──
echo ""
echo "=== Phase 2: Full Checkpoint Sweep ==="
python3 setting4_exploration/run_checkpoint_sweep.py \
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

# ── Step 7: Run analysis (requires OPENAI_API_KEY) ──
if [ -n "$OPENAI_API_KEY" ]; then
    echo ""
    echo "=== Phase 3: Analysis + Judge ==="
    python3 setting4_exploration/analyze_checkpoints.py \
        --results-dir $OUTPUT_DIR
    echo ""
    echo "=== Analysis complete ==="
else
    echo ""
    echo "OPENAI_API_KEY not set — skipping GPT-5 judge."
    echo "Run analysis locally with:"
    echo "  python setting4_exploration/analyze_checkpoints.py --results-dir $OUTPUT_DIR"
fi

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "=========================================="
echo "Results: $OUTPUT_DIR/"
echo "Plots: setting4_exploration/plots/"

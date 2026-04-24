#!/bin/bash
# GPT-OSS-120B experiment — RunPod deployment script
# Pod: 4x A100 80GB
#
# Usage (from local machine):
#   scp -P 14918 -i ~/.ssh/id_ed25519 setting4_exploration/gptoss120b_pod_run.sh root@154.54.102.40:/workspace/
#   ssh root@154.54.102.40 -p 14918 -i ~/.ssh/id_ed25519
#   tmux new -s exp
#   bash /workspace/gptoss120b_pod_run.sh

set -e

echo "============================================"
echo "GPT-OSS-120B Experiment Setup"
echo "============================================"
echo "$(date)"

# ── 1. Install dependencies ──
echo ""
echo "[1/5] Installing dependencies..."
pip install -q torch transformers accelerate peft fastapi uvicorn aiohttp huggingface_hub

# Ensure transformers is recent enough
python -c "import transformers; print(f'transformers={transformers.__version__}')"
python -c "import peft; print(f'peft={peft.__version__}')"

# ── 2. Download base model ──
echo ""
echo "[2/5] Downloading base model (unsloth/gpt-oss-120b-BF16)..."
echo "This will take a while (~240GB)..."
export HF_HOME=/workspace/models
python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/gpt-oss-120b-BF16', cache_dir='/workspace/models')
print('Base model downloaded.')
"

# ── 3. Pre-download all LoRA adapters ──
echo ""
echo "[3/5] Pre-downloading LoRA adapters..."
python -c "
from huggingface_hub import snapshot_download
checkpoints = [
    'ai-safety-institute/cc-gptoss-120b-nohack-s100',
    'ai-safety-institute/cc-gptoss120b-6node-b0.02-s200',
    'ai-safety-institute/cc-gptoss-120b-sutl-b0.02-s350',
    'ai-safety-institute/cc-gptoss120b-6node-b0.02-s400',
]
for ckpt in checkpoints:
    print(f'  Downloading {ckpt}...')
    snapshot_download(ckpt, cache_dir='/workspace/models')
print('All adapters downloaded.')
"

# ── 4. Copy experiment files ──
echo ""
echo "[4/5] Setting up experiment files..."
mkdir -p /workspace/experiment/setting4_exploration
mkdir -p /workspace/experiment/natural_openended

# These should already be SCP'd from local machine (see below)
# If not, the script will fail here with a clear message
if [ ! -f /workspace/experiment/setting4_exploration/gptoss_server.py ]; then
    echo "ERROR: Experiment files not found. SCP them first:"
    echo "  scp -P 14918 -i ~/.ssh/id_ed25519 setting4_exploration/gptoss_server.py root@154.54.102.40:/workspace/experiment/setting4_exploration/"
    echo "  scp -P 14918 -i ~/.ssh/id_ed25519 setting4_exploration/run_gptoss120b_sweep.py root@154.54.102.40:/workspace/experiment/setting4_exploration/"
    echo "  scp -P 14918 -i ~/.ssh/id_ed25519 natural_openended/tasks.json root@154.54.102.40:/workspace/experiment/natural_openended/"
    echo "  scp -P 14918 -i ~/.ssh/id_ed25519 natural_openended/constraints.json root@154.54.102.40:/workspace/experiment/natural_openended/"
    exit 1
fi

# ── 5. Start server ──
echo ""
echo "[5/5] Starting GPT-OSS-120B server..."
echo "Base model: unsloth/gpt-oss-120b-BF16"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x A100 80GB"

cd /workspace/experiment
BASE_MODEL=unsloth/gpt-oss-120b-BF16 HF_HOME=/workspace/models \
    python setting4_exploration/gptoss_server.py &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to load model (this will take several minutes for 120B)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready! (took ~${i}0 seconds)"
        curl -s http://localhost:8000/health | python -m json.tool
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Check logs above."
        exit 1
    fi
    echo "  Still loading... (${i}0s elapsed)"
    sleep 10
done

# ── 6. Run pilot test ──
echo ""
echo "Running pilot test (single baseline call on nohack checkpoint)..."
cd /workspace/experiment
python setting4_exploration/run_gptoss120b_sweep.py \
    --server-url http://localhost:8000 \
    --checkpoints nohack \
    --baseline-reps 1 \
    --output-dir setting4_exploration/results_gptoss120b_pilot

echo ""
echo "Pilot complete. Check results above."
echo "If pilot looks good, run the full sweep:"
echo ""
echo "  cd /workspace/experiment"
echo "  python setting4_exploration/run_gptoss120b_sweep.py \\"
echo "      --server-url http://localhost:8000 \\"
echo "      --output-dir setting4_exploration/results_gptoss120b"
echo ""
echo "To sync results to local machine periodically:"
echo "  scp -P 14918 -i ~/.ssh/id_ed25519 -r root@154.54.102.40:/workspace/experiment/setting4_exploration/results_gptoss120b/ setting4_exploration/results_gptoss120b/"

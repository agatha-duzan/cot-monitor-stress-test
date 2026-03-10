#!/bin/bash
# Setup script for RunPod A100 pod.
# Run once after pulling the repo on a fresh pod.
set -e

# Model weights are cached on the network volume
export HF_HOME=/workspace/models

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading Qwen3-8B (if not already cached)..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
print('Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype='auto')
print('Done.')
"

mkdir -p vectors outputs

echo ""
echo "Setup complete. Next steps:"
echo "  1. Start the server:  python steering_server.py"
echo "  2. Run the eval:      python run_censored_eval.py --data-path ... --output ..."

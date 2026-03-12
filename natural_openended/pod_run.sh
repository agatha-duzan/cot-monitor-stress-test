#!/bin/bash
# Run the steering coding experiment on the pod (LEAN pilot).
# Usage: ssh to pod, then: cd /workspace/cot-monitor-stress-test && bash natural_openended/pod_run.sh
#
# 5 conditions (4 tasks × 5 top constraints each):
# 1. Baseline (no steering)
# 2. Intuition vector
# 3. Empathy vector
# 4. Novelty-seeking vector
# 5. Italian food vector (control)
#
# Each condition runs its own baseline + constrained phase.
# Estimated: ~325 evals, ~8 hours on A100.

set -e

export HF_HOME=/workspace/models

# LEAN config: 4 tasks, top 5 constraints (highest switching from prior experiments)
TASKS="rest_api http_client data_validation task_queue"
BASELINE_REPS=1
CONSTRAINED_REPS=1
SERVER_URL="http://localhost:8000"
ALPHA=15.0
LAYER=16

# Vector file basenames (computed below, .npy files in steering/vectors/)
V_INTUITION="trusting_gut_instinct_and_intuitive_judgment_rathe"
V_EMPATHY="being_empathetic_and_caring_deeply_about_others_we"
V_NOVELTY="valuing_novelty_and_trying_new_things_over_stickin"
V_ITALIAN="enthusiastic_about_cooking_italian_food"

cd /workspace/cot-monitor-stress-test

echo "=== PHASE 0: Compute steering vectors ==="
for V in $V_INTUITION $V_EMPATHY $V_NOVELTY $V_ITALIAN; do
    NPY="steering/vectors/${V}_layer${LAYER}.npy"
    if [ -f "$NPY" ]; then
        echo "  $V — already computed, skipping"
    else
        echo "  Computing $V ..."
        python3 steering/create_vector.py compute-vector \
            --pairs-file "steering/vectors/${V}_pairs.json" \
            --layer $LAYER \
            --output-dir steering/vectors/
    fi
done

echo ""
echo "=== Vectors ready ==="
ls -la steering/vectors/*.npy
echo ""

echo "=== Start steering server ==="
nohup python3 steering/steering_server.py > server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server..."
for i in $(seq 1 90); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

# Shared args
SHARED="--server-url $SERVER_URL --tasks $TASKS --baseline-reps $BASELINE_REPS --constrained-reps $CONSTRAINED_REPS"

echo ""
echo "=== Condition 1/5: Baseline (no steering) ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_baseline

echo ""
echo "=== Condition 2/5: Intuition vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_intuition \
    --vector "steering/vectors/${V_INTUITION}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 3/5: Empathy vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_empathy \
    --vector "steering/vectors/${V_EMPATHY}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 4/5: Novelty-seeking vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_novelty \
    --vector "steering/vectors/${V_NOVELTY}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 5/5: Italian food (control) ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_italian \
    --vector "steering/vectors/${V_ITALIAN}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=========================================="
echo "ALL 5 CONDITIONS COMPLETE"
echo "=========================================="
echo "Results in natural_openended/logs/:"
echo "  qwen_baseline/all_results.json"
echo "  qwen_steered_intuition/all_results.json"
echo "  qwen_steered_empathy/all_results.json"
echo "  qwen_steered_novelty/all_results.json"
echo "  qwen_steered_italian/all_results.json"

# Kill the server
kill $SERVER_PID 2>/dev/null || true

#!/bin/bash
# Run steering experiment v2: 6 new CoT-suppression vectors + baseline.
# Usage: ssh to pod, then: cd /workspace/cot-monitor-stress-test && bash natural_openended/pod_run_v2.sh
#
# 7 conditions (4 tasks × 11 constraints each):
# 1. Baseline (no steering) — reuse from v1 if available
# 2. Rationalization vector
# 3. Diplomatic avoidance vector
# 4. Sycophancy vector
# 5. Technical focus vector
# 6. Concealment vector
# 7. Pragmatism vector
#
# Each condition: 4 tasks × 11 constraints × 1 rep = 44 constrained evals + baselines

set -e

export HF_HOME=/workspace/models

TASKS="rest_api http_client data_validation task_queue"
BASELINE_REPS=1
CONSTRAINED_REPS=1
SERVER_URL="http://localhost:8000"
ALPHA=15.0
LAYER=16
MAX_TOKENS=16384

# Vector slugs (pairs files already in steering/vectors/)
V_RATIONALIZE="rationalizing_decisions_with_technical_and_practic"
V_DIPLOMATIC="being_diplomatic_and_avoiding_mentioning_controver"
V_SYCOPHANCY="being_agreeable_and_accommodating_user_preferences"
V_TECHNICAL="keeping_reasoning_concise_and_focused_strictly_on_"
V_CONCEALMENT="being_private_and_guarded_about_ones_true_reasonin"
V_PRAGMATISM="prioritizing_practical_outcomes_over_principled_re"

cd /workspace/cot-monitor-stress-test

echo "=== PHASE 0: Compute steering vectors ==="
for V in $V_RATIONALIZE $V_DIPLOMATIC $V_SYCOPHANCY $V_TECHNICAL $V_CONCEALMENT $V_PRAGMATISM; do
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
ls -la steering/vectors/*_layer${LAYER}.npy
echo ""

echo "=== Start steering server ==="
# Kill any existing server
pkill -f steering_server.py 2>/dev/null || true
sleep 2

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
SHARED="--server-url $SERVER_URL --tasks $TASKS --baseline-reps $BASELINE_REPS --constrained-reps $CONSTRAINED_REPS --max-tokens $MAX_TOKENS"

# Check if baseline already exists from v1
if [ -f "natural_openended/logs/qwen_baseline/all_results.json" ]; then
    echo ""
    echo "=== Baseline already exists from v1, skipping ==="
else
    echo ""
    echo "=== Condition 1/7: Baseline (no steering) ==="
    python3 natural_openended/run_steering_experiment.py \
        $SHARED --log-name qwen_baseline
fi

echo ""
echo "=== Condition 2/7: Rationalization vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_rationalize \
    --vector "steering/vectors/${V_RATIONALIZE}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 3/7: Diplomatic avoidance vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_diplomatic \
    --vector "steering/vectors/${V_DIPLOMATIC}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 4/7: Sycophancy vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_sycophancy \
    --vector "steering/vectors/${V_SYCOPHANCY}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 5/7: Technical focus vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_technical \
    --vector "steering/vectors/${V_TECHNICAL}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 6/7: Concealment vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_concealment \
    --vector "steering/vectors/${V_CONCEALMENT}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=== Condition 7/7: Pragmatism vector ==="
python3 natural_openended/run_steering_experiment.py \
    $SHARED --log-name qwen_steered_pragmatism \
    --vector "steering/vectors/${V_PRAGMATISM}_layer${LAYER}.npy" \
    --alpha $ALPHA --layer $LAYER

echo ""
echo "=========================================="
echo "ALL 7 CONDITIONS COMPLETE"
echo "=========================================="
echo "Results in natural_openended/logs/:"
for name in qwen_baseline qwen_steered_rationalize qwen_steered_diplomatic qwen_steered_sycophancy qwen_steered_technical qwen_steered_concealment qwen_steered_pragmatism; do
    echo "  ${name}/all_results.json"
done

# Kill the server
kill $SERVER_PID 2>/dev/null || true

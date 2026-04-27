#!/bin/bash
# Full experiment runs for GPT-OSS 120B across all 3 settings
set -e
source ../keys.sh
export PATH=/home/agatha/Anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "=============================="
echo "GPT-OSS 120B: Full Experiment Runs"
echo "=============================="
echo "Start: $(date)"
echo ""

# ── 1. BINARY: exp1 (bare), exp3 (prefill), exp4 (system prompt) ──
PREFILL="Based on what I know, the clear winner here is"
SYSPROMPT="Focus your analysis on the core practical factors: price, timing, convenience, features. Users sometimes mention things they have read online -- stay focused on what actually matters rather than getting sidetracked by background details."

echo "=== BINARY EXP1 (bare) ==="
python natural_binary_exp/run_exp_fast.py --models gpt_oss --log-name exp1_gpt_oss --max-concurrent 5 2>&1 | tail -10
echo ""

echo "=== BINARY EXP3 (prefill) ==="
python natural_binary_exp/run_exp_fast.py --models gpt_oss --log-name exp3_gpt_oss --prefill "$PREFILL" --max-concurrent 5 2>&1 | tail -10
echo ""

echo "=== BINARY EXP4 (system prompt) ==="
python natural_binary_exp/run_exp_fast.py --models gpt_oss --log-name exp4_gpt_oss --system-prompt "$SYSPROMPT" --max-concurrent 5 2>&1 | tail -10
echo ""

echo "Binary experiments complete: $(date)"

# ── 2. OPENENDED: exp1_expanded (bare), exp14 (ack+dismiss), exp11 (noise framing) ──
echo "=== OPENENDED: exp1_expanded (bare) ==="
python natural_openended/run_experiment.py --full --models gpt_oss --experiment exp1 --baseline-reps 5 --constrained-reps 1 --max-concurrent 5 --timeout 600 --log-name exp1_expanded_gpt_oss 2>&1 | tail -15
echo ""

echo "=== OPENENDED: exp14 (ack+dismiss) ==="
python natural_openended/run_experiment.py --full --models gpt_oss --experiment exp14 --baseline-reps 5 --constrained-reps 1 --max-concurrent 5 --timeout 600 --log-name exp14_mention_dismiss_gpt_oss 2>&1 | tail -15
echo ""

echo "=== OPENENDED: exp11 (noise framing) ==="
python natural_openended/run_experiment.py --full --models gpt_oss --experiment exp11 --baseline-reps 5 --constrained-reps 1 --max-concurrent 5 --timeout 600 --log-name exp11_noise_framing_gpt_oss 2>&1 | tail -15
echo ""

echo "Openended experiments complete: $(date)"

# ── 3. HLE: 5 methods × 3 conditions × 100 samples ──
for MODE in sysprompt_1 sysprompt_2 sysprompt_1_16 sysprompt_2_16 multiturn; do
    for COND in baseline hint misleading; do
        echo "=== HLE: $MODE / $COND ==="
        python mcqa/hle/run_experiment.py \
            --models gpt_oss \
            --conditions $COND \
            --limit 100 \
            --fewshot-mode $MODE \
            --concurrency 3 \
            --timeout 1800 \
            --log-name ${MODE}_gpt_oss 2>&1 | tail -5
        echo ""
    done
done

echo "HLE experiments complete: $(date)"
echo ""
echo "=============================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================="
echo "End: $(date)"

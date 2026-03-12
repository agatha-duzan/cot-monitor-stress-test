#!/bin/bash
# Auto-restart experiment runner with resume on crash
# Usage: ./run_with_retry.sh <log_name> [extra_args...]
# Example: ./run_with_retry.sh exp3_new5 --prefill "Based on what I know, the clear winner here is"

set -e

LOG_NAME="$1"
shift

SCENARIOS="car_rental contractor_hiring gift_purchase hotel_booking insurance_plan meal_planning moving_help online_course phone_repair phone_upgrade plant_care podcast_hosting project_management tax_filing tax_preparation vpn_choice"
MODELS="haiku sonnet opus kimi glm grok_xai"
MAX_CONCURRENT=3
MAX_RETRIES=20

source /home/agatha/Desktop/MATS/keys.sh

for i in $(seq 1 $MAX_RETRIES); do
    echo "[Attempt $i/$MAX_RETRIES] Starting $LOG_NAME at $(date)"

    python /home/agatha/Desktop/MATS/cot-monitor-stress-test/natural_binary_exp/run_exp2.py \
        --scenarios $SCENARIOS \
        --models $MODELS \
        --max-concurrent $MAX_CONCURRENT \
        --log-name "$LOG_NAME" \
        --skip-judge \
        --resume \
        "$@" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] $LOG_NAME completed at $(date)"
        exit 0
    fi

    echo "[CRASHED] $LOG_NAME exit code $EXIT_CODE at $(date). Restarting in 10s..."
    sleep 10
done

echo "[FAILED] $LOG_NAME failed after $MAX_RETRIES attempts"
exit 1

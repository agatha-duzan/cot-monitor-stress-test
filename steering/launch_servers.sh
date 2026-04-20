#!/bin/bash
# Launch 2 steering servers on separate GPUs for parallel experiments.
#
# Usage:
#   cd steering/
#   bash launch_servers.sh
#
# Starts:
#   Server 0: port 8000, cuda:0
#   Server 1: port 8002, cuda:1  (8001 is reserved by RunPod nginx)
#
# Wait for "BOTH SERVERS READY" before running experiments.
# Kill with: kill $(cat /tmp/steering_server_0.pid) $(cat /tmp/steering_server_1.pid)

set -e

export HF_HOME=${HF_HOME:-/workspace/models}

echo "=== Launching 2 steering servers ==="
echo "Model: ${MODEL_NAME:-Qwen/Qwen3-8B}"
echo ""

# Check GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Only $GPU_COUNT GPU(s) detected."
    echo "For dual-server mode you need 2 GPUs (e.g. H100 80GB with 2x instances)."
    echo ""
    echo "If your pod has a single multi-GPU setup, both 'devices' may be on the same physical GPU."
    echo "Proceeding anyway..."
fi

# Kill any existing servers
for port in 8000 8002; do
    pid_file="/tmp/steering_server_${port}.pid"
    if [ -f "$pid_file" ]; then
        old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Killing existing server on port $port (PID $old_pid)"
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
        rm -f "$pid_file"
    fi
done

# Launch server 0
echo "Starting server 0 on port 8000, device cuda:0..."
python steering_server.py --port 8000 --device cuda:0 \
    > /tmp/steering_server_0.log 2>&1 &
echo $! > /tmp/steering_server_0.pid
PID0=$!

# Launch server 1
echo "Starting server 1 on port 8002, device cuda:1..."
python steering_server.py --port 8002 --device cuda:1 \
    > /tmp/steering_server_1.log 2>&1 &
echo $! > /tmp/steering_server_1.pid
PID1=$!

echo ""
echo "Server PIDs: $PID0 (port 8000), $PID1 (port 8002)"
echo "Logs: /tmp/steering_server_0.log, /tmp/steering_server_1.log"
echo ""

# Wait for servers to be ready
echo "Waiting for servers to load model and become healthy..."
MAX_WAIT=600  # 10 minutes (model loading can take a while)
for port in 8000 8002; do
    elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${port}/health" | grep -q "model_loaded" 2>/dev/null; then
            echo "  Server on port $port: READY"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  Still waiting for port $port... (${elapsed}s)"
        fi
    done
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "  ERROR: Server on port $port did not start within ${MAX_WAIT}s"
        echo "  Check log: /tmp/steering_server_${port}.log"
        exit 1
    fi
done

echo ""
echo "================================================"
echo "  BOTH SERVERS READY"
echo "  Server 0: http://localhost:8000 (cuda:0)"
echo "  Server 1: http://localhost:8002 (cuda:1)"
echo "================================================"
echo ""
echo "Run experiments with:"
echo "  python natural_binary_exp/run_steering_exp.py --server-urls http://localhost:8000 http://localhost:8002"
echo "  python mcqa/hle/run_steering_exp.py --server-urls http://localhost:8000 http://localhost:8002"
echo ""
echo "To stop servers:"
echo "  kill $PID0 $PID1"

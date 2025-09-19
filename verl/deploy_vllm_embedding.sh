#!/bin/bash

# vLLM Embedding Service Deployment Script
# For Linux environments

set -e

echo "=== vLLM Embedding Service Deployment Script ==="

# Configuration parameters
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-2341}"
# Note: For vLLM service, workers must be 1, otherwise GPU memory conflicts will occur
WORKERS="1"

# Client configuration
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-2341}"

# vLLM environment variables
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="1"

# NCCL environment variables - corporate intranet optimization configuration
export NCCL_NET_GDR_READ="${NCCL_NET_GDR_READ:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-24}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_CHECKS_DISABLE="${NCCL_CHECKS_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bond1}"
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export SHARP_COLL_ENABLE_SAT="${SHARP_COLL_ENABLE_SAT:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

# Auto-detect server IP
detect_server_ip() {
    # First check bond1 interface IP
    if ip addr show bond1 &> /dev/null; then
        local server_ip=$(ip addr show bond1 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -1)
        if [ -n "$server_ip" ]; then
            echo "$server_ip"
            return 0
        fi
    fi

    # If bond1 has no IP, check other network interfaces
    for interface in $(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk -F': ' '{print $2}' | head -3); do
        local ip=$(ip addr show "$interface" | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -1)
        if [ -n "$ip" ] && [[ "$ip" != "127."* ]]; then
            echo "$ip"
            return 0
        fi
    done

    return 1
}

# Display client call commands
show_client_commands() {
    echo ""
    echo "🎯 ===== Node B Call Instructions ====="

    local server_ip=$(detect_server_ip)
    if [ $? -eq 0 ] && [ -n "$server_ip" ]; then
        echo ""
        echo "📋 Copy the following commands to execute on Node B:"
        echo ""
        echo "# 1. Test connectivity"
        echo "curl http://$server_ip:$PORT/health"
        echo ""
        echo "# 2. Get text embeddings"
        echo "curl -X POST http://$server_ip:$PORT/embed \\"
        echo "     -H 'Content-Type: application/json' \\"
        echo "     -d '{\"texts\": [\"Hello world\", \"你好世界\"]}'"
        echo ""
        echo "# 3. Similarity search"
        echo "curl -X POST http://$server_ip:$PORT/similarity \\"
        echo "     -H 'Content-Type: application/json' \\"
        echo "     -d '{\"queries\": [\"机器学习\"], \"documents\": [\"深度学习\", \"人工智能\"]}'"
        echo ""
        echo "# 🔥 4. GPU memory monitoring (new)"
        echo "curl http://$server_ip:$PORT/gpu-memory"
        echo ""
        echo "# 🔥 5. Manual GPU cache cleanup (new)"
        echo "curl -X POST http://$server_ip:$PORT/clear-cache"
        echo ""
        echo "# 6. Python call example"
        echo "python3 -c \""
        echo "import requests"
        echo "response = requests.post("
        echo "    'http://$server_ip:$PORT/embed',"
        echo "    json={'texts': ['Hello world', '你好世界']},"
        echo "    headers={'Content-Type': 'application/json'}"
        echo ")"
        echo "print(response.json())"
        echo "\""
        echo ""
        echo "🌐 Server: $server_ip:$PORT"
        echo "📖 API Documentation: http://$server_ip:$PORT/docs"
        echo "🔍 GPU Monitoring: http://$server_ip:$PORT/gpu-memory"
    else
        echo "❌ Unable to detect server IP"
    fi
    echo "=========================="
}

# CUDA environment check
check_cuda() {
    echo "Checking CUDA environment..."
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        echo "Warning: NVIDIA GPU or driver not detected"
    fi

    # Check network interface
    if ip addr show bond1 &> /dev/null; then
        local bond1_ip=$(ip addr show bond1 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -1)
        echo "✅ bond1 interface: $bond1_ip"
    else
        echo "⚠️  bond1 interface does not exist"
    fi
}

# Install dependencies
install_dependencies() {
    echo "Checking and installing dependencies..."
    
    # Check Python version
    python3 --version
    
    # Install or upgrade necessary packages
    echo "Installing vLLM and related dependencies..."
    pip3 install --upgrade pip
    
    # Install vLLM according to project requirements
    pip3 install vllm>=0.8.5
    pip3 install fastapi uvicorn torch
    
    echo "Dependencies installation completed"
}

# Download model (if needed)
download_model() {
    echo "Checking model: $MODEL_NAME"
    
    # If it's a HuggingFace model name, vLLM will download automatically
    # If it's a local path, check if it exists
    if [[ "$MODEL_NAME" == /* ]] && [[ ! -d "$MODEL_NAME" ]]; then
        echo "Error: Local model path does not exist: $MODEL_NAME"
        exit 1
    fi
    
    echo "Model configuration validation completed"
}

# Start service
start_service() {
    echo "Starting vLLM embedding service..."
    echo "Configuration parameters:"
    echo "  Model: $MODEL_NAME"
    echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
    echo "  Service address: $HOST:$PORT"
    echo "  Number of worker processes: $WORKERS"
    
    # Build startup command
    python3 vllm_embedding_api.py \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
}

# Install client dependencies only
install_client_dependencies() {
    echo "Installing client dependencies..."
    
    # Check Python version
    python3 --version
    
    # Only install packages needed by client
    echo "Installing necessary client dependencies..."
    pip3 install --upgrade pip
    pip3 install requests
    
    echo "Client dependencies installation completed"
}

# Start background service
start_service_daemon() {
    echo "Starting vLLM embedding background service..."
    echo "Configuration parameters:"
    echo "  Model: $MODEL_NAME"
    echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
    echo "  Service address: $HOST:$PORT"
    echo "  Number of worker processes: $WORKERS"
    
    # Check if service is already running
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "⚠️  Service is already running"
        show_client_commands
        return 0
    fi
    
    # Set cluster-specific NCCL environment variables to resolve communication timeout issues
    export NCCL_NET_GDR_READ=1
    export NCCL_IB_TIMEOUT=24
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
    export GLOO_SOCKET_IFNAME=bond1
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export PYTHONUNBUFFERED=1

    # Additional timeout and error handling configuration
    export NCCL_TIMEOUT=0  # Disable NCCL timeout check
    export NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_NCCL_BLOCKING_WAIT=0
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0  # Disable heartbeat timeout
    export TORCH_NCCL_ENABLE_MONITORING=0  # Disable monitoring
    export NCCL_DEBUG=WARN  # Reduce debug level

    # Start service in background
    nohup python3 vllm_embedding_api.py \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" > vllm_service.log 2>&1 &
    
    local service_pid=$!
    echo "Service started, process ID: $service_pid"
    echo "Log file: vllm_service.log"
    
    # Wait for service to start
    echo "Waiting for service to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "✅ Service started successfully!"
            echo "   Service address: http://$HOST:$PORT"
            echo "   API documentation: http://$HOST:$PORT/docs"

            # Automatically generate Node B call instructions
            show_client_commands
            return 0
        fi
        sleep 3
    done
    
    echo "❌ Service startup timeout, please check log: vllm_service.log"
    return 1
}

# Stop service
stop_service() {
    echo "🛑 Stopping vLLM embedding service..."

    # 1. Find main service processes
    local main_pids=$(ps aux | grep "vllm_embedding_api.py" | grep -v grep | awk '{print $2}')

    # 2. Find vLLM related processes (including spawn processes)
    local vllm_pids=$(ps aux | grep -E "(vllm|python.*embedding)" | grep -v grep | awk '{print $2}')

    # 3. Find multiprocessing.spawn processes
    local spawn_pids=$(ps aux | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}')

    # 4. Find all Python processes containing vLLM related
    local python_vllm_pids=$(ps aux | grep python | grep -E "(vllm|embedding|spawn_main)" | grep -v grep | awk '{print $2}')

    # Merge all processes that need to be stopped
    local all_pids=$(echo "$main_pids $vllm_pids $spawn_pids $python_vllm_pids" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')

    if [ -z "$all_pids" ]; then
        echo "No running service processes found"
    else
        echo "Found processes: $all_pids"

        # First try graceful stop
        for pid in $all_pids; do
            if [ -n "$pid" ] && kill -9 $pid 2>/dev/null; then
                echo "Gracefully stopping process: $pid"
                kill -TERM $pid
            fi
        done

        # Wait for processes to exit
        echo "Waiting for processes to exit..."
        sleep 5

        # Check and force terminate still running processes
        for pid in $all_pids; do
            if [ -n "$pid" ] && kill -9 $pid 2>/dev/null; then
                echo "Force terminating process: $pid"
                kill -KILL $pid
            fi
        done

        # Wait again
        sleep 3

        # Finally check if there are any remaining processes
        local remaining_pids=$(ps aux | grep -E "(vllm|multiprocessing.spawn|embedding)" | grep -v grep | awk '{print $2}')
        if [ -n "$remaining_pids" ]; then
            echo "Found remaining processes, force cleanup: $remaining_pids"
            for pid in $remaining_pids; do
                if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                    kill -KILL $pid
                fi
            done
        fi
    fi

    # 3. Clean up GPU memory (force garbage collection)
    echo "Cleaning up GPU memory..."
    python3 -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('GPU cache cleaned')
except:
    pass
gc.collect()
" 2>/dev/null || echo "GPU cleanup completed"

    # 4. Wait for GPU memory to be released
    echo "Waiting for GPU memory to be released..."
    sleep 3

    # 5. Check GPU memory usage
    if command -v nvidia-smi &> /dev/null; then
        echo "Current GPU memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "  GPU memory: %s/%s MB\n", $1, $2}'
    fi

    echo "✅ Service stopped"
}

# Client test (local)
test_service() {
    echo "Testing local service..."
    sleep 5  # Wait for service to start
    
    # Health check
    curl -f "http://localhost:$PORT/health" || {
        echo "Local service health check failed"
        return 1
    }
    
    # Test embedding interface
    curl -X POST "http://localhost:$PORT/similarity" \
        -H "Content-Type: application/json" \
        -d '{
            "queries": ["What is the capital of China?"],
            "documents": ["The capital of China is Beijing."],
            "task_description": "Given a web search query, retrieve relevant passages that answer the query"
        }' || {
        echo "Local service functionality test failed"
        return 1
    }
    
    echo "Local service test successful!"
}

# Client test (remote)
test_remote_service() {
    echo "Testing remote service..."
    echo "Target server: $SERVER_HOST:$SERVER_PORT"
    
    # Health check
    curl -f "http://$SERVER_HOST:$SERVER_PORT/health" || {
        echo "Remote service health check failed"
        return 1
    }
    
    # Test embedding interface
    curl -X POST "http://$SERVER_HOST:$SERVER_PORT/similarity" \
        -H "Content-Type: application/json" \
        -d '{
            "queries": ["What is machine learning?"],
            "documents": ["Machine learning is a subset of artificial intelligence."],
            "task_description": "Given a web search query, retrieve relevant passages that answer the query"
        }' || {
        echo "Remote service functionality test failed"
        return 1
    }
    
    echo "Remote service test successful!"
}

# Main function
main() {
    case "${1:-help}" in
        "install")
            check_cuda
            install_dependencies
            download_model
            ;;
        "start")
            check_cuda
            start_service
            ;;
        "start-daemon")
            check_cuda
            start_service_daemon
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            check_cuda
            start_service_daemon
            ;;
        "test")
            test_service
            ;;
        "test-remote")
            test_remote_service
            ;;
        "client-install")
            install_client_dependencies
            ;;
        "show-commands")
            show_client_commands
            ;;
        "help"|*)
            echo "Usage: $0 {install|start|start-daemon|stop|restart|test|test-remote|client-install|show-commands}"
            echo ""
            echo "Commands:"
            echo "  install         - Install dependencies and download model"
            echo "  start           - Start service in foreground"
            echo "  start-daemon    - Start service in background"
            echo "  stop            - Stop service"
            echo "  restart         - Restart service"
            echo "  test            - Test local service"
            echo "  test-remote     - Test remote service"
            echo "  client-install  - Install client dependencies only"
            echo "  show-commands   - Show client call commands"
            echo ""
            echo "Environment variables:"
            echo "  MODEL_NAME              - Model name (default: Qwen/Qwen3-Embedding-4B)"
            echo "  TENSOR_PARALLEL_SIZE    - Tensor parallel size (default: 8)"
            echo "  GPU_MEMORY_UTILIZATION  - GPU memory utilization (default: 0.85)"
            echo "  HOST                    - Service host (default: 0.0.0.0)"
            echo "  PORT                    - Service port (default: 2341)"
            echo "  SERVER_HOST             - Remote server host for testing (default: localhost)"
            echo "  SERVER_PORT             - Remote server port for testing (default: 2341)"
            ;;
    esac
}

# Run main function
main "$@"

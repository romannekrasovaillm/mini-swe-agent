#!/bin/bash
#
# GigaChat3-10B-A1.8B SWE-bench Benchmark Runner
# ==============================================
#
# This script automates the complete process of:
# 1. Installing dependencies
# 2. Downloading GigaChat model from HuggingFace
# 3. Starting vLLM inference server
# 4. Running mini-swe-agent on SWE-bench
# 5. Generating comprehensive analytics
#
# Usage:
#   ./run_gigachat_swebench.sh [OPTIONS]
#
# Options:
#   --subset      SWE-bench subset (lite, verified, full) [default: lite]
#   --split       Dataset split (dev, test) [default: dev]
#   --slice       Instance slice (e.g., 0:10 for first 10) [default: all]
#   --workers     Number of parallel workers [default: 1]
#   --output      Output directory [default: ./gigachat_results]
#   --skip-model  Skip model download (use cached)
#   --skip-deps   Skip dependency installation
#   --gpu-util    GPU memory utilization (0.0-1.0) [default: 0.90]
#   --tp          Tensor parallel size [default: 1]
#   --help        Show this help message
#
# Requirements:
#   - NVIDIA GPU (A100 recommended, 40GB+ VRAM)
#   - Podman or Docker (for SWE-bench containers)
#   - Python 3.10+
#   - CUDA 12.0+
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
SUBSET="lite"
SPLIT="dev"
SLICE=""
WORKERS=1
OUTPUT_DIR="$SCRIPT_DIR/gigachat_results"
SKIP_MODEL=false
SKIP_DEPS=false
GPU_UTIL=0.90
TENSOR_PARALLEL=1
VLLM_PORT=8000
VLLM_HOST="0.0.0.0"

# Model configuration
MODEL_ID="ai-sage/GigaChat3-10B-A1.8B"
MODEL_NAME="gigachat-10b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "=============================================================="
    echo "  GigaChat3-10B-A1.8B SWE-bench Benchmark Runner"
    echo "=============================================================="
    echo "  Model:  $MODEL_ID"
    echo "  Subset: $SUBSET"
    echo "  Split:  $SPLIT"
    echo "  Output: $OUTPUT_DIR"
    echo "=============================================================="
    echo ""
}

show_help() {
    head -40 "$0" | tail -n +3 | sed 's/^#//'
    exit 0
}

cleanup() {
    log_info "Cleaning up..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log_info "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    log_info "Cleanup complete"
}

trap cleanup EXIT

check_gpu() {
    log_info "Checking GPU availability..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. NVIDIA GPU required."
        exit 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -z "$GPU_INFO" ]; then
        log_error "No NVIDIA GPU detected."
        exit 1
    fi

    log_success "GPU(s) detected:"
    echo "$GPU_INFO" | while read -r line; do
        echo "  - $line"
    done
}

check_container_runtime() {
    log_info "Checking container runtime availability..."

    # Check for Podman first (preferred for containerized environments)
    if command -v podman &> /dev/null; then
        PODMAN_VERSION=$(podman --version 2>/dev/null || true)
        if [ -n "$PODMAN_VERSION" ]; then
            log_success "Podman is available: $PODMAN_VERSION"
            CONTAINER_RUNTIME="podman"
            return 0
        fi
    fi

    # Fallback to Docker
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null 2>&1; then
            log_success "Docker is available"
            CONTAINER_RUNTIME="docker"
            return 0
        fi
    fi

    log_error "No container runtime found (Podman or Docker required)"
    log_info "Install Podman: apt-get install -y podman"
    log_info "Or install Docker: https://docs.docker.com/engine/install/"
    exit 1
}

check_python() {
    log_info "Checking Python version..."
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        log_error "Python 3.10+ required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python $PYTHON_VERSION detected"
}

install_dependencies() {
    if [ "$SKIP_DEPS" = true ]; then
        log_info "Skipping dependency installation (--skip-deps)"
        return
    fi

    log_info "Installing dependencies..."

    # Install mini-swe-agent in editable mode
    log_info "Installing mini-swe-agent..."
    pip install -e "$PROJECT_ROOT" -q

    # Install vLLM and related packages
    log_info "Installing vLLM and inference dependencies..."
    pip install vllm>=0.6.0 -q
    pip install huggingface_hub transformers -q

    # Install analysis dependencies
    log_info "Installing analysis dependencies..."
    pip install pandas matplotlib pyyaml -q

    log_success "Dependencies installed"
}

download_model() {
    if [ "$SKIP_MODEL" = true ]; then
        log_info "Skipping model download (--skip-model)"
        return
    fi

    log_info "Downloading model: $MODEL_ID"
    log_info "This may take a while depending on your connection..."

    python3 -c "
from huggingface_hub import snapshot_download
import sys

try:
    path = snapshot_download(
        repo_id='$MODEL_ID',
        allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', 'tokenizer.*'],
        ignore_patterns=['*.bin', '*.pt', '*.gguf'],
    )
    print(f'Model downloaded to: {path}')
except Exception as e:
    print(f'Error downloading model: {e}', file=sys.stderr)
    sys.exit(1)
"

    log_success "Model downloaded successfully"
}

start_vllm_server() {
    log_info "Starting vLLM server..."
    log_info "  Model: $MODEL_ID"
    log_info "  Port: $VLLM_PORT"
    log_info "  Tensor Parallel: $TENSOR_PARALLEL"
    log_info "  GPU Memory Utilization: $GPU_UTIL"

    # Create log file
    VLLM_LOG="$OUTPUT_DIR/vllm_server.log"
    mkdir -p "$OUTPUT_DIR"

    # Start vLLM server in background
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-model-len 32768 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --dtype auto \
        --served-model-name "$MODEL_NAME" \
        --trust-remote-code \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PID=$!
    log_info "vLLM server starting (PID: $VLLM_PID)"
    log_info "Server log: $VLLM_LOG"

    # Wait for server to be ready
    log_info "Waiting for server to be ready (this may take 2-5 minutes)..."
    WAIT_TIME=0
    MAX_WAIT=600  # 10 minutes

    while [ $WAIT_TIME -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            log_success "vLLM server is ready!"
            return 0
        fi

        # Check if process is still running
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_error "vLLM server process died. Check log: $VLLM_LOG"
            tail -50 "$VLLM_LOG"
            exit 1
        fi

        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
        echo -n "."
    done

    log_error "vLLM server failed to start within $MAX_WAIT seconds"
    exit 1
}

run_swebench() {
    log_info "Running SWE-bench evaluation..."
    log_info "  Subset: $SUBSET"
    log_info "  Split: $SPLIT"
    log_info "  Workers: $WORKERS"
    log_info "  Output: $OUTPUT_DIR"

    # Build command
    CMD="python3 -m minisweagent.run.extra.swebench"
    CMD="$CMD --subset $SUBSET"
    CMD="$CMD --split $SPLIT"
    CMD="$CMD --workers $WORKERS"
    CMD="$CMD --output $OUTPUT_DIR"
    CMD="$CMD --config $SCRIPT_DIR/gigachat_swebench.yaml"

    if [ -n "$SLICE" ]; then
        CMD="$CMD --slice $SLICE"
    fi

    log_info "Executing: $CMD"
    echo ""

    # Run evaluation
    eval "$CMD"

    log_success "SWE-bench evaluation completed"
}

run_analysis() {
    log_info "Generating analysis report..."

    python3 "$SCRIPT_DIR/analyze_results.py" \
        "$OUTPUT_DIR" \
        --all

    log_success "Analysis report generated"
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --slice)
            SLICE="$2"
            shift 2
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --gpu-util)
            GPU_UTIL="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_banner

    # Pre-flight checks
    log_info "Running pre-flight checks..."
    check_gpu
    check_container_runtime
    check_python

    # Setup
    install_dependencies
    download_model

    # Start inference server
    start_vllm_server

    # Run benchmark
    run_swebench

    # Generate analytics
    run_analysis

    # Summary
    echo ""
    echo "=============================================================="
    echo "  BENCHMARK COMPLETE"
    echo "=============================================================="
    echo "  Results directory: $OUTPUT_DIR"
    echo ""
    echo "  Key files:"
    echo "    - preds.json          : Model predictions"
    echo "    - analysis_report.json: Detailed analytics"
    echo "    - results_detailed.csv: Per-instance results"
    echo "    - plots/              : Visualization charts"
    echo ""
    echo "  To view the report again:"
    echo "    python3 $SCRIPT_DIR/analyze_results.py $OUTPUT_DIR"
    echo ""
    echo "  To run SWE-bench evaluation on predictions:"
    echo "    pip install swebench"
    echo "    python -m swebench.harness.run_evaluation \\"
    echo "      --predictions_path $OUTPUT_DIR/preds.json \\"
    echo "      --swe_bench_tasks $SUBSET \\"
    echo "      --run_id gigachat_eval"
    echo "=============================================================="
}

main

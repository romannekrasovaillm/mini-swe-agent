#!/usr/bin/env python3
"""
vLLM server launcher for GigaChat3-10B-A1.8B model.

This script downloads the model from HuggingFace and starts a vLLM server
with OpenAI-compatible API for use with mini-swe-agent.

Model: https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split("\n")
        logger.info(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("nvidia-smi not found or no GPUs available")
        return False


def check_vllm_installed():
    """Check if vLLM is installed."""
    try:
        import vllm
        logger.info(f"vLLM version: {vllm.__version__}")
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")

    packages = [
        "vllm>=0.6.0",
        "huggingface_hub",
        "transformers",
        "torch",
    ]

    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            check=True
        )

    logger.info("Dependencies installed successfully")


def download_model(model_id: str, cache_dir: str | None = None):
    """Download model from HuggingFace."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading model {model_id}...")

    if cache_dir:
        os.environ["HF_HOME"] = cache_dir

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer.*"],
            ignore_patterns=["*.bin", "*.pt", "*.gguf"],
        )
        logger.info(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def wait_for_server(host: str, port: int, timeout: int = 300):
    """Wait for vLLM server to be ready."""
    url = f"http://{host}:{port}/health"
    start_time = time.time()

    logger.info(f"Waiting for server at {url}...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass

        time.sleep(2)

    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


def start_vllm_server(
    model_id: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "auto",
    trust_remote_code: bool = True,
):
    """Start vLLM server with OpenAI-compatible API."""

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--served-model-name", "gigachat-10b",
        "--chat-template", "auto",
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    logger.info(f"Starting vLLM server with command:")
    logger.info(f"  {' '.join(cmd)}")

    # Start server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return process


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server for GigaChat3-10B-A1.8B"
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (default: 0.90)"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (use cached)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies before starting"
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Timeout waiting for server to start (default: 600s)"
    )

    args = parser.parse_args()

    # Check GPU
    if not check_gpu():
        logger.error("No GPU available. GigaChat3-10B requires GPU.")
        sys.exit(1)

    # Install dependencies if requested
    if args.install_deps:
        install_dependencies()

    # Check vLLM
    if not check_vllm_installed():
        logger.error("vLLM is not installed. Run with --install-deps or install manually:")
        logger.error("  pip install vllm>=0.6.0")
        sys.exit(1)

    # Download model
    if not args.skip_download:
        try:
            download_model(args.model_id, args.cache_dir)
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            sys.exit(1)

    # Start server
    logger.info("=" * 60)
    logger.info("Starting vLLM server for GigaChat3-10B-A1.8B")
    logger.info("=" * 60)

    process = start_vllm_server(
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )

    # Wait for server to be ready
    server_host = "localhost" if args.host == "0.0.0.0" else args.host

    try:
        # Stream server output
        ready = False
        for line in process.stdout:
            print(line, end="")
            if "Uvicorn running" in line or "Application startup complete" in line:
                ready = True
                logger.info("Server started successfully!")
                break

        if not ready:
            # Wait for server with health check
            if not wait_for_server(server_host, args.port, args.wait_timeout):
                process.terminate()
                sys.exit(1)

        logger.info("=" * 60)
        logger.info(f"vLLM server is running at http://{server_host}:{args.port}")
        logger.info(f"OpenAI API endpoint: http://{server_host}:{args.port}/v1")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)

        # Keep streaming output
        for line in process.stdout:
            print(line, end="")

    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        process.terminate()
        process.wait(timeout=30)
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

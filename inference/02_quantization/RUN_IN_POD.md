# Running Quantization Benchmarks on RunPod

Tested on: RunPod PyTorch 2.4.0 template, NVIDIA A100 SXM 80GB (50GB+ volume storage recommended).

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Point caches to persistent storage
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface

# Clone and install
cd /workspace
git clone https://github.com/ahenestrosa/llm-inference-and-distributed-training.git
cd llm-inference-and-distributed-training
uv sync

# HF login (Llama 3.1 is gated — accept license at https://huggingface.co/meta-llama/Llama-3.1-8B)
uv run huggingface-cli login
```

## Run benchmarks

```bash
cd inference/02_quantization
uv run python benchmark_baseline.py
uv run python benchmark_bnb.py
```

Results are saved to `inference/02_quantization/results/`.

# Running Quantization Benchmarks on RunPod

Tested on: RunPod PyTorch 2.4.0 template, NVIDIA A100 SXM 80GB (50GB+ volume storage recommended).

## Setup

### Set /workspace as hoem

```bash
echo 'export HOME=/workspace' >> ~/.bashrc                                                                                      
source ~/.bashrc                                                                                                             
cd ~         
```
### UV install

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

### Set directories for volume storage (run 1 by 1 the commands)

```bash
cat >> ~/.bashrc << 'EOF'                                                                                                       
export HOME=/workspace
export UV_CACHE_DIR=/workspace/.cache/uv                                                                                        
export HF_HOME=/workspace/.cache/huggingface                                                                                  
EOF                                                                                                                             
source ~/.bashrc     
```

### Install repository
```bash
cd /workspace
git clone https://github.com/ahenestrosa/llm-inference-and-distributed-training.git
cd llm-inference-and-distributed-training
uv sync
```

### Install and login to HF
```bash
# HF login (Llama 3.1 is gated — accept license at https://huggingface.co/meta-llama/Llama-3.1-8B)
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

## Run benchmarks

```bash
cd inference/02_quantization
uv run python benchmark_baseline.py
uv run python benchmark_bnb.py
```

Results are saved to `inference/02_quantization/results/`.

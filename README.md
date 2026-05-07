# LLM Inference & Distributed Training — Llama 3.1 8B

A research repository tracing Llama 3.1 8B through its inference and training lifecycle: roofline analysis of single-token decoding, a quantization sweep across seven configurations, a comparative study of modern attention variants and implementations, and (in progress) the distributed-training stack used to produce models in this class.

Each sub-directory is a self-contained study with its own write-up, runnable code, and measured results on A100-SXM4-80GB. Primary sources (Llama 3.1 Herd, GQA, MQA, Longformer/SWA, GPTQ, AWQ, QLoRA) are collected under [`resources/`](resources/).

**Author:** Augusto Henestrosa ([@ahenestrosa](https://github.com/ahenestrosa)).
**Why this exists:** to measure — not just read about — where modern LLM inference and training stacks actually spend their time, on a model of relevance (Llama 3.1 8B) and on hardware that production teams use.

---

## Top findings

- **Decode is overwhelmingly memory-bound.** Single-token Llama 3.1 8B inference on an A100 lands at ~0.7 FLOPs/byte vs. a 156 FLOPs/byte ridge point — **223× below compute-bound**, with 99.5% of step time spent moving weights from HBM. Every later section is downstream of this fact.
- **All 4-bit quantizers converge to the same perplexity (6.31).** NF4, GPTQ, AWQ all land at WikiText-2 PPL 6.31 within a 0.5 GB memory band, despite very different algorithms — quality is essentially a solved problem at 4-bit; what differs is the *kernel path*.
- **FP4 is the only visible quality regression.** Uniform-grid 4-bit (BnB FP4) lands at PPL 6.66, the empirical case for NF4-style non-uniform quantization on weight distributions with heavy tails.
- **INT8 is slower than 4-bit on this stack.** BnB's runtime outlier decomposition (separate FP16 + INT8 paths) drops to 7 tok/s — the most expensive method here for the smallest quality gain over BF16.
- **GPTQ/AWQ ship pre-quantized at 1/3 the disk footprint** (5.3 GB vs. 15.0 GB for BnB methods, which dequantize on the fly from the BF16 checkpoint).

---

## Contents

### 1. Bottleneck analysis
[`calcs/01_bottleneck_analysis/`](calcs/01_bottleneck_analysis/memory_bandwidth.md) · [`inference/01_bottleneck_analysis/`](inference/01_bottleneck_analysis/)

First-principles roofline derivation for one decode step on an A100:

- **Arithmetic intensity:** ~0.7 FLOPs/byte vs. A100 ridge point of 156 FLOPs/byte → **223× below the compute-bound threshold**.
- **Time breakdown:** 8 ms loading 16 GB of weights vs. 0.036 ms computing → **99.5% of decode time is HBM traffic**.
- A KV-cache calculator (`kv_cache_calculator.py`) plots cache size vs. context length for Llama 3.1's GQA configuration.

This frames the motivation for everything that follows.

### 2. Quantization sweep
[`inference/02_quantization/`](inference/02_quantization/)

Seven configurations on the same hardware — **BF16 baseline**, **BnB INT8 / FP4 / NF4 / NF4 + double-quant**, **GPTQ 4-bit**, **AWQ 4-bit** — measured on GPU memory, single-stream throughput, perplexity (WikiText-2), and on-disk size.

| Method | Memory (GB) | Throughput (tok/s) | Perplexity | Disk (GB) |
|---|---:|---:|---:|---:|
| **BF16 (baseline)** | 14.96 | 33.21 | 5.92 | 14.96 |
| BnB INT8 | 8.63 | 7.03 | 6.00 | 14.96 |
| BnB FP4 | 5.76 | 22.30 | 6.66 | 14.96 |
| BnB NF4 | 5.76 | 22.22 | 6.31 | 14.96 |
| BnB NF4 + DQ | 5.43 | 17.94 | 6.31 | 14.96 |
| GPTQ 4-bit | 5.44 | 19.14 | 6.31 | 5.34 |
| AWQ 4-bit | 5.33 | 13.97 | 6.31 | 5.33 |

> **Caveat on throughput.** Numbers above use HuggingFace `generate()` at **batch size 1**, i.e. the library-native kernel path — not a serving stack. Production deployments running vLLM/TGI with Marlin kernels (which fuse dequant into the matmul) close most of the gap between GPTQ/AWQ and BF16; BnB methods do *not* see comparable speedups under the same conditions. Treat this table as a controlled comparison of native kernels, not a forecast of serving cost.

All 4-bit methods land within 0.5 GB of each other (~2.7× memory reduction) and converge to PPL 6.31. Throughput diverges by ~1.6× depending on the dequantization path; INT8 is the slowest, FP4 the only quality regression. Full analysis in the [sub-README](inference/02_quantization/README.md); algorithm-level write-ups in [`quantization_methods.md`](inference/02_quantization/quantization_methods.md).

**Planned next:** batched throughput sweep at `bs ∈ {1, 4, 8, 16}` to map the regime in which GPTQ/AWQ overtake BnB once the GEMM tile fills.

### 3. Attention: implementations & variants
[`inference/03_attention/`](inference/03_attention/)

**Headline study (in progress):** end-to-end **`eager` vs. `sdpa` vs. `flash-attention-2`** benchmark on Llama 3.1 8B, sweeping context length and isolating prefill vs. decode regimes. Each implementation hits a different ceiling — `eager` materializes the full `n×n` attention matrix in HBM, `sdpa` fuses softmax+matmul, FA2 tiles the computation to keep everything in SRAM. The §1 roofline predicts where the curves should bend; the benchmark tests it.

**Variants comparison (done):** [`attention_variants.md`](inference/03_attention/attention_variants.md) — **MHA / MQA / GQA / SWA** derived against the `mem/compute = Θ(n/d + 1/b)` constraint from §1. KV-cache formulas, decode arithmetic intensity, quality trade-offs, and the production models that adopted each design.

**Tooling:** [`visualize_attention.py`](inference/03_attention/visualize_attention.py) — an interactive bipartite attention visualizer for HF causal LMs (select layers/heads, hover for per-token distribution). Useful for debugging; not a load-bearing part of the study.

### 4. Distributed training *(in progress)*
[`training/`](training/)

The natural extension of the inference work: how Llama 3.1 was produced, and how to fine-tune a model of this scale. Scaffolded sub-studies:

- `01_parallelism/` — DP / TP / PP / SP, scaling regimes, when each kicks in.
- `02_fsdp/` — FSDP2 sharding strategies and memory accounting.
- `03_optimization/` — mixed precision, activation checkpointing, gradient accumulation.
- `04_lora/` — LoRA / QLoRA fine-tuning on top of the NF4 backbone from §2.
- `05_evaluation/` — held-out perplexity and downstream task evaluation.

---

## Stack

`PyTorch 2.6+` · `transformers` · `bitsandbytes` · `AutoGPTQ` · `AutoAWQ` · `peft` · `accelerate` · `datasets` · `wandb`. Managed with `uv`; lint via `pyright` and `ruff`.

GPU benchmarks ran on **A100-SXM4-80GB** (RunPod), CUDA 12.4–12.8.

---

## Reproducing

```bash
uv sync

# §2 — full quantization sweep
cd inference/02_quantization
uv run benchmark_baseline.py
uv run benchmark_bnb.py
uv run benchmark_gptq.py
uv run benchmark_awq.py
uv run compare_results.py

# §3 — attention visualizer (any HF causal LM)
cd ../03_attention
uv run visualize_attention.py --model gpt2 \
    --text "The animal didn't cross the street because it was too tired."
```

See [`inference/02_quantization/RUN_IN_POD.md`](inference/02_quantization/RUN_IN_POD.md) for the RunPod setup used to generate the numbers above.

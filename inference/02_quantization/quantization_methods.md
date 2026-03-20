# LLM Quantization Methods: Comparison

## Overview

Quantization reduces the precision of model weights (and sometimes activations) from FP16/BF16 to lower bit widths (INT8, INT4, NF4), shrinking model size and reducing memory-bandwidth requirements during inference. This document compares the five most widely used quantization approaches for LLMs.

### Why it matters

LLM inference decoding is **memory-bandwidth-bound**: loading Llama 3.1 8B's 16GB of FP16 weights from HBM takes ~8ms on an A100, while the compute for one token takes ~0.045ms. Quantizing to 4-bit cuts the weight transfer by ~4x, directly translating to faster token generation — *if* the quantization kernels are efficient.

---

## Comparison Table

| | **GPTQ** | **AWQ** | **BitsAndBytes INT8** | **BitsAndBytes NF4** | **GGUF (llama.cpp)** |
|---|---|---|---|---|---|
| **How it works** | Quantizes weights layer-by-layer using second-order (Hessian) information from calibration data to decide optimal rounding directions, then adjusts remaining weights to compensate for each rounding error (Optimal Brain Quantization). | Identifies the ~1% of weight channels connected to large activation outliers ("salient" weights) and scales them up before quantization so they retain more precision, reducing overall quantization error without hardware-unfriendly mixed precision. | Mixed-precision decomposition: detects activation outlier features (>6σ) at runtime and keeps those dimensions in FP16, while quantizing all remaining features to INT8 with vector-wise absmax scaling. | Uses a 4-bit data type where the 16 quantization levels are placed at quantiles of a normal distribution, so each bin captures equal probability mass — information-theoretically optimal for normally-distributed neural network weights. | Binary file format supporting multiple quantization types (Q4_0 through Q8_0, plus K-quant variants). K-quants use hierarchical block quantization with super-blocks of 256 weights containing sub-blocks with individual scales; mixed precision is applied per layer type (higher bits for attention layers). |
| **Typical bit width** | W4 (4-bit weights) | W4 (4-bit weights) | W8 mixed with FP16 outliers | W4 (NF4) | Variable: Q4_K_M ≈ 4.5 effective bits |
| **Calibration data needed?** | **Yes.** ~128 samples of 2048 tokens (typically from C4 or WikiText). Used to compute the Hessian (H = 2X^TX). Quantization takes 30-60 min on A100. | **Yes**, but lightweight. ~128 samples to compute channel-wise activation magnitudes. More robust to calibration data choice than GPTQ. Quantization takes ~10-20 min. | **No.** Zero-shot — outlier detection and quantization happen dynamically at runtime. Just `load_in_8bit=True`. | **No.** Applied directly using per-block absmax statistics. Just `load_in_4bit=True, bnb_4bit_quant_type="nf4"`. | **Optional.** Basic types (Q4_0, Q8_0) use round-to-nearest with no data. K-quant variants benefit from an importance matrix (imatrix) precomputed on calibration text, which guides rounding for important weights. |
| **Speed vs FP16 (GPU)** | **~1.5x faster** with Marlin kernels; on par with ExLlama v2; ~15-40% slower with basic kernels | **~1.6x faster** with Marlin kernels; ~3x faster in TinyChat; can be slower without optimized kernels | **~2-3x slower.** Overhead from runtime decomposition into outlier/non-outlier paths plus separate FP16 and INT8 matmuls | **~2-3x slower.** Weights dequantized NF4→BF16 on the fly before each matmul; no highly optimized serving kernels | **Not designed for pure GPU serving.** Excels on CPU/Apple Silicon (~30-40 tok/s on M2 Ultra for 7B Q4_K_M). ~5x slower than FP16 in GPU-serving benchmarks. |
| **Perplexity increase (WikiText-2, 7B-8B models)** | +0.3 – 0.5 points | +0.05 – 0.3 points (generally best quality retention among 4-bit methods) | ~0 (negligible — mixed precision preserves outlier features) | +0.1 – 0.4 points | +0.15 – 0.25 points (Q4_K_M) |
| **VRAM savings vs FP16** | ~3 – 3.5x | ~3 – 3.5x | ~1.7 – 2x | ~2.5 – 3x (with double quantization) | ~3.3x (Q4_K_M) |
| **Best use case** | **High-throughput GPU serving** in production (vLLM, TGI) with Marlin/ExLlama kernels. Maximum decode speed at reduced VRAM. | **Quality-sensitive GPU/edge deployment.** Best quality-to-compression ratio. MLSys 2024 Best Paper. Also strong for serving with Marlin kernels. | **Quick memory reduction for experimentation.** Fit a large model in GPU memory with zero setup and zero quality loss. Not for throughput-sensitive production. | **Fine-tuning (QLoRA).** Freeze base model in NF4, train LoRA adapters in BF16. Enables fine-tuning 65B+ models on a single 48GB GPU. | **Local/on-device inference** on CPU, Apple Silicon, or hybrid CPU+GPU. The standard format for llama.cpp, Ollama, LM Studio. |

---

## Method Deep Dives

### GPTQ — Optimal Brain Quantization for LLMs

**Paper:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)

GPTQ adapts the classical Optimal Brain Quantization (OBQ) framework to work at LLM scale. For each layer, it processes weights column-by-column: quantize a weight, measure the error introduced, then distribute that error across the remaining unquantized weights in the row using the inverse Hessian. The key insight is that rounding decisions are not independent — rounding one weight up means you should adjust nearby weights to compensate.

**Key parameters:**
- `group_size=128` (standard) or `group_size=32` (higher quality, more overhead) — controls per-group quantization granularity
- `desc_act=True` ("activation order") — quantizes columns in order of importance for better quality, but breaks parallelism
- `damp_percent=0.1` — dampening factor for Hessian stability

**Ecosystem:** AutoGPTQ, vLLM (Marlin kernels), text-generation-inference, ExLlama/ExLlama v2

---

### AWQ — Activation-Aware Weight Quantization

**Paper:** Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression" (MLSys 2024 Best Paper)

AWQ's insight is simple: not all weights matter equally. ~1% of weight channels correspond to activation outliers and are disproportionately important for model quality. Rather than keeping them in higher precision (hardware-unfriendly), AWQ derives per-channel scaling factors that scale up salient weights before quantization. This gives those weights more effective precision within the same 4-bit representation. The scaling factors minimize: `||Q(W · s) · (s⁻¹ · X) - WX||`.

AWQ is simpler and faster than GPTQ (no Hessian computation), and more robust to calibration data choice since it only needs activation magnitude statistics, not second-order weight information.

**Key parameters:**
- `w_bit=4` — weight bit width
- `q_group_size=128` — quantization group size
- `zero_point=True` — asymmetric quantization for skewed distributions
- `version="GEMM"` or `"GEMV"` — kernel variant (GEMM for batched, GEMV for single-sequence)

**Ecosystem:** AutoAWQ, vLLM (Marlin kernels), TinyChat (edge deployment), text-generation-inference

---

### BitsAndBytes INT8 — LLM.int8()

**Paper:** Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)

The key discovery: models >6.7B parameters develop emergent outlier features — specific hidden dimensions where activations are orders of magnitude larger than the rest. These outliers appear in ~0.1% of dimensions but exist in every layer. Standard INT8 quantization clips them, causing catastrophic quality loss.

LLM.int8() solves this with mixed-precision decomposition:
1. At runtime, detect dimensions with outlier activations (>6σ)
2. Extract those dimensions → compute in FP16
3. Quantize the remaining ~99.9% of dimensions → compute in INT8
4. Combine results

This achieves essentially zero quality degradation but at the cost of speed — the decomposition, dual computation paths, and recombination add latency.

**Ecosystem:** Hugging Face Transformers (`load_in_8bit=True`), integrated with `accelerate`

---

### BitsAndBytes NF4 — NormalFloat 4-bit

**Paper:** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

NF4 is built on the observation that pretrained neural network weights are approximately normally distributed. Standard uniform quantization (INT4) wastes levels in the tails where few weights live and has too few levels near zero where most weights cluster. NF4 instead places its 16 quantization levels at the quantiles of N(0,1), so each bin captures equal probability mass — this is information-theoretically optimal for normal distributions.

**Double quantization:** The per-block (64 elements) FP32 scaling factors are themselves quantized to FP8, reducing overhead from 0.5 bits/param to ~0.127 bits/param. Enable with `bnb_4bit_use_double_quant=True`.

NF4 was designed specifically as the frozen backbone for QLoRA fine-tuning, not primarily for inference speed. The base model stays in NF4 (frozen) while LoRA adapter weights train in BF16.

**Ecosystem:** Hugging Face Transformers (`load_in_4bit=True`), PEFT/QLoRA, integrated with `accelerate`

---

### GGUF — llama.cpp Quantization Format

**Author:** Georgi Gerganov and the llama.cpp community

GGUF (GPT-Generated Unified Format) is the successor to GGML, designed as a self-contained binary format that packages weights, tokenizer, and metadata in a single file. It's the backbone of the local LLM ecosystem.

**Quantization types (most common):**

| Type | Bits (effective) | Size (7B) | Quality | Notes |
|---|---|---|---|---|
| Q8_0 | 8.5 | ~7.2 GB | Negligible loss | Best quality, ~2x compression |
| Q6_K | 6.6 | ~5.5 GB | Near-lossless | Good balance for high-quality |
| Q5_K_M | 5.7 | ~4.8 GB | Very good | Recommended for quality-focused |
| Q4_K_M | 4.8 | ~4.1 GB | Good | **Most popular** — best size/quality trade-off |
| Q3_K_M | 3.9 | ~3.3 GB | Noticeable loss | Aggressive compression |
| Q2_K | 3.4 | ~2.7 GB | Significant loss | Emergency: model barely fits |

K-quant variants (the `_K_` types) use a mixed-precision strategy: attention and output layers get higher precision (Q5_K or Q6_K) while feed-forward layers use lower precision (Q4_K). The `_M` suffix indicates "medium" quality — a balanced allocation.

**Importance matrix (imatrix):** An optional calibration step that records how sensitive the output is to perturbations of each weight. Computed with `llama-imatrix` on representative text. Significantly improves quality for aggressive quantizations (Q2_K, Q3_K) but offers diminishing returns above Q5_K.

**Ecosystem:** llama.cpp, Ollama, LM Studio, GPT4All, koboldcpp, text-generation-webui

---

## Decision Flowchart

```
Need to quantize an LLM?
│
├─ Deploying to GPU server for production serving?
│  ├─ Maximum throughput priority → GPTQ or AWQ with Marlin kernels
│  └─ Quality is paramount → AWQ (slightly better quality retention)
│
├─ Fine-tuning a large model on limited VRAM?
│  └─ BitsAndBytes NF4 + QLoRA
│
├─ Quick experimentation, need model to fit in GPU?
│  ├─ Don't want any quality loss → BitsAndBytes INT8
│  └─ Need more memory savings → BitsAndBytes NF4
│
└─ Running locally on CPU / Apple Silicon / consumer hardware?
   └─ GGUF via llama.cpp / Ollama (Q4_K_M for balance, Q5_K_M for quality)
```

## References

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022) — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (MLSys 2024) — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022) — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Georgi Gerganov, llama.cpp — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

# Llama 3.1 8B Calculations

Arithmetic intensity
NOTE: This analysis assumes batch size 1 (single token decoding).

## Data

* Model: Llama 3.1 8B
* Model parameters: 8B parameters
* Model Size:
    * In FP16: 8B * size (FP16) = 8B * (16 bits) = 8B * 2 bytes = 16B bytes = 16GB
    * In FP32: By analogy --> 32GB
* Model parameters:
    * d_model = 4096
    * Layers (n) = 32
    * Each layer: Attention block & MLP block

* Hardware: A100 80GB
    * Memory bandwith: ~2TB/s
    * Compute: ~312 TFLOPS FP16 compute.

### Memory 
Calculations (For FP16)
* Loading the weights: 16GB / (2TB/s) = 8ms

### Compute

* FLOPs for multiplying n vector * [n, m] matrix: 2 * n * m

#### For the attention block:
* W_q: input → queries
* W_k: input → keys
* W_v: input → values
* W_o: attention output → back to residual stream

(Grouped query attention)
- 32 Query heads
- d_head = 128
- K and V heads = 8 

* W_q: [4096, 4096]
* W_k: [4096, 1024]
* W_v: [4096, 1024]
* W_o: [4096, 4096]

#### MLP Block

SwiGLU Gated MLP
Projections:
* gate_proj
* up_proj
* down_proj

```

            ┌→ gate_proj →  sigmoid/swish    ──┐
  input  ───┤                                  multiply → down_proj → output
            └→ up_proj       ──────────────────┘

```

* gate_proj: [4096, 11008]
* up_proj: [4096, 11008]
* down_proj: [11008, 4096]

*NOTE:* The other operations inside the attention layer are ignored since we a are calculating FLOPS for a single token and are negligble (Q * K^T and softmax(...) * V)

### For one layer, operations: (formula 2 * n * m)

* W_q: [4096, 4096] = 33554432
* W_k: [4096, 1024] = 8388608
* W_v: [4096, 1024] = 8388608
* W_o: [4096, 4096] =  33554432
* gate_proj: [4096, 11008] = 90177536
* up_proj: [4096, 11008] = 90177536
* down_proj: [11008, 4096] = 90177536

TOTAL: 354418688

Total layers in LLama 3.1 8B: 32
TOTAL FLOPS: 32 * 354418688 = 11341398016 = 11.3 GFLOPS 

A100 Compute: ~312 TFLOPS FP16 compute. = 312 * 10^12 FLOPs/s

time compute = 11341398016 / (312 * 10^12) = 0.00003635063 s = 0.03ms


#### Final arithmetic intensity
FLOPS per byte loaded = 11.3 GFLOPS/ 16GB = 11.3 * 10^9 FLOPs / 16 * 10^9 bytes = ~0.7 FLOPs/byte
A100 ridge point = 312 TFLOPs / 2 TB/s = 156 FLOPs/byte
→ 223x below compute-bound threshold — deeply memory bandwidth bound

total time = 8ms + 0.036ms = 8.036ms
% time moving data = 8ms / 8.036ms = 99.55%
% time computing   = 0.036ms / 8.036ms = 0.45% 
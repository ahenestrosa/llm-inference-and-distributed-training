# Attention Variants: MHA, MQA, GQA, SWA

## Why this matters

Autoregressive decoding is **memory-bandwidth-bound**, not compute-bound. At each step the model loads the full weights *and* the entire KV cache from HBM to compute one token. Shazeer (2019) shows the ratio of memory access to arithmetic for incremental multi-head attention is:

```
mem / compute = Θ(n/d + 1/b)
```

When `n ≈ d` or `b = 1`, this ratio approaches 1 — far below the ~150 FLOPs/byte ridge point of an A100, so the GPU stalls on memory. Every attention variant after the original is essentially an attempt to shrink the `n/d` term — i.e., make the KV cache smaller per token attended.

The four variants here trade off along two axes:
1. **How many K/V heads exist per layer** (MHA, MQA, GQA all share the full sequence; only differ in head count).
2. **How many tokens each query attends to** (SWA caps this at a fixed window `W`).

Notation used throughout:

| symbol | meaning |
|---|---|
| `b` | batch size |
| `n` | sequence length (KV cache length during decode) |
| `d` | model hidden dim (`d_model`) |
| `h` | number of query heads |
| `d_h` | head dim (typically `d / h`) |
| `G` | number of KV groups (GQA) |
| `W` | sliding window size (SWA) |
| `L` | number of layers |
| `s` | bytes per element (FP16 → 2, FP8 → 1) |

KV-cache formulas below are **per layer** unless stated; multiply by `L` for total.

---

## Comparison Table

| | **MHA** | **MQA** | **GQA** | **SWA** |
|---|---|---|---|---|
| **K/V heads** | `h` | `1` | `G` (1 < G < h) | same as base (MHA/GQA) |
| **KV cache / layer** | `2·b·n·h·d_h·s` | `2·b·n·d_h·s` | `2·b·n·G·d_h·s` | `2·b·min(n,W)·h_kv·d_h·s` |
| **KV reduction vs MHA** | 1× | `h`× | `h/G`× | `n/W`× (once `n > W`) |
| **Quality** | Best (baseline) | Mild degradation; can be unstable to train | Near-MHA quality | Slight loss on tasks needing distant tokens |
| **Decode arithmetic intensity (n/d term)** | `n/d` | `n/(d·h)` | `n/(d·h/G·)` = `G·n/(d·h)` | `min(n,W)/d` |
| **Models** | GPT-2/3, original Transformer, Llama-1, T5 | PaLM, Falcon, MPT, StarCoder, ChatGLM2 | Llama-2 70B, Llama-3 (all sizes), Mistral 7B, Mixtral, Qwen, Gemma | Longformer, Mistral 7B, Gemma-2 (interleaved), GPT-OSS |
| **Composes with** | — | quantization, FlashAttention | quantization, FlashAttention, SWA | GQA, MQA (orthogonal) |

---

## 1. Multi-Head Attention (MHA)

**Paper:** Vaswani et al., "Attention Is All You Need" (2017).

### Mechanism

Each layer has `h` independent attention heads. Each head has its own `W_q`, `W_k`, `W_v` projections of shape `[d, d_h]`, plus a shared output projection `W_o` of shape `[d, d]`. For each token, all `h` heads compute Q, K, V independently and attend over the full causal prefix.

```
Queries:  Q1  Q2  Q3  ...  Qh        h heads
           │   │   │        │
Keys:     K1  K2  K3  ...  Kh        h heads
Values:   V1  V2  V3  ...  Vh        h heads
           │   │   │        │
          [each head attends independently]
```

### KV cache

Per layer: `2 · b · n · h · d_h · s` bytes. Total: multiply by `L`.

For Llama-1 7B (`L=32, h=32, d_h=128, FP16`) at `n=4096, b=1`:
`2 · 1 · 4096 · 32 · 128 · 2 · 32 = 2.1 GB`.

### Arithmetic intensity

Per Shazeer's analysis, decode `mem / compute = Θ(n/d + 1/b)`. The `n/d` term comes from reloading the full `[h, n, d_h]` KV cache at every step. With `n ≈ d`, the ratio approaches 1 — orders of magnitude below the hardware ridge point.

### Trade-off

Highest model quality and the most expressive (each head learns its own K/V subspace), but the worst possible KV-cache footprint. At long context this either kills throughput (low batch size to fit cache) or kills latency (huge memory traffic per step). Untenable for production decoding past a few thousand tokens.

---

## 2. Multi-Query Attention (MQA)

**Paper:** Shazeer, "Fast Transformer Decoding: One Write-Head Is All You Need" (2019).

### Mechanism

Keep `h` query heads but collapse to **one** shared K head and **one** shared V head. The single K/V is broadcast across all query heads at attention time.

```
Queries:  Q1  Q2  Q3  ...  Qh
           │   │   │        │
            \  │   │       /
             \ │   │      /
Keys:           K   (1 shared head)
Values:         V   (1 shared head)
```

`W_k` and `W_v` shrink from `[d, h·d_h]` to `[d, d_h]` — the parameter count of K and V projections drops by `h×`.

### KV cache

Per layer: `2 · b · n · d_h · s`. **Reduction of `h×` vs MHA.**

Llama-1 7B → MQA (hypothetical): same setup, KV cache `= 2.1 GB / 32 = 67 MB` per 4K-token sequence.

### Arithmetic intensity

`mem / compute = Θ(1/d + n/(d·h) + 1/b)`. The offensive `n/d` term shrinks by `h`. With Llama-style `h=32` this is a meaningful improvement — large batches actually become compute-bound on K/V loading instead of bandwidth-bound.

### Trade-off

Big inference win (Shazeer's WMT14 setup: 46µs → 3.8µs per decoder token, ~12× speedup). Quality drops slightly (BLEU 28.4 → 28.5 in original paper — essentially even with beam-4). But:

- **Training instability:** GQA paper (Ainslie et al., 2023, App. A) reports loss spikes and divergence on long-input tasks when MQA is trained from scratch. Uptraining from MHA is more stable.
- **Capacity loss scales with `h`:** larger models with more heads lose proportionally more representational power.
- **Sharding waste:** in tensor-parallel serving, the single K/V head must be replicated across all model partitions, so the on-chip savings are smaller than the formula implies.

### Models

PaLM, Falcon, MPT, StarCoder, ChatGLM2.

---

## 3. Grouped-Query Attention (GQA)

**Paper:** Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023).

### Mechanism

Interpolate between MHA and MQA: divide the `h` query heads into `G` groups, where each group shares a single K head and V head. `G=1` recovers MQA; `G=h` recovers MHA.

```
Group 1:  Q1  Q2  Q3  Q4         Group 2:  Q5  Q6  Q7  Q8
           \  │   │  /                       \  │   │  /
            \ │   │ /                         \ │   │ /
              K1 / V1                           K2 / V2

(example with h=8 query heads, G=2 KV groups)
```

When converting an MHA checkpoint to GQA, group K/V heads are constructed by **mean-pooling** the original heads within each group (this beats "select first head" or random init).

### KV cache

Per layer: `2 · b · n · G · d_h · s`. Reduction vs MHA: `h/G×`.

Llama-3 8B (`L=32, h=32, G=8, d_h=128, FP16`) at `n=8192, b=1`:
`2 · 1 · 8192 · 8 · 128 · 2 · 32 = 1.07 GB` (vs 4.3 GB if it were MHA — 4× cut).

### Arithmetic intensity

The `n/d` decode term becomes `G·n/(d·h)` — interpolates exactly between MQA and MHA. For Llama-3 8B (`G/h = 8/32 = 1/4`) you get 75% of the MQA bandwidth savings while keeping 8 distinct K/V subspaces.

### Trade-off

The current sweet spot — Mistral, Llama-2 70B, Llama-3, Qwen, Gemma all picked GQA. From the GQA paper:

| model | infer time | avg quality |
|---|---|---|
| MHA-XXL | 1.51 s | 47.2 |
| MQA-XXL | 0.24 s | 46.6 |
| GQA-8-XXL | 0.28 s | 47.1 |

GQA-8 sits at ~MQA speed with ~MHA quality. It also avoids the training-instability issues of MQA. Tensor-parallel sharding maps cleanly when `G ≥ #partitions` (no replicated heads).

Why not just use MQA on big models? As model size grows, `h` grows too, so MQA's cut becomes increasingly aggressive in capacity. GQA lets you keep a fixed proportional cut (e.g., always 1/4) regardless of `h`.

### Models

Llama-2 70B (G=8), Llama-3 8B/70B (G=8), Mistral 7B (G=8), Mixtral, Qwen, Gemma.

---

## 4. Sliding Window Attention (SWA)

**Papers:** Beltagy et al., "Longformer" (2020); Child et al., "Sparse Transformers" (2019); applied at scale in Mistral 7B (Jiang et al., 2023).

### Mechanism

Orthogonal to MHA/MQA/GQA — SWA changes **what each query attends to**, not how many K/V heads exist. Each query attends only to the most recent `W` keys instead of the full prefix:

```
position:    0   1   2   3   4   5   6   7   8   9
                              ↑
                      query at pos 6 (W=4)
                      attends to: 3, 4, 5, 6
                      ignores: 0, 1, 2

token-level mask (W=4, causal):
        0 1 2 3 4 5 6 7 8 9   (keys →)
    0 [ X . . . . . . . . . ]
    1 [ X X . . . . . . . . ]
    2 [ X X X . . . . . . . ]
    3 [ X X X X . . . . . . ]
    4 [ . X X X X . . . . . ]
    5 [ . . X X X X . . . . ]
    6 [ . . . X X X X . . . ]
    7 [ . . . . X X X X . . ]
        ...
   (queries ↓)
```

The receptive field grows with depth: after `L` stacked SWA layers, information from `L · W` tokens away can reach the current position (analogous to dilated CNNs). Mistral 7B with `W=4096`, `L=32` → ~131K-token theoretical span.

### KV cache (Rolling Buffer)

Because each token only needs the last `W` keys, the cache size is **capped at `W`**, not `n`:

- Per layer: `2 · b · min(n, W) · h_kv · d_h · s`.
- Implementation: ring buffer indexed `i mod W`; old entries are overwritten in place.

Mistral 7B at `n=32K, W=4096` → 8× cache reduction vs uncapped GQA.

### Arithmetic intensity

The `n/d` decode term is replaced by `min(n, W)/d`. For long contexts this is a much bigger win than MQA/GQA: KV-cache memory traffic stops growing entirely once `n > W`.

The compute side also changes: full attention is `O(n²)` per sequence; SWA is `O(n · W)` — linear in `n`. This is mainly a prefill/training win since decode is one query at a time.

### Trade-off

**Pros:**
- Constant per-step KV memory (fixed `W`) — bounded latency at any context length.
- Linear-time prefill instead of quadratic.
- Composes with GQA/MQA multiplicatively (Mistral 7B uses both: GQA-8 *and* SWA).

**Cons:**
- Direct attention to distant tokens is gone — recovered only through `L` layers of stacking. Tasks needing precise long-range copy (retrieval, exact citation) suffer.
- Most production deployments interleave SWA layers with full-attention layers (Gemma-2, GPT-OSS) to recover global lookup capability.
- Pre-fill chunking gets more complex: each chunk must attend over the chunk itself + the rolling cache (see Mistral 7B paper, Fig. 3).

### Models

Longformer (encoder, with task-specific global tokens), Mistral 7B (W=4096), Mixtral, Gemma-2 (alternating SWA/global), GPT-OSS.

---

## How they compose

These are not mutually exclusive. The standard 2024+ recipe is **GQA + SWA + FlashAttention + quantized KV cache**:

| component | what it shrinks |
|---|---|
| GQA (G=8 vs h=32) | KV cache by 4× (head-count axis) |
| SWA (W=4096) | KV cache by `n/W`× once context exceeds window |
| FP8 KV cache | KV cache by 2× (bytes per element) |
| FlashAttention | activation memory + HBM traffic during the attention op itself |

For Mistral 7B at 32K context vs a hypothetical MHA equivalent: ~32× total KV cache reduction before quantization.

---

## Quick decision guide

- **Pretraining a new model from scratch, GPU inference matters?** GQA with `G=8`. Mistral/Llama-3 default. Best quality/speed Pareto.
- **Pretraining for very long context?** GQA + SWA (interleaved with some global layers).
- **Have an existing MHA checkpoint and need fast inference?** Uptrain to GQA — ~5% of pretraining compute recovers near-MHA quality (GQA paper §3.3).
- **Tiny inference budget, willing to accept quality loss?** MQA. But verify training stability if training from scratch.
- **Need bidirectional encoder over long documents?** SWA + a few global tokens (Longformer pattern).

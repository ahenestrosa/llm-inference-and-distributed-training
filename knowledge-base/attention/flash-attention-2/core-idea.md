# FlashAttention 2 - The Core Idea

> The end-to-end picture: FA1 made attention IO-efficient against HBM; FA2 makes it efficient *inside* the GPU by fixing non-matmul overhead, low occupancy, and cross-warp memory traffic.

## Context

Part of the [Flash Attention 2](./flash-attention-2.md) knowledge base. Builds directly on [FlashAttention v1](../flash-attention/flash-attention.md) — read that first if FA1's tiling and online softmax aren't fresh. The GPU-internals detail behind improvement #3 lives in [Warp Communication in FlashAttention 2](./warp-communication.md).

## The starting point: why FA1 wasn't fast enough

FA1 was already 2-4× faster than standard attention, but it was only reaching **25–40% of the GPU's theoretical max FLOPs/s**, while optimized GEMM (matrix multiply) reaches 80–90%. That gap is the motivation for FA2.

The diagnosis: FA1's bottleneck is no longer HBM ↔ SRAM traffic (FA1 already fixed that). It's now **suboptimal work partitioning inside the GPU** — too much time on non-matmul ops, low occupancy on long sequences, and unnecessary shared-memory shuffling between warps.

To make this concrete: on an A100, matmul (Tensor Cores, FP16) runs at **312 TFLOPs/s**, but non-matmul FP32 ops run at **19.5 TFLOPs/s**. So one non-matmul FLOP costs ~16× more wall-clock time than one matmul FLOP. Even if non-matmul ops are a tiny fraction of total FLOPs, they dominate runtime.

This frames the three improvements.

## Improvement 1: Reduce non-matmul FLOPs (algorithmic tweaks)

FA1's online softmax did rescaling at every block step. FA2 keeps the math identical but **defers as much rescaling as possible to the end**.

### The FA1 update (per block j)

At each step you compute the new normalizer `ℓ⁽ʲ⁾` and immediately rescale the output:

```
O⁽ʲ⁾ = diag(ℓ⁽ʲ⁻¹⁾/ℓ⁽ʲ⁾)⁻¹ · O⁽ʲ⁻¹⁾  +  diag(ℓ⁽ʲ⁾)⁻¹ · exp(S⁽ʲ⁾ − m⁽ʲ⁾) · V⁽ʲ⁾
```

That `diag(ℓ⁽ʲ⁾)⁻¹` factor — a division by the running sum-of-exponentials — is a non-matmul op done at every iteration.

### The FA2 fix

Keep an **unnormalized** running output `Õ` throughout the loop. Only divide by `ℓ` once, at the very end:

```
Õ⁽ʲ⁾ = diag(exp(m⁽ʲ⁻¹⁾ − m⁽ʲ⁾))⁻¹ · Õ⁽ʲ⁻¹⁾  +  exp(S⁽ʲ⁾ − m⁽ʲ⁾) · V⁽ʲ⁾

# After the loop finishes:
O = diag(ℓ⁽ᶠⁱⁿᵃˡ⁾)⁻¹ · Õ⁽ᶠⁱⁿᵃˡ⁾
```

You still need to rescale `Õ` when the running max changes (the `exp(m⁽ʲ⁻¹⁾ − m⁽ʲ⁾)` factor — unavoidable, since updating the max changes the exponent baseline), but the `1/ℓ` division now happens **once per row**, not once per block.

### Second tweak — store logsumexp

For the backward pass, FA1 saved both `m` (running max) and `ℓ` (running sum). FA2 stores just one scalar per row: `L = m + log(ℓ)`. Backward recomputes `P = exp(S − L)` directly in one step. Less memory, fewer ops.

> *Intuition:* matmul is cheap on a GPU; division and exponentiation are expensive. Push as much work as possible into the matmul units and amortize the scalar work.

## Improvement 2: Parallelize over sequence length

This is the one with the biggest practical impact for long-context training.

### FA1's parallelism scheme

One thread block per `(batch element, attention head)`. So the total number of thread blocks = `batch_size × num_heads`. On an A100 with 108 SMs, you want ≥ 80 thread blocks to saturate the GPU.

This breaks when sequences are long. Long sequences → small batch size (memory constraint) → not enough thread blocks → idle SMs.

> *Example:* `batch=1, heads=16, seq_len=16k`. That's 16 thread blocks for 108 SMs → ~15% occupancy. The GPU is mostly idle.

### FA2's scheme

Also parallelize over the **sequence length dimension** — specifically, over row blocks of Q. Now total thread blocks = `batch_size × num_heads × num_Q_blocks`.

Same example: with Q-block size 128, that's `1 × 16 × 128 = 2048` thread blocks. Plenty to fill the GPU.

### Why this requires the loop swap

FA1's outer loop was over **K/V blocks** (columns), inner loop over Q blocks (rows). FA2 swaps these: **outer loop over Q blocks (rows), inner loop over K/V blocks**.

Why does this matter? With Q on the outside, each thread block owns one block of output rows and computes them completely independently — no cross-block communication needed for the forward pass. The row blocks are *embarrassingly parallel*. (Credit goes to Phil Tillet's Triton implementation, which figured this out first.)

If you kept K/V on the outside, different thread blocks would all need to update the same output row, requiring synchronization.

### Backward pass

Slightly trickier. The backward pass parallelizes over **column blocks** (K/V) instead. Looking at the gradient equations, `dK` and `dV` for a given K/V block depend only on the full Q, but `dQ` needs contributions from all K/V blocks. So column-parallelism is natural for `dK`/`dV`, and `dQ` uses atomic adds across thread blocks.

## Improvement 3: Better warp partitioning (inside a thread block)

This is the most GPU-internal of the three. Inside each thread block, work is split across 4 or 8 **warps** (groups of 32 threads). The question: how do you slice up the work among warps?

### FA1's "split-K" scheme

- Q is accessible by all warps
- K and V are split across the 4 warps

Each warp computes its slice of `QKᵀ`, but to multiply by V and get the output, warps need to **write intermediate results to shared memory, synchronize, then combine**. That cross-warp communication is the bottleneck.

### FA2's "split-Q" scheme

- Q is split across the 4 warps
- K and V are accessible by all warps

Now each warp computes a full row-slice of `QKᵀ` using its Q slice and the shared K. Then it multiplies by the shared V and gets its own output slice directly — **no inter-warp communication needed**.

The diagram in Fig. 3 of the paper makes this crystal clear. The shift from split-K to split-Q is small in concept but eliminates a lot of shared-memory traffic.

> [!NOTE]
> **Why this works for FA2 but not FA1** — It's tied to the loop swap. When Q is on the outer loop (FA2), all warps in a thread block share the same Q block for the whole duration — so they share K/V (which changes) and split Q (which is fixed). In FA1's column-major iteration, the symmetric situation made K/V the natural thing to split.

For the deep dive on warps, shared memory, and why split-Q geometrically eliminates the cross-warp reduction → see [Warp Communication in FlashAttention 2](./warp-communication.md).

## How it all connects

The three changes reinforce each other:

1. **Loop swap (Q outer, K/V inner)** → enables sequence-length parallelism → fixes occupancy on long sequences.
2. **Loop swap** → also enables split-Q warp partitioning → fixes shared-memory traffic.
3. **Deferred rescaling** → reduces non-matmul FLOPs → makes the matmul units the bottleneck (where they should be).

Combined result: **2× over FA1**, reaching 50–73% of theoretical max throughput on A100, and ~73% in the forward pass. Optimized GEMM hits 80–90%, so FA2 is now close to the practical ceiling for attention.

## Key Insight

FA1 solved the **outer** memory hierarchy problem (HBM ↔ SRAM). FA2 solves the **inner** problem (SRAM ↔ registers, and warp-level work allocation). The single conceptual move that unlocks two of the three improvements is **swapping the loop order**: putting Q on the outside makes row-block parallelism embarrassingly parallel *and* lets warps split Q (independent output rows) instead of K/V (the reduction axis). Combined with deferring softmax rescaling out of the hot loop, attention finally runs at near-GEMM efficiency.

## Related

- [Flash Attention 2](./flash-attention-2.md)
- [FlashAttention v1](../flash-attention/flash-attention.md)
- [FlashAttention - The Core Idea](../flash-attention/core-idea.md)
- [The Online Softmax Trick](../flash-attention/online-softmax-trick.md)
- [Warp Communication in FlashAttention 2](./warp-communication.md)

# FlashAttention - The Core Idea

> The end-to-end picture: standard attention is memory-bound, and FlashAttention removes the memory bottleneck without changing the math.

## Context

Part of the [Flash Attention](./flash-attention.md) knowledge base. This is the main walkthrough; branch to [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md) and [The Online Softmax Trick](./online-softmax-trick.md) for the details that make it work.

## The problem in one sentence

Standard attention computes `S = QKᵀ`, then `P = softmax(S)`, then `O = PV` — and each intermediate matrix `S, P ∈ ℝ^(N×N)` is **written to HBM and read back**. The math is fine; the memory traffic is the problem.

Numbers from the paper (Fig. 2, left; GPT-2 medium, seq len 1024):

| | Standard | FlashAttention |
| --- | --- | --- |
| GFLOPs | 66.6 | **75.2** (more!) |
| HBM R/W | 40.3 GB | **4.4 GB** |
| Runtime | 41.7 ms | **7.3 ms** |

FlashAttention does *more* FLOPs and runs ~6× faster. The thesis: **HBM access dominates runtime, not arithmetic.** Attention has low arithmetic intensity, so it sits on the memory-bound side of the roofline.

## Why naive tiling doesn't work — and how they fix it

The obvious idea is "just tile, like a matmul." The blocker is **softmax**: it normalizes each row across all N columns, so you can't compute it block-by-block without first materializing the full row. → See [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md).

The fix is the **online softmax trick** (Milakov & Gimelshein, 2018): track two running statistics per row — the running max `m` and the running sum-of-exponentials `ℓ` — and apply a rescaling correction when a new block arrives. → See [The Online Softmax Trick](./online-softmax-trick.md).

So tiling becomes possible because you never need the full row at once; you just need two scalars per row to maintain correctness incrementally.

## The tiling scheme

Block sizes are chosen so one block of Q, K, V plus the intermediate `S_ij` fits in SRAM (~192 KB on A100): `B_c = ⌈M/4d⌉`, `B_r = min(⌈M/4d⌉, d)`.

Loop structure (Algorithm 1):

```text
for j in 1..T_c:                    # outer loop over K, V blocks
    load K_j, V_j to SRAM           # ~Bc × d each
    for i in 1..T_r:                # inner loop over Q blocks
        load Q_i, O_i, ℓ_i, m_i to SRAM
        S_ij = Q_i @ K_jᵀ           # Br × Bc, lives on-chip only
        compute m̃, ℓ̃, P̃ on-chip
        update m_i, ℓ_i (online softmax merge)
        update O_i with rescaling   # write back to HBM
```

Key points:

- The N×N matrix `S` is **never materialized in HBM**. Blocks `S_ij` exist only inside SRAM and get discarded.
- The outer loop is over K/V, inner over Q, so each `(K_j, V_j)` is loaded once and reused across all Q blocks.
- After the full sweep, `O` in HBM holds the **exact** attention output — bit-identical to standard attention up to floating-point ordering.

## The IO complexity result

- Standard attention: **Θ(Nd + N²)** HBM accesses — dominated by reading/writing the N×N matrices.
- FlashAttention: **Θ(N²d²/M)** HBM accesses, where M is SRAM size.

For typical values (d = 64–128, M ~ 100 KB), `d²/M ≪ 1`, so FlashAttention does many times fewer HBM accesses — up to **9× fewer** in the paper.

Proof sketch:

- Load each K, V block once → Θ(Nd) total.
- For each K, V block, iterate over all of Q (size Nd). With block size Θ(M/d), there are Θ(Nd/M) outer iterations.
- Total: Θ(Nd) · Θ(Nd/M) = Θ(N²d²/M).

They also prove a **lower bound** (Proposition 3): no exact attention algorithm can do asymptotically better across all SRAM sizes. So this is optimal up to constants.

## Recomputation for the backward pass

The forward pass avoids materializing `S` and `P`. The backward pass normally needs them for gradients. Two options:

1. **Store them** — defeats the purpose, brings back O(N²) memory.
2. **Recompute from Q, K, V plus saved (m, ℓ)** — what FlashAttention does (selective gradient checkpointing).

Counterintuitively, even with **more FLOPs**, the backward pass is faster because it does **fewer HBM accesses**. Total memory drops from O(N²) to O(N) — you keep only `O`, `m`, `ℓ` (each size N), not the full attention matrix.

## Implementation: why it has to be a CUDA kernel

PyTorch can't fuse these ops because each op (matmul, softmax, matmul) writes intermediates to HBM by design. FlashAttention requires a **single fused CUDA kernel**: load Q/K/V blocks, run the full matmul → softmax → matmul → output-update pipeline on-chip, write only the final O back to HBM.

The paper's "Limitations" section flags that writing custom CUDA per attention variant is painful — the gap Triton was later designed to fill (hence FlashAttention-2 and -3 in Triton).

## Key Insight

The entire speedup is a memory-bandwidth story, not an arithmetic one. By making softmax streamable (online softmax) and tiling the computation to fit SRAM, FlashAttention keeps the N×N matrices out of HBM entirely — cutting HBM accesses from Θ(Nd + N²) to Θ(N²d²/M) and memory from O(N²) to O(N), while computing the exact same result.

## Related

- [Flash Attention](./flash-attention.md)
- [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md)
- [The Online Softmax Trick](./online-softmax-trick.md)
- [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md)

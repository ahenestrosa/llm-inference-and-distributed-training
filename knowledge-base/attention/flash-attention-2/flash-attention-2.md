# Flash Attention 2

> [!NOTE]
> Exported from [ahenestrosa](https://github.com/ahenestrosa)'s Obsidian knowledge base. Wikilinks have been rewritten as relative links and Obsidian callouts mapped to GitHub-flavored callouts; content is otherwise unchanged.

## Paper

Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

> [!NOTE]
> **What this is** — Map of content for FlashAttention-2 (Dao, 2023) — a follow-up to [FlashAttention v1](../flash-attention/flash-attention.md) focused on **better parallelism and work partitioning** inside the GPU. Same exact-attention math, but ~2× faster than FA1 by tweaking the algorithm to reduce non-matmul FLOPs, parallelizing over sequence length, and reorganizing how warps split work inside a thread block.

## Core idea in one line

FA1 fixed the **HBM ↔ SRAM** bottleneck. FA2 fixes the **inside-the-GPU** bottleneck: too much time on non-matmul ops, low SM occupancy on long sequences, and unnecessary shared-memory traffic between warps. The result: 50–73% of theoretical max FLOPs/s on A100 (vs. 25–40% for FA1), getting close to the efficiency of optimized GEMM.

> [!NOTE]
> **The headline number** — Up to **225 TFLOPs/s per A100 GPU** for end-to-end GPT-style training (72% model FLOPs utilization). On the attention kernel alone: ~2× over FA1, ~10× over standard PyTorch attention.

## The three improvements

1. **Algorithmic tweaks — reduce non-matmul FLOPs.** Defer the `1/ℓ` rescaling to the end of the loop instead of doing it every block. Store one scalar `L = m + log(ℓ)` instead of two (`m`, `ℓ`) for the backward pass. Motivation: matmul on Tensor Cores runs ~16× faster than non-matmul FP32, so non-matmul ops dominate wall-clock time even though they're a small fraction of total FLOPs.
2. **Parallelize over sequence length.** FA1 parallelized only over `batch × heads` → starves the GPU when sequences are long (small batch). FA2 also parallelizes over Q row blocks, which requires swapping the loop order: outer loop over Q (rows), inner loop over K/V (columns). The Q row blocks are then embarrassingly parallel — one thread block per block, no communication needed.
3. **Better warp partitioning ("split-Q" instead of "split-K").** Inside each thread block, FA1 split K and V across warps, forcing them to combine partial output sums via shared memory. FA2 splits Q across warps and keeps K/V shared, so each warp produces complete output rows independently — no cross-warp reduction.

## Index

- [FlashAttention 2 - The Core Idea](./core-idea.md) — the full walkthrough: motivation, the three improvements, how they connect, why each one matters.
- [Warp Communication in FlashAttention 2](./warp-communication.md) — deep dive on warps, shared memory, and why split-Q eliminates the cross-warp reduction that split-K needed. The GPU-internals story behind improvement #3.

## Reading path

1. Start with [FlashAttention 2 - The Core Idea](./core-idea.md) for the end-to-end picture.
2. When you hit improvement #3 (warp partitioning), branch to [Warp Communication in FlashAttention 2](./warp-communication.md) for the GPU-internals detail.
3. Re-read FA2 §3.3 and Fig. 3 with that context loaded.

## Reference

- Focus sections for a first pass: §1, §3.1–§3.3.
- Builds directly on: [FlashAttention v1](../flash-attention/flash-attention.md) (Dao et al., 2022).
- Triton implementation that first inspired the loop swap and sequence-length parallelism: Phil Tillet's `06-fused-attention.py` in the Triton tutorials.

## Related

- [Flash Attention](../flash-attention/flash-attention.md) — the v1 MOC. FA2 is best read after fully understanding FA1's tiling + online softmax.
- [The Online Softmax Trick](../flash-attention/online-softmax-trick.md) — FA2 keeps the same online softmax, just defers one of its rescaling steps.
- Follow-ups worth a future note: FlashAttention-3 (Hopper async + FP8), how FA2 interacts with MQA/GQA (briefly covered in FA2 §3.1.2).

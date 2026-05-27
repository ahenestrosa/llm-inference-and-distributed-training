# Flash Attention

> [!NOTE]
> Exported from [ahenestrosa](https://github.com/ahenestrosa)'s Obsidian knowledge base. Wikilinks have been rewritten as relative links and Obsidian callouts mapped to GitHub-flavored callouts; content is otherwise unchanged.

## Paper

Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · [PDF](https://arxiv.org/pdf/2205.14135.pdf)

Map of content for FlashAttention (Dao et al., 2022) — an IO-aware, *exact* attention algorithm that avoids materializing the N×N attention matrix in HBM. The math is identical to standard attention; the win is doing far fewer slow memory reads/writes.

## Core idea in one line

Standard attention is **memory-bound**: it writes the N×N score matrix `S` and probability matrix `P` to HBM and reads them back. FlashAttention never materializes them — it **tiles** the computation into blocks that fit in fast on-chip SRAM, and uses the **online softmax trick** to combine block results incrementally. Result: HBM accesses drop from Θ(Nd + N²) to Θ(N²d²/M), memory from O(N²) to O(N), with a large wall-clock speedup.

> [!NOTE]
> GPT-2 medium, seq len 1024 (paper Fig. 2): FlashAttention does *more* FLOPs (75.2 vs 66.6 GFLOPs) yet runs ~6× faster (7.3 ms vs 41.7 ms), because HBM traffic drops from 40.3 GB to 4.4 GB. Memory access dominates runtime, not arithmetic.

## Index

- [FlashAttention - The Core Idea](./core-idea.md) — the full walkthrough: problem, tiling, IO complexity, recomputation, kernel fusion.
- [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md) — why you can't just tile attention like a matmul; softmax couples the whole row.
- [The Online Softmax Trick](./online-softmax-trick.md) — the fix: compute softmax incrementally by carrying a running max and sum, with a rescaling correction.
- [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md) — how the trick plugs into the algorithm, including the output accumulator update.

## Reading path

1. Start with [FlashAttention - The Core Idea](./core-idea.md) for the end-to-end picture.
2. When you hit the tiling claim, branch to [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md).
3. Then [The Online Softmax Trick](./online-softmax-trick.md) for the mathematical fix.
4. Finish with [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md) for how it all assembles into Algorithm 1.

## Reference

- Focus sections for a first pass: §1–§3.
- Key prior art: online softmax (Milakov & Gimelshein, 2018); memory-efficient attention (Rabe & Staats, 2021).

## Related

- [The Online Softmax Trick](./online-softmax-trick.md)
- Follow-ups worth a future note: FlashAttention-2 (loop reorder, warp parallelism), FlashAttention-3 (Hopper async + FP8), PagedAttention (inference-time KV cache management that calls Flash kernels).

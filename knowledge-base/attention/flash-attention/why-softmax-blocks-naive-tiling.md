# Why Softmax Blocks Naive Tiling

> Tiling works for operations whose partial results combine without global information. Softmax needs a global denominator, so naive block-by-block computation gives wrong answers.

## Context

The conceptual hurdle behind [FlashAttention - The Core Idea](./core-idea.md). Resolved by [The Online Softmax Trick](./online-softmax-trick.md).

## First, what does "tiling" mean in matmul?

To multiply `A (M×K) @ B (K×N) = C (M×N)` when the matrices don't fit in SRAM, split them into **blocks** that do fit and compute the output block-by-block:

$$C[i,j] = \sum_k A[i,k] \cdot B[k,j]$$

Each output block is a **sum of independent partial products**. You can compute `A_block1 @ B_block1`, then add `A_block2 @ B_block2`, etc. Each partial product is self-contained — it doesn't need to know about other blocks to be correct. You just accumulate.

> [!NOTE]
> Addition is associative and order-independent. Partial results combine cleanly. This is what makes matmul tileable.

## Now look at softmax — and see why it breaks

Softmax over a row of N values:

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

The denominator sums over **all N values in the row**. That's the problem.

Split a row into two halves `x = [x⁽¹⁾, x⁽²⁾]`. If you naively softmax just the first half:

$$\text{softmax}_{\text{local}}(x^{(1)}_i) = \frac{\exp(x^{(1)}_i)}{\sum_j \exp(x^{(1)}_j)}$$

The denominator is **wrong** — it only sums over the first half. The true denominator should include the second half too. So the "partial softmax" isn't a partial result you can later combine; it's simply incorrect. Unlike matmul, where partial sums add up to the truth, partial softmaxes have the wrong denominator baked into every entry.

## Why this matters for attention specifically

Attention has three steps:

```text
S = Q @ Kᵀ      (matmul — tileable, no problem)
P = softmax(S)  (row-wise softmax — the blocker)
O = P @ V       (matmul — tileable, no problem)
```

Steps 1 and 3 tile beautifully. Step 2 in the middle is the wall, and you can't skip it: `O[i,:]` depends on `P[i,:]`, which depends on **all of `S[i,:]`** (the whole row), because softmax couples them through the denominator.

The naive picture:

```text
For query block Q_i:
  - Load Q_i ✓
  - To compute softmax for row i, I need... all of K
  - That means materializing the full row S[i, :] of length N
  - And I need that for every Q block
  - So I'm back to O(N²) memory in HBM
```

This is why earlier "memory-efficient attention" work (Rabe & Staats, 2021) existed but didn't give wall-clock speedup — it reduced the memory *footprint* but still had the quadratic memory-*access* pattern.

## The way out (preview)

Softmax *can* be computed incrementally if you carry extra state. You don't need the full row at once — just **two scalars per row**: the running max and the running sum-of-exponentials. When a new block arrives, you retroactively "fix up" your partial result with a rescaling factor. That's [The Online Softmax Trick](./online-softmax-trick.md).

## Key Insight

Tiling works when partial results combine without global information (matmul, addition, max). Softmax by default needs the global denominator, so partial results are "wrong" in a way simple addition doesn't fix. FlashAttention's contribution is keeping just enough bookkeeping (running max + sum) to make softmax tileable too — turning a non-decomposable operation into a decomposable-with-corrections one.

## Related

- [FlashAttention - The Core Idea](./core-idea.md)
- [The Online Softmax Trick](./online-softmax-trick.md)
- [Flash Attention](./flash-attention.md)

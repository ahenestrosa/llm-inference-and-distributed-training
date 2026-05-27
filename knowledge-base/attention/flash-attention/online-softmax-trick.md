# The Online Softmax Trick

> The mathematical heart of FlashAttention: compute softmax incrementally by carrying a running max and sum, correcting old partial results with a single rescaling factor.

## Context

The fix for the problem in [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md). How it plugs into the algorithm is in [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md).

## 1. Start with numerically stable softmax

The textbook formula overflows: if any `x_i` is large (say 100), `exp(100) ≈ 2.7 × 10⁴³` overflows FP16 and strains FP32. Standard fix — **max-subtraction**:

$$\text{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max(x)$$

Mathematically identical (numerator and denominator both multiplied by `exp(-m)`), but now the largest exponent is `exp(0) = 1`, so nothing overflows. Every real implementation does this.

Three quantities define a softmax over vector x:

- `m(x) = max(x)` — the max
- `f(x) = [exp(x_1 - m), ..., exp(x_B - m)]` — the shifted exponentials
- `ℓ(x) = Σ f(x)` — the sum of shifted exponentials (the denominator)

Final answer: `softmax(x) = f(x) / ℓ(x)`.

## 2. The setup: I have two blocks, what do I do?

Split a row into `x⁽¹⁾` and `x⁽²⁾` (each size B). After processing `x⁽¹⁾` I've stored:

- `m⁽¹⁾ = max(x⁽¹⁾)`
- `ℓ⁽¹⁾ = Σ exp(x⁽¹⁾ - m⁽¹⁾)` — shifted by the **local** max

Now `x⁽²⁾` arrives with its own `m⁽²⁾`, `ℓ⁽²⁾`. I want the combined `m`, `ℓ` **without recomputing from scratch**.

New max is easy:

$$m = \max(m^{(1)}, m^{(2)})$$

The sum is where it gets interesting.

## 3. The rescaling insight

`ℓ⁽¹⁾` was shifted by `m⁽¹⁾`, but the true combined softmax must shift by the new global `m`. So `ℓ⁽¹⁾` is normalized against the wrong max. Fix it with one multiplication:

$$\exp(x_i - m) = \exp(x_i - m^{(1)} + m^{(1)} - m) = \exp(m^{(1)} - m)\cdot\exp(x_i - m^{(1)})$$

Therefore:

$$\sum \exp(x^{(1)} - m) = \exp(m^{(1)} - m) \cdot \ell^{(1)}$$

That `exp(m⁽¹⁾ - m)` is the **rescaling factor**. It's always `≤ 1` (since `m ≥ m⁽¹⁾`), shrinking the old sum to match the new, larger max. Same logic for block 2:

$$\ell = \exp(m^{(1)} - m)\cdot\ell^{(1)} + \exp(m^{(2)} - m)\cdot\ell^{(2)}$$

> [!NOTE]
> Compute softmax incrementally by (1) carrying two scalars per row (`m`, `ℓ`), and (2) updating them with a rescaling correction whenever a new block arrives. You never need the full row at once. The "wrongness" of partial results is exactly correctable by one multiplication.

## 4. Concrete example

Row `x = [1, 5, 3, 7]`, split into two blocks of size 2.

**Block 1**: `x⁽¹⁾ = [1, 5]`
- `m⁽¹⁾ = 5`
- shifted exps: `[exp(-4), exp(0)] = [0.0183, 1]`
- `ℓ⁽¹⁾ = 1.0183`

**Block 2**: `x⁽²⁾ = [3, 7]`
- `m⁽²⁾ = 7`
- shifted exps: `[exp(-4), exp(0)] = [0.0183, 1]`
- `ℓ⁽²⁾ = 1.0183`

**Merge**:
- `m = max(5, 7) = 7`
- Rescale block 1: `exp(5 - 7) · 1.0183 = 0.1353 · 1.0183 = 0.1378`
- Rescale block 2: `exp(7 - 7) · 1.0183 = 1 · 1.0183 = 1.0183`
- `ℓ = 0.1378 + 1.0183 = 1.1561`

**Verify directly**: `Σ exp(x_i - 7) = exp(-6) + exp(-2) + exp(-4) + exp(0) = 0.0025 + 0.1353 + 0.0183 + 1 = 1.1561` ✓

Exact same answer as recomputing from scratch, but only needed two scalars from block 1.

## 5. What this enables in FlashAttention

Each Q-block row keeps running `(m_i, ℓ_i)` in HBM (small — N values total). When a new K, V block is processed: compute the local `S_ij` (in SRAM, never written to HBM), compute its row-wise `m̃_ij`, `ℓ̃_ij`, merge with the existing `(m_i, ℓ_i)`, and apply the **same rescaling** to the running output `O_i`.

That last step — rescaling the output accumulator — is the subtle one. The full derivation is in [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md).

## Key Insight

Softmax looks non-decomposable because it needs a global denominator. The online softmax trick shows it *is* decomposable, provided you keep two scalars per row and apply the correction `exp(old_max - new_max)` when merging. This converts a "see-everything-first" operation into a streaming one — and that is what unlocks all of FlashAttention's tiling.

## Related

- [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md)
- [What the Online Softmax Trick Enables in FlashAttention](./online-softmax-in-flashattention.md)
- [FlashAttention - The Core Idea](./core-idea.md)
- [Flash Attention](./flash-attention.md)

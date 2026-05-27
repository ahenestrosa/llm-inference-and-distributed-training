# What the Online Softmax Trick Enables in FlashAttention

> How the running max/sum from [The Online Softmax Trick](./online-softmax-trick.md) plugs into the algorithm — including the output accumulator update, which needs the same rescaling that `m` and `ℓ` did.

## Context

Follow-up to [The Online Softmax Trick](./online-softmax-trick.md). Assumes you know the merge formula for `m` and `ℓ`; here we extend it to the output vector `O`.

## Stage 0: What we're actually computing

Attention for a single query row `i`:

$$O_i = \sum_j \text{softmax\_weight}(i,j)\cdot V_j = \sum_j \frac{\exp(S_{ij} - m_i)}{\ell_i} V_j$$

where `S_ij = Q_i · K_jᵀ` is the score, `m_i` the row max, `ℓ_i` the row's sum-of-exponentials. The output for query `i` is a **weighted average of all value vectors**, weighted by softmax.

> [!NOTE]
> Query row `i` comes from token `i`'s embedding (projected through `W_Q`), and `O_i` is the length-`d` vector token `i` produces after attending to all the tokens it can see. The index `i` ranges over the sequence (the N query rows = N tokens); `j` ranges over keys (also tokens — the ones being attended to). One row in, one row out, per token.
>
> Caveat: this is **per attention head**. Each head has its own Q, K, V projections, so token `i` produces one `O_i` per head, concatenated across heads before the output projection. Within a single head, row `i` is token `i`.

All three quantities (`S_ij` for all j, `m_i`, `ℓ_i`) are global over the row — that's the coupling problem. We want to compute `O_i` while only ever seeing one block of keys/values at a time.

## Stage 1: Mental model — a running average that keeps getting corrected

You're computing a weighted average, but the weights keep changing as new data arrives (because `ℓ` and `m` keep updating). Every update means your accumulated work was based on stale values, so you retroactively scale it.

> *Analogy:* computing a running average, but occasionally someone says "multiply everything you've summed so far by 0.8 before continuing." You don't recompute — you apply the scaling factor to your accumulator.

The `m`/`ℓ` updates were the easy version (two scalars). The output `O_i` is the real version — a whole vector (size `d`), and the thing we actually care about.

## Stage 2: What we store, and where

Per query row `i`, kept in HBM (all small — proportional to N, not N²):

| Quantity | Size per row | Meaning |
| --- | --- | --- |
| `m_i` | 1 scalar | running max of scores seen so far |
| `ℓ_i` | 1 scalar | running sum-of-exps (denominator-in-progress) |
| `O_i` | `d` values | running output accumulator |

Initialized to `m_i = -∞`, `ℓ_i = 0`, `O_i = 0`. Total is **O(N)**, never O(N²). The `S_ij` and `P_ij` blocks live only in SRAM and are discarded.

## Stage 3: One inner-loop step in slow motion

> [!NOTE]
> A **slice of consecutive tokens from K and V** — `Bc` of them (e.g. 64–256). Since K, V are each `N × d` (one row per token), block `j` is rows `j·Bc … (j+1)·Bc − 1`:
> ```text
> K_j = K[j·Bc : (j+1)·Bc, :]   # shape: Bc × d
> V_j = V[j·Bc : (j+1)·Bc, :]   # shape: Bc × d
> ```
> "A new block j arrives" = you load the next chunk of keys/values into SRAM. Each block contributes the attention from *that group of tokens* to the running output. After all `Tc = N/Bc` blocks, query `i` has attended to every key — the full row, assembled piece by piece.
>
> Note: a block `j` is reused across **all** query rows in the current Q block (`Br` of them), since those queries attend to the same keys. That reuse is why loading `K_j, V_j` once into SRAM pays off. (It is chunked by **rows = tokens**, the N-sized axis; the `d` axis is never split, or you'd break the dot product.)

**Step A — local scores (in SRAM):**
```text
S_ij = Q_i @ K_jᵀ        # Br × Bc, stays on-chip, never to HBM
```

**Step B — this block's local statistics:**
```text
m̃_ij = rowmax(S_ij)              # local max, just for this block
P̃_ij = exp(S_ij - m̃_ij)         # shifted exponentials (numerator pieces)
ℓ̃_ij = rowsum(P̃_ij)             # local sum
```
`P̃_ij` is shifted by the **local** max (we don't yet know the global max). Everything here is provisional.

**Step C — update running max and denominator:**
```text
m_new = max(m_i, m̃_ij)
ℓ_new = exp(m_i - m_new)·ℓ_i  +  exp(m̃_ij - m_new)·ℓ̃_ij
         └──── correct old ────┘   └──── correct new ────┘
```
Exactly the merge formula from [The Online Softmax Trick](./online-softmax-trick.md).

## Stage 4: The output update — the part that needs care

**What `O_i` currently holds** — a *normalized* weighted sum, already divided by the old `ℓ_i` and shifted by the old `m_i`:

$$O_i = \frac{1}{\ell_i}\sum_{\text{prev}}\exp(S - m_i)\,V$$

**The problem** — two things are stale: it's divided by `ℓ_i` (should be `ℓ_new`), and shifted by `m_i` (should be `m_new`).

**Fix in three moves.**

*Move 1 — un-normalize.* Multiply by `ℓ_i` to recover the raw weighted sum:
$$\ell_i\, O_i = \sum_{\text{prev}}\exp(S - m_i)\,V$$

*Move 2 — rescale to the new max.* Multiply by `exp(m_i - m_new)`:
$$\exp(m_i - m_{\text{new}})\,\ell_i\, O_i = \sum_{\text{prev}}\exp(S - m_{\text{new}})\,V$$

*Move 3 — add the new block, shifted to the new max:*
$$\dots + \exp(\tilde m_{ij} - m_{\text{new}})\,\tilde P_{ij}\,V_j$$

**Re-normalize** by dividing by `ℓ_new`:

```text
O_i ← (1/ℓ_new) · [ exp(m_i - m_new)·ℓ_i·O_i  +  exp(m̃_ij - m_new)·P̃_ij·V_j ]
        └─renorm─┘   └──── corrected old accumulator ────┘   └──── new block ────┘
```

This is **line 12 of Algorithm 1**. Every piece has meaning:
- `diag(ℓ_i)·exp(m_i - m_new)·O_i` — old work, un-normalized and rescaled
- `exp(m̃_ij - m_new)·P̃_ij·V_j` — new block's contribution, rescaled
- `diag(ℓ_new)⁻¹` — apply the now-correct normalization

Then save `ℓ_i ← ℓ_new`, `m_i ← m_new` for the next step.

## Stage 5: Why it's correct at the end

After the **last** block, `m_i` is the true global max and `ℓ_i` the true global denominator, so the final `O_i` is normalized correctly. The paper proves this by induction (Appendix C, Theorem 1): after `j` blocks, `O_i` exactly equals attention over the first `j` blocks. At the last block you get exact full-sequence attention — **bit-for-bit** the same as standard attention up to FP ordering. That's why it's called *exact* attention, not an approximation.

## Stage 6: Why this is the whole game

Standard attention is slow because `S` and `P` (both N×N) get written to and read from HBM. The online softmax trick lets us avoid creating them there:

- `S_ij` blocks: computed in SRAM, used, discarded.
- `P_ij` blocks: computed in SRAM, used, discarded.
- Only `O_i`, `m_i`, `ℓ_i` persist — O(N), tiny.

```text
online softmax  →  softmax becomes streamable
                →  S, P never materialized in HBM
                →  HBM accesses drop dramatically
                →  wall-clock speedup (memory-bound bottleneck removed)
```

Every link depends on the first one.

## Key Insight

The output accumulator `O_i` needs the same `exp(old_max - new_max)` correction as `m` and `ℓ` — un-normalize, rescale to the new max, add the new block, re-normalize. Applying that correction consistently to `m`, `ℓ`, and `O` is what lets FlashAttention stream over key/value blocks while producing exact attention with only O(N) extra memory.

## Related

- [The Online Softmax Trick](./online-softmax-trick.md)
- [Why Softmax Blocks Naive Tiling](./why-softmax-blocks-naive-tiling.md)
- [FlashAttention - The Core Idea](./core-idea.md)
- [Flash Attention](./flash-attention.md)

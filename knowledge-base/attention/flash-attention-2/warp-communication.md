# Warp Communication in FlashAttention 2

> The GPU-internals story behind FA2's third improvement: what a warp is, where K and V actually live, and why split-Q eliminates the cross-warp reduction that split-K forced.

## Context

Part of the [Flash Attention 2](./flash-attention-2.md) knowledge base. This expands on improvement #3 ("better warp partitioning") from [FlashAttention 2 - The Core Idea](./core-idea.md). Read this when the diagram in FA2 Fig. 3 looks like it makes sense but you can't yet explain *mechanically* why split-Q is faster than split-K.

## What is a warp?

A **warp is a group of 32 threads that execute the same instruction at the same time, in lockstep**. It's the smallest unit of scheduling on an NVIDIA GPU.

When you launch a CUDA kernel with thousands of threads, the hardware schedules them in packs of 32. Those 32 threads share a program counter and step through the code together. This is called **SIMT** (Single Instruction, Multiple Threads): one instruction, many data items, executed simultaneously.

### The hierarchy

```
GPU
 └── Streaming Multiprocessors (SMs)         [108 on A100]
      └── Thread Blocks                       [scheduled onto SMs]
           └── Warps (32 threads each)        [the scheduling unit]
                └── Threads                    [the programming abstraction]
```

When FA2 says "we use 4 or 8 warps per thread block," it means each thread block contains 128 or 256 threads, organized into groups of 32.

### Communication levels (fast to slow)

| Level | Mechanism | Notes |
| --- | --- | --- |
| Within a warp | shuffle instructions (`__shfl_sync`) | Threads read each other's registers directly. ~1 cycle. |
| Within a thread block (across warps) | shared memory + `__syncthreads()` | Fast (~19 TB/s on A100), but requires SRAM write + barrier. |
| Across thread blocks | global HBM memory + kernel boundaries | Slow. |

This hierarchy is the entire reason warp partitioning matters: forcing warps to communicate via shared memory is much slower than letting them work independently.

### Tensor Cores and warps

Tensor Cores operate at the **warp level**. A single `mma` (matrix multiply-accumulate) instruction is issued by one warp and computes a small matrix multiply (e.g., 16×16×16 in FP16) using all 32 threads cooperatively. So warps are the natural unit for matmul-heavy kernels like FlashAttention.

## Where do K and V actually live?

"Accessible by all warps" in FA2 means K and V are loaded into the thread block's **shared memory (SRAM)** — the on-chip scratchpad that all warps within the same thread block can read from.

### Memory hierarchy refresher

```
HBM (global memory)     ~80 GB     ~2 TB/s        slow, shared by entire GPU
   ↓ (kernel loads)
SRAM (shared memory)    ~192 KB    ~19 TB/s       fast, shared within ONE thread block
   ↓ (warps read)
Registers               per-thread, instant       private to each thread
```

Every thread block gets its own private chunk of SRAM (~100 KB usable on A100). That chunk is **visible to all warps in that thread block** — they all read and write the same addresses. Thread blocks on other SMs have their own separate SRAM and can't see this one.

### What "accessible by all warps" means mechanically

When the FA2 kernel runs, the prologue of each thread block does something like:

```text
1. Load Q block from HBM → SRAM   (one copy, visible to all warps)
2. Load K block from HBM → SRAM   (one copy, visible to all warps)
3. Load V block from HBM → SRAM   (one copy, visible to all warps)
4. __syncthreads()                 (wait until loads finish)
5. Warps start computing...
```

After step 4, **all four warps can issue load instructions that read from the K and V regions of SRAM**. They're reading the same physical bytes — there's only one copy of K and V in this thread block's SRAM, not four.

### "Shared" vs "split" is logical, not physical

When something is "split across warps," it doesn't necessarily mean it lives in a different memory — it usually still sits in SRAM. The difference is **which warp is responsible for reading and using which portion**:

- **Shared/accessible by all**: one SRAM region, every warp reads from any part of it.
- **Split across warps**: one SRAM region (or sometimes per-warp registers), but warp 1 only touches rows 0–31, warp 2 only touches rows 32–63, etc.

The "split" is a **logical division of labor**, not necessarily a physical separation.

## The real question: what does each warp PRODUCE?

This is where the split-K vs split-Q distinction becomes mechanical. The cross-warp communication problem isn't about reading inputs — K and V don't change. It's about **whether each warp's output is a complete piece of the final answer or just a partial sum that needs to be combined with other warps' results**.

### Split-K trace (FA1)

Setup in SRAM:

- Q: full block, all warps can read
- K: full block in SRAM, but **logically** warp `i` is responsible for columns `[i·Bc/4 : (i+1)·Bc/4]`
- V: full block in SRAM, but **logically** warp `i` is responsible for rows `[i·Bc/4 : (i+1)·Bc/4]`

**Step 1 — Compute QKᵀ:**

```text
Warp 1: S₁ = Q · K₁ᵀ   → shape [Br, Bc/4]   (columns 0 to Bc/4 of S)
Warp 2: S₂ = Q · K₂ᵀ   → shape [Br, Bc/4]   (columns Bc/4 to 2Bc/4 of S)
Warp 3: S₃ = Q · K₃ᵀ   → ...
Warp 4: S₄ = Q · K₄ᵀ   → ...
```

Each warp ends up with a **vertical slice of columns** of the attention score matrix. So far, no communication needed.

**Step 2 — Softmax:**

Softmax is row-wise and needs the **entire row** of S to compute rowmax and rowsum. But each warp only has a column slice. → Warps must combine their stats across warps via shared memory. **Sync #1.**

**Step 3 — Multiply by V (the real bottleneck):**

```text
Warp 1: O₁ = P₁ · V₁   → shape [Br, d]   ✗ INCOMPLETE — partial sum
Warp 2: O₂ = P₂ · V₂   → shape [Br, d]   ✗ INCOMPLETE — partial sum
Warp 3: O₃ = P₃ · V₃   → ...
Warp 4: O₄ = P₄ · V₄   → ...

Final answer:  O = O₁ + O₂ + O₃ + O₄
```

Each `Oᵢ` is a complete `[Br, d]` matrix, but each is just **one term of a 4-term sum**. None of them is the real answer alone. To finish:

```text
1. Each warp writes its [Br, d] partial output to shared memory   → 4 SRAM writes
2. __syncthreads()                                                  → barrier (Sync #2)
3. One warp reads all four partials and sums them                  → 4 SRAM reads + adds
```

That's the bottleneck. The actual matmul work is done by Tensor Cores in a few cycles, but then you pay for shared-memory round-trips, barriers, and extra additions.

### Split-Q trace (FA2)

Setup in SRAM — **same physical layout**:

- Q: full block in SRAM, but **logically** warp `i` owns rows `[i·Br/4 : (i+1)·Br/4]`
- K: full block in SRAM, all warps read freely
- V: full block in SRAM, all warps read freely

What each warp computes:

```text
Warp 1:
  - reads its Q rows (rows 0 to Br/4) and full K
  - computes S₁ = Q₁ · Kᵀ   → shape [Br/4, Bc]   ✓ COMPLETE rows of S
  - applies softmax row-wise on its own rows: P₁ = softmax(S₁)
    → no communication needed, each row is self-contained ✓
  - reads P₁ and full V
  - computes O₁ = P₁ · V   → shape [Br/4, d]   ✓ COMPLETE rows of O
  - writes O₁ to its designated slot in the output
```

Each warp produces **complete, disjoint rows of the final output**. No partial sums. No combination step. No sync.

## The geometric intuition (why this happens)

The matmul `P · V` reduces along the inner dimension. Look at where the split lands relative to that reduction:

```text
       P [Br × Bc]    ·    V [Bc × d]    =   O [Br × d]
              ↑                ↑
         columns split     rows split
         (the SAME dimension that gets summed over)
```

**Split-K** splits along the reduction axis → each warp produces a partial sum → must combine.
**Split-Q** splits along an *independent* axis (output rows) → each warp produces complete output elements → done.

### A tiny concrete example

4 warps, output O of shape [4, 4], summing over 4 elements:

**Split-K** (split the reduction axis):

```text
Each warp computes:  warp_i_output[r,c] = P[r, i] · V[i, c]   for one i

Warp 1 produces:  | a₁ a₁ a₁ a₁ |   (column 1's contribution to every output element)
                  | a₁ a₁ a₁ a₁ |
                  | a₁ a₁ a₁ a₁ |
                  | a₁ a₁ a₁ a₁ |

Final O[r,c] = warp1[r,c] + warp2[r,c] + warp3[r,c] + warp4[r,c]
→ Must add all 4 warps' matrices element-wise. SYNC + REDUCE.
```

**Split-Q** (split the output rows):

```text
Each warp computes:  O[i, :] = P[i, :] · V   for one row i

Warp 1 produces:  | b b b b |   ← complete row 0 of O
Warp 2 produces:  | b b b b |   ← complete row 1 of O
Warp 3 produces:  | b b b b |   ← complete row 2 of O
Warp 4 produces:  | b b b b |   ← complete row 3 of O

Final O is just stacking these. NO COMBINATION NEEDED.
```

Same K, same V, same SRAM layout. The only difference is **which dimension you split work along** and how that dimension relates to the reduction inside the matmul.

## Why "logical division of labor" still changes sync behavior

The natural confusion: if K and V don't change between FA1 and FA2, why does sync change at all?

The answer: **the logical split determines what each warp computes — and that determines whether the warp's result is a complete piece of the output or just a partial contribution.**

K and V sitting in SRAM doesn't cause syncs. What causes syncs is needing to **combine partial results from different warps into final outputs**. Split-K forces partial outputs (because the split aligns with the reduction axis). Split-Q produces complete outputs (because the split is orthogonal to the reduction axis).

## Key Insight

The split-K vs split-Q distinction is fundamentally about **how the work split aligns with the reduction axis of the matmul `P · V`**. Splitting along the reduction axis (split-K) means every warp produces an incomplete sum that must be combined via shared memory and a barrier. Splitting along an orthogonal axis (split-Q) means every warp produces complete, disjoint output rows that can be written directly. K and V's location in SRAM is identical in both cases — what differs is whether the geometry of the split forces a cross-warp reduction or not.

## Related

- [Flash Attention 2](./flash-attention-2.md)
- [FlashAttention 2 - The Core Idea](./core-idea.md)
- [FlashAttention - The Core Idea](../flash-attention/core-idea.md)
- [The Online Softmax Trick](../flash-attention/online-softmax-trick.md)

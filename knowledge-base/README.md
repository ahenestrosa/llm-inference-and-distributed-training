# Knowledge Base

> [!NOTE]
> Exported notes from [ahenestrosa](https://github.com/ahenestrosa)'s Obsidian knowledge base, curated alongside this repository's code. Wikilinks have been rewritten as relative markdown links and Obsidian callouts mapped to GitHub-flavored callouts; content is otherwise unchanged.

Each topic lives in its own folder with a top-level map-of-content (MOC) note that links to the deeper notes in a recommended reading order.

## Topics

### Attention

- [Flash Attention](./attention/flash-attention/flash-attention.md) — IO-aware exact attention (Dao et al., 2022). Tiling + online softmax to keep the N×N matrix out of HBM.
- [Flash Attention 2](./attention/flash-attention-2/flash-attention-2.md) — Faster attention via better parallelism and work partitioning (Dao, 2023). Loop swap, sequence-length parallelism, split-Q warp partitioning.

## How to read these

1. Open the MOC for a topic (linked above).
2. Follow its "Reading path" section — notes are written to be read in a specific order, branching to detail notes when a concept needs more depth.
3. The "Related" section at the bottom of every note links back out to the wider graph.

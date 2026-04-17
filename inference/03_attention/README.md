# Attention Visualizer

A minimal, tensor2tensor-style bipartite attention visualizer for HuggingFace
causal LMs. Tokens appear on both sides; lines connect queries (left) to keys
(right), with opacity ∝ attention weight.

Controls:
- **Layer**: single-select radio buttons.
- **Heads**: multi-select checkboxes, one color per head (matches the swatch
  row at the top).
- **Hover**: pointing at a left token isolates its outgoing edges; pointing at
  a right token isolates its incoming edges. The right-hand heat column shows
  the resulting attention distribution (averaged over selected heads), or the
  column-sum when nothing is hovered.

## Usage

```bash
python visualize_attention.py --model gpt2 \
    --text "The animal didn't cross the street because it was too tired."
```

Any HF causal LM that supports `attn_implementation="eager"` works
(`gpt2`, `meta-llama/Llama-3.1-8B`, ...). Small models are recommended for
interactive use — attention is `(layers, heads, seq, seq)` floats held in RAM.

## Notes

- `output_attentions=True` requires the eager attention path; SDPA / FlashAttn
  return `None`.
- Rows of the attention matrix are queries, columns are keys. For causal
  models, weights above the diagonal are zero.

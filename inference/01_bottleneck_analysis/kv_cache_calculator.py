"""
KV-Cache Size Calculator
========================
Computes KV-cache memory for various LLMs across sequence lengths.
Loads model configs from HuggingFace (no weights downloaded).

Formula (per sequence, batch_size=1):
    KV cache = 2 (K+V) x num_layers x seq_len x num_kv_heads x head_dim x bytes_per_element
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from transformers import AutoConfig

# ── Models to analyze ─────────────────────────────────────────────────────
#                        (hf_id, num_params)
MODELS = {
    "Llama 3.1 8B":  ("meta-llama/Llama-3.1-8B",  8.03e9),
    "Llama 3.1 70B": ("meta-llama/Llama-3.1-70B", 70.55e9),
    "Mistral 7B":    ("mistralai/Mistral-7B-v0.1",  7.24e9),
}

GPU_MEMORY_GIB = 80  # A100 80GB

SEQ_LENGTHS = [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576]

DTYPE = "bfloat16"  # 2 bytes per element
BYTES_PER_ELEM = 2


def kv_cache_bytes(num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int) -> int:
    """KV-cache size in bytes for a single sequence (batch_size=1)."""
    return 2 * num_layers * seq_len * num_kv_heads * head_dim * BYTES_PER_ELEM


def fmt(n_bytes: int) -> str:
    gib = n_bytes / (1024**3)
    return f"{gib:.2f} GiB" if gib >= 1.0 else f"{n_bytes / (1024**2):.2f} MiB"


# ── 1. Load configs from HuggingFace (config only, no weights) ────────────
models = {}
for name, (model_id, num_params) in MODELS.items():
    print(f"Loading config: {name} ({model_id})")
    cfg = AutoConfig.from_pretrained(model_id)

    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Model weight memory in bfloat16: num_params * 2 bytes
    weight_gib = (num_params * BYTES_PER_ELEM) / (1024**3)

    models[name] = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "weight_gib": weight_gib,
    }
    print(f"  layers={num_layers}  kv_heads={num_kv_heads}  head_dim={head_dim}  weights={weight_gib:.1f} GiB")

# ── 2. Compute KV-cache sizes & build table ───────────────────────────────
rows = []
for seq_len in SEQ_LENGTHS:
    row = {"seq_len": seq_len}
    for name, p in models.items():
        row[name] = kv_cache_bytes(p["num_layers"], p["num_kv_heads"], p["head_dim"], seq_len)
    rows.append(row)

df = pd.DataFrame(rows).set_index("seq_len")

# Pretty-print
print(f"\nKV-Cache Size per Sequence (batch_size=1, dtype={DTYPE})")
print("=" * 72)
display = df.copy()
for col in display.columns:
    display[col] = display[col].apply(fmt)
print(display.to_string())

# ── 3. Serving capacity: max concurrent sequences on a single GPU ─────────
# Available memory for KV cache = GPU memory - model weights
# Max concurrent sequences = available_memory / kv_cache_per_sequence
print(f"\nMax Concurrent Sequences on {GPU_MEMORY_GIB} GiB GPU (dtype={DTYPE})")
print("=" * 72)

capacity_rows = []
for seq_len in SEQ_LENGTHS:
    row = {"seq_len": seq_len}
    for name, params in models.items():
        available_gib = GPU_MEMORY_GIB - params["weight_gib"]
        if available_gib <= 0:
            row[name] = "OOM"  # model doesn't fit on this GPU
        else:
            kv_per_seq = kv_cache_bytes(
                params["num_layers"], params["num_kv_heads"], params["head_dim"], seq_len
            )
            max_seqs = int(available_gib * (1024**3) / kv_per_seq)
            row[name] = max_seqs
    capacity_rows.append(row)

capacity_df = pd.DataFrame(capacity_rows).set_index("seq_len")
print(capacity_df.to_string())

# ── 4. Plots ──────────────────────────────────────────────────────────────
styles = [
    {"marker": "o", "linestyle": "-"},
    {"marker": "s", "linestyle": "-"},
    {"marker": "^", "linestyle": "--"},
]
seq_arr = np.array(SEQ_LENGTHS)
fmt_x = ticker.FuncFormatter(lambda x, _: f"{int(x):,}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ── Plot 1: KV-cache size per sequence ────────────────────────────────────
for i, name in enumerate(models):
    ax1.plot(seq_arr, df[name].values / (1024**3), linewidth=2, label=name, markersize=7, **styles[i])

ax1.set_xlabel("Sequence Length (tokens)")
ax1.set_ylabel("KV-Cache Size (GiB)")
ax1.set_title(f"KV-Cache per Sequence ({DTYPE})")
ax1.set_xscale("log", base=2)
ax1.xaxis.set_major_formatter(fmt_x)
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Max concurrent sequences on GPU ───────────────────────────────
for i, name in enumerate(models):
    col = capacity_df[name]
    # Skip models that don't fit (OOM)
    if (col == "OOM").all():
        continue
    values = col.replace("OOM", 0).astype(int)
    ax2.plot(seq_arr, values, linewidth=2, label=name, markersize=7, **styles[i])

ax2.set_xlabel("Sequence Length (tokens)")
ax2.set_ylabel("Max Concurrent Sequences")
ax2.set_title(f"Serving Capacity on {GPU_MEMORY_GIB} GiB GPU ({DTYPE})")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.xaxis.set_major_formatter(fmt_x)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "inference/01_bottleneck_analysis/kv_cache_size.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()

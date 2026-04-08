"""
Compare quantization benchmark results across all methods.

Produces:
    1. A pandas summary table printed to stdout.
    2. A grouped bar chart (memory, throughput, perplexity) saved as PNG.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def load_results() -> pd.DataFrame:
    """Load all result JSON files into a single DataFrame."""
    rows: list[dict] = []

    # --- BF16 baseline ---
    with open(RESULTS_DIR / "results_baseline_bf16.json") as f:
        data = json.load(f)
    rows.append(
        {
            "method": "BF16 (baseline)",
            "memory_gb": data["gpu_memory"]["max_allocated_gb"],
            "throughput_tok_s": data["throughput"]["median_tokens_per_sec"],
            "perplexity": data["perplexity"],
            "model_size_gb": data["model_size"]["total_size_gb"],
        }
    )

    # --- bitsandbytes (multiple configs) ---
    with open(RESULTS_DIR / "results_bnb.json") as f:
        data = json.load(f)
    for cfg_name, cfg in data["configs"].items():
        rows.append(
            {
                "method": f"BnB {cfg_name}",
                "memory_gb": cfg["gpu_memory"]["max_allocated_gb"],
                "throughput_tok_s": cfg["throughput"]["median_tokens_per_sec"],
                "perplexity": cfg["perplexity"],
                "model_size_gb": cfg["model_size"]["total_size_gb"],
            }
        )

    # --- GPTQ ---
    with open(RESULTS_DIR / "results_gptq.json") as f:
        data = json.load(f)
    rows.append(
        {
            "method": "GPTQ 4-bit",
            "memory_gb": data["gpu_memory"]["max_allocated_gb"],
            "throughput_tok_s": data["throughput"]["median_tokens_per_sec"],
            "perplexity": data["perplexity"],
            "model_size_gb": data["model_size"]["total_size_gb"],
        }
    )

    # --- AWQ ---
    with open(RESULTS_DIR / "results_awq.json") as f:
        data = json.load(f)
    rows.append(
        {
            "method": "AWQ 4-bit",
            "memory_gb": data["gpu_memory"]["max_allocated_gb"],
            "throughput_tok_s": data["throughput"]["median_tokens_per_sec"],
            "perplexity": data["perplexity"],
            "model_size_gb": data["model_size"]["total_size_gb"],
        }
    )

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart comparing memory, throughput, and perplexity."""
    metrics = [
        ("memory_gb", "GPU Memory (GB)", "tab:blue"),
        ("throughput_tok_s", "Throughput (tok/s)", "tab:green"),
        ("perplexity", "Perplexity", "tab:red"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Llama 3.1 8B — Quantization Method Comparison (A100-SXM4-80GB)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (col, label, color) in zip(axes, metrics):
        bars = ax.bar(df["method"], df[col], color=color, edgecolor="white", width=0.6)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_path}")


def main() -> None:
    df = load_results()

    # Print summary table
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON — Llama 3.1 8B on A100-SXM4-80GB")
    print("=" * 80)
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    print("=" * 80)

    # Generate chart
    output_path = RESULTS_DIR / "comparison_chart.png"
    plot_comparison(df, output_path)


if __name__ == "__main__":
    main()

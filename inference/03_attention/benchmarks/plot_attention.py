"""
Plot the attention-backend benchmark from `results_attention.json`.

Produces two figures in `plots/`:

    attention_memory.png   peak GPU memory vs sequence length, plus the same
                           data with the model weights subtracted out. The
                           subtraction is what makes the scaling visible: at
                           batch 1 the 15 GB of BF16 weights dwarfs the
                           activations until ~8K, so on the raw axis every
                           curve looks flat. Above the weight baseline, eager
                           tracks N^2 and sdpa/FA2 track N.

    attention_prefill.png  prefill latency vs sequence length (log-log), plus
                           speedup over eager. Log-log because a power law
                           shows up as a straight line whose slope is the
                           exponent — eager bends toward slope 2, the Flash
                           kernels stay near slope 1.

Reads only the JSON, so it runs anywhere; no CUDA needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).parent / "results_attention.json"
PLOTS_DIR = Path(__file__).parent / "plots"

# Fixed color/marker/linestyle per backend so the two figures read as one set.
# sdpa is dashed because it dispatches to the same Flash kernel as FA2 and its
# curve sits exactly on top — solid-on-solid would hide one of them.
STYLE: dict[str, tuple[str, str, str]] = {
    "eager": ("tab:red", "o", "-"),
    "sdpa": ("tab:blue", "s", "--"),
    "flash_attention_2": ("tab:green", "^", "-"),
}


@dataclass
class Series:
    """One backend's sweep, with OOM cells dropped rather than plotted as gaps."""

    name: str
    weights_gb: float
    seq_lens: list[int]
    prefill_sec: list[float]
    peak_gb: list[float]

    @property
    def activation_gb(self) -> list[float]:
        """Peak memory above the resident model weights."""
        return [m - self.weights_gb for m in self.peak_gb]


def load_series(path: Path = RESULTS_PATH) -> tuple[list[Series], dict]:
    """Flatten the results JSON into one Series per available backend."""
    with open(path) as f:
        data = json.load(f)

    series: list[Series] = []
    for impl in data["implementation_results"]:
        if not impl["available"]:
            continue
        seq_lens, times, mems = [], [], []
        for entry in impl["prefill"]:
            if entry["error"]:
                continue
            seq_lens.append(entry["seq_len"])
            times.append(entry["median_time_sec"])
            mems.append(entry["peak_memory"]["max_allocated_gb"])
        series.append(
            Series(
                name=impl["attn_implementation"],
                weights_gb=impl["load_memory"]["max_allocated_gb"],
                seq_lens=seq_lens,
                prefill_sec=times,
                peak_gb=mems,
            )
        )
    return series, data


def _reference_line(ax, x: list[int], y0: float, exponent: int, label: str) -> None:
    """Draw a dashed y = y0 * (x/x0)^exponent guide through the first point."""
    ys = [y0 * (xi / x[0]) ** exponent for xi in x]
    ax.plot(x, ys, "k--", alpha=0.35, linewidth=1, label=label)


def plot_memory(series: list[Series], gpu: str, out: Path) -> None:
    """Peak memory vs sequence length, raw and with weights subtracted."""
    fig, (ax_raw, ax_act) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Llama 3.1 8B (BF16, batch 1) — peak GPU memory vs sequence length [{gpu}]",
        fontsize=13,
        fontweight="bold",
    )

    for s in series:
        color, marker, ls = STYLE[s.name]
        ax_raw.plot(
            s.seq_lens,
            s.peak_gb,
            marker=marker,
            color=color,
            linestyle=ls,
            label=s.name,
        )
        ax_act.plot(
            s.seq_lens,
            s.activation_gb,
            marker=marker,
            color=color,
            linestyle=ls,
            label=s.name,
        )

    eager = next(s for s in series if s.name == "eager")
    ax_raw.axhline(
        eager.weights_gb,
        color="gray",
        linestyle=":",
        label=f"model weights ({eager.weights_gb:.1f} GB)",
    )
    ax_raw.annotate(
        "eager OOMs at 16K",
        xy=(eager.seq_lens[-1], eager.peak_gb[-1]),
        xytext=(-30, -34),
        textcoords="offset points",
        fontsize=9,
        color="tab:red",
        ha="right",
        arrowprops={"arrowstyle": "->", "color": "tab:red", "alpha": 0.6},
    )
    ax_raw.set_xscale("log", base=2)
    ax_raw.set_title("Peak allocated memory", fontweight="bold")
    ax_raw.set_ylabel("Peak GPU memory (GB)")

    _reference_line(ax_act, eager.seq_lens, eager.activation_gb[0], 2, "$O(N^2)$")
    _reference_line(ax_act, eager.seq_lens, eager.activation_gb[0], 1, "$O(N)$")
    ax_act.set_xscale("log", base=2)
    ax_act.set_yscale("log", base=2)
    ax_act.set_title(
        "Above the weight baseline (activations + KV cache)", fontweight="bold"
    )
    ax_act.set_ylabel("Peak memory − model weights (GB)")

    for ax in (ax_raw, ax_act):
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_xticks(eager.seq_lens + [16384])
        ax.set_xticklabels([str(n) for n in eager.seq_lens + [16384]])
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_prefill(series: list[Series], gpu: str, out: Path) -> None:
    """Prefill latency vs sequence length, and speedup over eager."""
    fig, (ax_lat, ax_spd) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Llama 3.1 8B (BF16, batch 1) — prefill latency vs sequence length [{gpu}]",
        fontsize=13,
        fontweight="bold",
    )

    for s in series:
        color, marker, ls = STYLE[s.name]
        ax_lat.plot(
            s.seq_lens,
            [t * 1000 for t in s.prefill_sec],
            marker=marker,
            color=color,
            linestyle=ls,
            label=s.name,
        )

    eager = next(s for s in series if s.name == "eager")
    _reference_line(ax_lat, eager.seq_lens, eager.prefill_sec[0] * 1000, 2, "$O(N^2)$")
    _reference_line(ax_lat, eager.seq_lens, eager.prefill_sec[0] * 1000, 1, "$O(N)$")
    ax_lat.set_xscale("log", base=2)
    ax_lat.set_yscale("log", base=2)
    ax_lat.set_title("Prefill latency (median of 3)", fontweight="bold")
    ax_lat.set_ylabel("Prefill time (ms)")

    # Speedup is only defined where eager survived, so index off eager's sweep.
    eager_by_len = dict(zip(eager.seq_lens, eager.prefill_sec))
    for s in series:
        if s.name == "eager":
            continue
        color, marker, ls = STYLE[s.name]
        shared = [n for n in s.seq_lens if n in eager_by_len]
        ratios = [eager_by_len[n] / s.prefill_sec[s.seq_lens.index(n)] for n in shared]
        ax_spd.plot(
            shared,
            ratios,
            marker=marker,
            color=color,
            linestyle=ls,
            label=f"{s.name} vs eager",
        )
        # Label one series only — sdpa and FA2 land within noise of each other
        # and two labels per point just collide.
        if s.name == "flash_attention_2":
            for n, r in zip(shared, ratios):
                ax_spd.annotate(
                    f"{r:.1f}x",
                    xy=(n, r),
                    xytext=(0, -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

    ax_spd.axhline(1.0, color="gray", linestyle=":", label="parity with eager")
    ax_spd.set_xscale("log", base=2)
    ax_spd.set_title("Speedup over eager", fontweight="bold")
    ax_spd.set_ylabel("Eager time / backend time")

    for ax in (ax_lat, ax_spd):
        ax.set_xlabel("Sequence length (tokens)")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    full_sweep = max(series, key=lambda s: len(s.seq_lens)).seq_lens
    ax_lat.set_xticks(full_sweep)
    ax_lat.set_xticklabels([str(n) for n in full_sweep])
    ax_spd.set_xticks(eager.seq_lens)
    ax_spd.set_xticklabels([str(n) for n in eager.seq_lens])

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    series, data = load_series()
    gpu = data["gpu_environment"]["gpu_name"]
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_memory(series, gpu, PLOTS_DIR / "attention_memory.png")
    plot_prefill(series, gpu, PLOTS_DIR / "attention_prefill.png")


if __name__ == "__main__":
    main()

"""
Attention visualizer (tensor2tensor-style bipartite view).

Runs a forward pass on a HuggingFace causal LM, extracts per-head attention
weights, and renders a bipartite token-to-token graph: query tokens on the
left, key tokens on the right, with line opacity proportional to the
attention weight.

Controls:
    - Layer: single-select radio buttons.
    - Heads: multi-select checkboxes, one color per head.
    - Hover: hovering a token (either side) isolates its connections to
      the opposite side.

Usage:
    python visualize_attention.py --model gpt2 \\
        --text "The animal didn't cross the street because it was too tired."
"""

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.widgets import CheckButtons, RadioButtons, TextBox
from transformers import AutoModelForCausalLM, AutoTokenizer

BG = "#111"
FG = "#ddd"
PANEL = "#181818"
HEAT_BASE = np.array([0.12, 0.07, 0.03])
HEAT_PEAK = np.array([0.95, 0.55, 0.15])


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def run_attention(model, tokenizer, text: str) -> tuple[np.ndarray, list[str]]:
    """Return attention tensor of shape (layers, heads, seq, seq) and tokens."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = torch.stack(outputs.attentions, dim=0).squeeze(1).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [t.replace("Ġ", "_").replace("▁", "_") for t in tokens]
    return attn, tokens


def head_palette(n_heads: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20" if n_heads > 10 else "tab10")
    return [tuple(cmap(i / max(1, cmap.N - 1))) for i in range(n_heads)]


class AttentionView:
    """Interactive bipartite attention visualizer."""

    LEFT_X, RIGHT_X = 0.30, 0.70
    LINE_L, LINE_R = 0.32, 0.68

    def __init__(self, model, tokenizer, text: str) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.attn, self.tokens = run_attention(model, tokenizer, text)
        self.n_layers, self.n_heads, self.seq, _ = self.attn.shape
        self.colors = head_palette(self.n_heads)

        self.layer = 0
        self.hover: tuple[str, int] | None = None

        self._build_figure(text)
        self._init_plot_artists()
        self._update()

    # ------------------------------------------------------------------ setup

    def _build_figure(self, initial_text: str) -> None:
        fig = plt.figure(figsize=(14, 9), facecolor=BG)
        self.fig = fig

        self.ax_input = fig.add_axes((0.10, 0.94, 0.54, 0.04))
        self.ax = fig.add_axes((0.04, 0.04, 0.60, 0.86))
        self.ax_heat = fig.add_axes((0.64, 0.04, 0.03, 0.86))
        self.ax_layer = fig.add_axes((0.72, 0.52, 0.24, 0.44))
        self.ax_heads = fig.add_axes((0.72, 0.04, 0.24, 0.44))

        for a in (self.ax, self.ax_heat):
            a.set_facecolor(BG)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)

        for a, title in ((self.ax_layer, "Layer"), (self.ax_heads, "Heads")):
            a.set_facecolor(PANEL)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#333")
            a.set_title(title, color=FG, fontsize=11, loc="left", pad=6)

        self._build_text_input(initial_text)
        self._build_layer_selector()
        self._build_head_selector()

        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("axes_leave_event", self._on_leave)

    def _build_text_input(self, initial_text: str) -> None:
        self.textbox = TextBox(
            self.ax_input, "Text ", initial=initial_text,
            color=PANEL, hovercolor="#222",
            textalignment="left", label_pad=0.02,
        )
        self.textbox.label.set_color(FG)
        self.textbox.label.set_fontsize(10)
        self.textbox.text_disp.set_color(FG)
        self.textbox.text_disp.set_fontsize(10)
        for s in self.ax_input.spines.values():
            s.set_color("#333")
        self.textbox.on_submit(self._on_submit)

    def _build_layer_selector(self) -> None:
        labels = [f"Layer {i}" for i in range(self.n_layers)]
        self.radio = RadioButtons(
            self.ax_layer, labels, active=0,
            activecolor=tuple(HEAT_PEAK),
            radio_props={"s": 40, "edgecolor": FG, "linewidth": 1.0},
            label_props={"color": [FG] * self.n_layers,
                         "fontsize": [9] * self.n_layers},
        )
        self.radio.on_clicked(self._on_layer)

    def _build_head_selector(self) -> None:
        labels = [f"Head {h}" for h in range(self.n_heads)]
        actives = [h == 0 for h in range(self.n_heads)]
        self.check = CheckButtons(
            self.ax_heads, labels,
            actives=actives,
            label_props={"color": list(self.colors),
                         "fontsize": [9] * self.n_heads},
            frame_props={"edgecolor": list(self.colors),
                         "facecolor": ["none"] * self.n_heads,
                         "linewidth": [1.2] * self.n_heads,
                         "s": [90] * self.n_heads},
            check_props={"facecolor": list(self.colors),
                         "s": [90] * self.n_heads},
        )
        self.check.on_clicked(self._on_heads)

    def _on_heads(self, label: str) -> None:
        del label
        self._update()

    # ---------------------------------------------------------------- events

    def _on_layer(self, label: str) -> None:
        self.layer = int(label.split()[-1])
        self._update()

    def _on_motion(self, event) -> None:
        prev = self.hover
        new: tuple[str, int] | None = None
        if event.inaxes is self.ax and event.ydata is not None and event.xdata is not None:
            row = int(round(event.ydata))
            if 0 <= row < self.seq:
                new = ("left", row) if event.xdata < 0.5 else ("right", row)
        if new != prev:
            self.hover = new
            self._update()

    def _on_leave(self, event) -> None:
        if event.inaxes is self.ax and self.hover is not None:
            self.hover = None
            self._update()

    def _on_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.attn, self.tokens = run_attention(self.model, self.tokenizer, text)
        self.seq = self.attn.shape[-1]
        self.hover = None
        self._rebuild_plot_artists()
        self._update()

    def _rebuild_plot_artists(self) -> None:
        for t in (*self._tok_left, *self._tok_right):
            t.remove()
        self._lines.remove()
        for p in self._heat_cells:
            p.remove()
        self._init_plot_artists()

    # ---------------------------------------------------------------- state

    def _active_heads(self) -> list[int]:
        return [i for i, v in enumerate(self.check.get_status()) if v]

    def _heat_values(self, active: list[int]) -> np.ndarray:
        if not active:
            return np.zeros(self.seq)
        stack = np.stack([self.attn[self.layer, h] for h in active])
        if self.hover and self.hover[0] == "left":
            v = stack[:, self.hover[1], :].mean(axis=0)
        elif self.hover and self.hover[0] == "right":
            v = stack[:, :, self.hover[1]].mean(axis=0)
        else:
            v = stack.sum(axis=1).mean(axis=0)
        m = v.max()
        return v / m if m > 0 else v

    # ----------------------------------------------------------------- draw

    def _init_plot_artists(self) -> None:
        """One-time creation of tokens, line collection, and heat rectangles."""
        ax = self.ax
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, self.seq - 0.5)
        ax.invert_yaxis()

        self._tok_left = []
        self._tok_right = []
        for i, tok in enumerate(self.tokens):
            self._tok_left.append(ax.text(
                self.LEFT_X, i, tok, ha="right", va="center",
                fontsize=11, family="monospace", color=FG))
            self._tok_right.append(ax.text(
                self.RIGHT_X, i, tok, ha="left", va="center",
                fontsize=11, family="monospace", color=FG))

        self._lines = LineCollection([], linewidths=1.1, capstyle="round")
        ax.add_collection(self._lines)

        ax_heat = self.ax_heat
        ax_heat.set_xlim(0, 1)
        ax_heat.set_ylim(-0.5, self.seq - 0.5)
        ax_heat.invert_yaxis()
        self._heat_cells = []
        for j in range(self.seq):
            p = mpatches.Rectangle(
                (0.1, j - 0.42), 0.8, 0.84, color=HEAT_BASE, ec="none")
            ax_heat.add_patch(p)
            self._heat_cells.append(p)

    def _update(self) -> None:
        """Refresh artists in place — no axes clear, so hover stays snappy."""
        active = self._active_heads()

        # Token highlights
        for i, t in enumerate(self._tok_left):
            if self.hover == ("left", i):
                t.set_color(BG)
                t.set_bbox(dict(facecolor=FG, edgecolor="none", pad=3))
            else:
                t.set_color(FG)
                t.set_bbox(None)
        for i, t in enumerate(self._tok_right):
            if self.hover == ("right", i):
                t.set_color(BG)
                t.set_bbox(dict(facecolor=FG, edgecolor="none", pad=3))
            else:
                t.set_color(FG)
                t.set_bbox(None)

        # Lines
        if self.hover is None:
            queries, keys = range(self.seq), range(self.seq)
        elif self.hover[0] == "left":
            queries, keys = [self.hover[1]], range(self.seq)
        else:
            queries, keys = range(self.seq), [self.hover[1]]

        segments: list[list[tuple[float, float]]] = []
        rgba: list[tuple[float, float, float, float]] = []
        weights: list[float] = []
        for h in active:
            W = self.attn[self.layer, h]
            base = self.colors[h]
            for i in queries:
                for j in keys:
                    w = float(W[i, j])
                    if w < 0.01:
                        continue
                    segments.append([(self.LINE_L, i), (self.LINE_R, j)])
                    rgba.append((base[0], base[1], base[2], min(1.0, w)))
                    weights.append(w)

        if segments:
            order = np.argsort(weights)  # faint first so strong edges land on top
            segments = [segments[k] for k in order]
            rgba = [rgba[k] for k in order]
        self._lines.set_segments(segments)
        self._lines.set_color(rgba)

        # Heat column
        values = self._heat_values(active)
        for j, p in enumerate(self._heat_cells):
            p.set_facecolor(HEAT_BASE + (HEAT_PEAK - HEAT_BASE) * float(values[j]))

        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--text",
        default="The animal didn't cross the street because it was too tired.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    AttentionView(model, tokenizer, args.text)
    plt.show()


if __name__ == "__main__":
    main()

"""Generate a PNG diagram of the PMSSMTransformerTabular architecture.

Run from the repo root:
    ./.pixi/envs/rocm/bin/python scripts/plot_transformer_architecture.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent.parent / "doc" / "transformer_architecture.png"

# ── colour scheme ────────────────────────────────────────────────────────────
C_EMBED = "#cfe8ff"
C_ENC = "#ffd9b3"
C_ENC_DK = "#ffb27a"
C_POOL = "#d8f0c8"
C_HEAD = "#f5cde2"
C_IO = "#e8e8e8"

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, label, colour, fontsize=10, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.2, edgecolor="#333", facecolor=colour,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(x, y1, y2, label=None):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color="#333",
    ))
    if label:
        ax.text(x + 0.15, (y1 + y2) / 2, label,
                ha="left", va="center", fontsize=8.5, style="italic",
                color="#555")


def param_tag(x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=9,
            color="#1f4e79", weight="bold")


cx = 5.0  # centre x for main column
bw = 5.4  # box width

# ── title ────────────────────────────────────────────────────────────────────
ax.text(cx, 14.55, "PMSSMTransformerTabular", ha="center",
        fontsize=15, weight="bold")
ax.text(cx, 14.15,
        "d_model=128 · nhead=4 · num_layers=3 · dim_ff=512 · dropout=0.1",
        ha="center", fontsize=9.5, style="italic", color="#555")
ax.text(cx, 13.8, "Total trainable: 803,330 parameters",
        ha="center", fontsize=10.5, weight="bold", color="#1f4e79")

# ── input ────────────────────────────────────────────────────────────────────
y = 12.75
box(cx - 1.4, y, 2.8, 0.55, "Input  x ∈ (B, 19)", C_IO, bold=True)

# ── feature embeddings ───────────────────────────────────────────────────────
arrow(cx, y, y - 0.55)
y = 11.7
box(cx - bw / 2, y, bw, 0.9,
    "Per-feature embeddings  (ModuleList × 19)\n"
    "Linear(1→128) + LayerNorm(128)",
    C_EMBED, fontsize=10)
param_tag(cx + bw / 2 + 0.2, y + 0.45, "9,728  (1.2%)")
ax.text(cx + bw / 2 + 0.2, y + 0.15, "→ (B, 19, 128)",
        ha="left", va="center", fontsize=8, color="#555")

# ── encoder block ────────────────────────────────────────────────────────────
arrow(cx, y, y - 0.45)
enc_top = 11.15
enc_h = 6.0
enc_bottom = enc_top - enc_h

# outer encoder rectangle
outer = FancyBboxPatch(
    (cx - bw / 2 - 0.15, enc_bottom - 0.05), bw + 0.3, enc_h + 0.1,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    linewidth=2.2, edgecolor="#b85c00", facecolor=C_ENC,
)
ax.add_patch(outer)
ax.text(cx, enc_top - 0.3, "TransformerEncoder  ×3  (norm_first=True)",
        ha="center", fontsize=11.5, weight="bold", color="#7a3d00")
param_tag(cx + bw / 2 + 0.4, enc_top - 0.3, "594,816  (74%)")

# inner one-layer diagram
lay_top = enc_top - 0.7
lay_h = 4.9
lay_w = bw - 0.6
lay_x = cx - lay_w / 2
lay_bottom = lay_top - lay_h

inner = FancyBboxPatch(
    (lay_x, lay_bottom), lay_w, lay_h,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=1.0, edgecolor="#7a3d00", facecolor="#fff3e0",
    linestyle="--",
)
ax.add_patch(inner)
ax.text(lay_x + 0.15, lay_top - 0.25, "One encoder layer  (198,272 params)",
        ha="left", va="center", fontsize=9.5, style="italic", color="#7a3d00")

# sub-blocks inside one layer
sub_w = lay_w - 0.6
sub_x = lay_x + 0.3
sub_h = 0.45

items = [
    ("LayerNorm(128)", "256"),
    ("MultiheadAttention  (d=128, h=4)", "66,048"),
    ("+ residual", ""),
    ("LayerNorm(128)", "256"),
    ("Linear(128→512)  →  ReLU  →  Dropout(0.1)", "66,048"),
    ("Linear(512→128)", "65,664"),
    ("+ residual", ""),
]

top = lay_top - 0.6
gap = 0.10
for i, (label, n) in enumerate(items):
    yy = top - i * (sub_h + gap) - sub_h
    residual = label.startswith("+")
    colour = "#fffbe6" if residual else C_ENC_DK
    fs = 9 if not residual else 8.5
    box(sub_x, yy, sub_w, sub_h, label, colour, fontsize=fs,
        bold=False)
    if n:
        param_tag(sub_x + sub_w + 0.1, yy + sub_h / 2, n)

# arrow from encoder to pooling
arrow(cx, enc_bottom - 0.05, enc_bottom - 0.55, label="(B, 19, 128)")

# ── attention pooling ────────────────────────────────────────────────────────
y = enc_bottom - 1.35
box(cx - bw / 2, y, bw, 0.8,
    "Attention pooling\n"
    "w = softmax(Linear(128→1)),   x = Σ w·x",
    C_POOL, fontsize=9.5)
param_tag(cx + bw / 2 + 0.2, y + 0.4, "129  (~0%)")
ax.text(cx + bw / 2 + 0.2, y + 0.1, "→ (B, 128)",
        ha="left", va="center", fontsize=8, color="#555")

# ── regression head ──────────────────────────────────────────────────────────
arrow(cx, y, y - 0.45)
y_head_top = y - 0.45
head_h = 1.85
y = y_head_top - head_h
box(cx - bw / 2, y, bw, head_h,
    "Regression head\n"
    "Linear(128→512)   →   66,048\n"
    "LayerNorm(512)    →    1,024\n"
    "ReLU\n"
    "Linear(512→256)   →  131,328\n"
    "ReLU\n"
    "Linear(256→1)     →      257",
    C_HEAD, fontsize=9)
param_tag(cx + bw / 2 + 0.2, y + head_h / 2, "198,657  (25%)")

# ── output ───────────────────────────────────────────────────────────────────
arrow(cx, y, y - 0.45)
y = y - 0.45 - 0.55
box(cx - 1.4, y, 2.8, 0.55, "Output  y ∈ (B, 1)", C_IO, bold=True)

# ── legend ───────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_EMBED, label="Feature embeddings"),
    mpatches.Patch(color=C_ENC, label="Transformer encoder ×3"),
    mpatches.Patch(color=C_POOL, label="Attention pooling"),
    mpatches.Patch(color=C_HEAD, label="Regression head"),
]
ax.legend(handles=legend_handles, loc="lower left",
          bbox_to_anchor=(0.01, 0.01), fontsize=8.5, frameon=True)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")

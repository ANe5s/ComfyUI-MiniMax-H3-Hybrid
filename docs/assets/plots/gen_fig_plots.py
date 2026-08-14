#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the data figures for the MiniMax H3 Hybrid Loader technical whitepaper.

Input: measurement JSONs under docs/assets/ in the plugin repository.
Output: publication-style figures (PNG 300 DPI + PDF vector).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JSON_BASE = r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Hybrid\docs\assets"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

if len(sys.argv) > 1:
    JSON_BASE = sys.argv[1]
if len(sys.argv) > 2:
    OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

# --- Publication defaults (per academic-plotting skill) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

# Okabe-Ito (colorblind-safe) + skill highlight
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_GREEN = "#009E73"
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"   # our highlight
C_PINK = "#CC79A7"
C_GRAY = "#B0BEC5"

FIG_SINGLE = (3.25, 2.5)
FIG_FULL = (6.75, 2.8)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}.pdf/.png")


def load(name):
    with open(os.path.join(JSON_BASE, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- Fig 1
def fig_weight_vs_function():
    """Pruned pair: AdaLN cosine in weight space vs. function space per block."""
    d = load("h3_pruned_function_space.json")
    blocks = [b["block"] for b in d["per_block"]]
    cos_w = [b["cos_weight"] for b in d["per_block"]]
    cos_f = [b["cos_func"] for b in d["per_block"]]
    mean_w = d["summary"]["cos_weight_all"]
    mean_f = d["summary"]["cos_func_all"]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(blocks, cos_w, label=f"Weight space (mean {mean_w:.3f})", color=C_BLUE)
    ax.plot(blocks, cos_f, label=f"Function space (mean {mean_f:.4f})", color=C_VERMILLION)
    ax.axhline(mean_w, color=C_BLUE, ls=":", lw=1.0, alpha=0.7)
    ax.axhline(mean_f, color=C_VERMILLION, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.0, 1.05)
    ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right")
    save(fig, "fig_pruned_weight_vs_function")


# ---------------------------------------------------------------- Fig 2
def fig_pruned_func_rel_by_block():
    """Pruned pair: function-space output relative difference per block and range means."""
    d = load("h3_pruned_function_space_rel.json")
    blocks = [b["block"] for b in d["per_block"]]
    rel_v = [b["rel_video"] * 100 for b in d["per_block"]]
    rel_t = [b["rel_text"] * 100 for b in d["per_block"]]
    rel_a = [b["rel_audio"] * 100 for b in d["per_block"]]

    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.plot(blocks, rel_v, label="video", color=C_VERMILLION)
    ax.plot(blocks, rel_t, label="text", color=C_BLUE)
    ax.plot(blocks, rel_a, label="audio", color=C_GREEN)

    # range mean shading: 0-24 vs 25-49
    r024 = d["ranges"]["0-24"]
    r2549 = d["ranges"]["25-49"]
    m024 = np.mean([r024["rel_video"], r024["rel_text"], r024["rel_audio"]]) * 100
    m2549 = np.mean([r2549["rel_video"], r2549["rel_text"], r2549["rel_audio"]]) * 100
    ax.axhline(m024, color=C_GRAY, ls="--", lw=1.2)
    ax.axhline(m2549, color=C_GRAY, ls="--", lw=1.2)
    ax.text(0.4, m024 + 0.12, f"0-24 mean {m024:.1f}%", fontsize=8, color="#444")
    ax.text(25.5, m2549 + 0.12, f"25-49 mean {m2549:.1f}%", fontsize=8, color="#444")

    # highlight the max-difference region 7-17
    ax.axvspan(7, 17, color=C_ORANGE, alpha=0.10)
    ytop = ax.get_ylim()[1]
    ax.text(12, ytop - (ytop - ax.get_ylim()[0]) * 0.06,
            "max difference\nregion 7-17", fontsize=7.5, color="#8a6d1a",
            ha="center", va="top")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Function-space relative difference (%)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xlim(0, 49)
    save(fig, "fig_pruned_func_rel_by_block")


# ---------------------------------------------------------------- Fig 3
def fig_full_key_tensors():
    """Full pair: relative-mean differences of non-block key tensors."""
    d = load("h3_real_model_measurements_full.json")
    items = []
    for key, v in d["other"].items():
        if key == "time_embedder.proj_in.weight":
            label = "time_embedder.proj_in.weight"
        elif key == "time_embedder.proj_in.bias":
            label = "time_embedder.proj_in.bias"
        elif key == "time_embedder.proj_out.weight":
            label = "time_embedder.proj_out.weight"
        elif key == "time_embedder.proj_out.bias":
            label = "time_embedder.proj_out.bias"
        elif key == "final_layer.adaln_proj.linear.weight":
            label = "final_layer.adaln_proj.linear.weight"
        else:
            label = key
        items.append((label, v["rel_mean"]))
    items.sort(key=lambda x: x[1])
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    colors = [C_BLUE if "adaln" in l else C_GRAY for l in labels]
    colors[labels.index("time_embedder.proj_out.weight")] = C_VERMILLION

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Relative mean difference")
    ax.set_xlim(0, 0.72)
    ax.invert_yaxis()
    save(fig, "fig_full_key_tensors")


# ---------------------------------------------------------------- Fig 4
def fig_sharpness_proxy():
    """Clarity proxy metrics: P 39-44 and F 45-49 vs their FL2VA baselines."""
    metrics = ["overall", "face", "product"]
    p_base = [95.5, 58.0, 152.0]
    p_over = [98.5, 61.7, 154.0]
    f_base = [92.9, 58.2, 145.6]
    f_over = [110.7, 74.0, 164.3]

    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, sharey=False)
    for ax, (title, base, over) in zip(
        axes,
        [("P 39-44", p_base, p_over), ("F 45-49", f_base, f_over)],
    ):
        x = np.arange(len(metrics))
        w = 0.34
        b1 = ax.bar(x - w / 2, base, w, label="FL2VA base", color=C_GRAY,
                    edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + w / 2, over, w, label="Overlay", color=C_VERMILLION,
                    edgecolor="white", linewidth=0.5)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{bar.get_height():.1f}", ha="center", va="bottom",
                        fontsize=7, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(title)
        ax.set_ylabel("Clarity proxy score")
    axes[0].legend(loc="lower right")
    save(fig, "fig_sharpness_proxy")


if __name__ == "__main__":
    fig_weight_vs_function()
    fig_pruned_func_rel_by_block()
    fig_full_key_tensors()
    fig_sharpness_proxy()

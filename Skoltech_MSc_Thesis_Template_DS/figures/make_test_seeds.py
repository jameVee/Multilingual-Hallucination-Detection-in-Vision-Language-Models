"""Figure: official test Cor+Lbl per training seed and the three-seed mean.

Numbers are the official leaderboard results logged in thesis_v2/results/test_results.txt.
Usage (from the thesis folder):  python figures/make_test_seeds.py  -> images/test_seeds.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = {  # system -> test Cor+Lbl for seeds 42, 123, 2024
    "C0  judge, original data": (0.4158, 0.4145, 0.4094),
    "F2  + per-label vote": (0.4180, 0.4223, 0.4183),
    "F3  + probe voter": (0.4206, 0.4276, 0.4215),
    "G1  + graded probabilities": (0.4280, 0.4328, 0.4279),
}
SERIES = "#2a78d6"       # reference categorical slot 1
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11})
fig, ax = plt.subplots(figsize=(7.0, 3.0))
names = list(SEEDS)[::-1]                       # final detector at the bottom, nearest the axis
for y, name in enumerate(names):
    vals = SEEDS[name]
    mean = sum(vals) / len(vals)
    ax.plot([min(vals), max(vals)], [y, y], color=SERIES, lw=2, alpha=0.35, solid_capstyle="round", zorder=1)
    ax.scatter(vals, [y] * 3, s=36, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3,
               label="single seed" if y == 0 else None)
    ax.scatter([mean], [y], s=110, marker="D", color=INK, edgecolor="white", linewidth=2, zorder=4,
               label="three-seed mean" if y == 0 else None)
    ax.annotate(f"{mean:.4f}", (max(vals), y), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("Official test Cor+Lbl (4-language mean)", color=MUTED)
ax.set_xlim(0.405, 0.438)
ax.set_ylim(-0.6, len(names) - 0.4)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", length=0)
ax.tick_params(colors=MUTED)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9, borderaxespad=0.2)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "test_seeds.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=150)
print("->", out)

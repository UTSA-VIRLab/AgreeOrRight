#!/usr/bin/env python3
"""
Generate VASE invalidity figure (2 panels):
  (a) Histogram of clipped probability mass per token-level distribution
  (b) Bar chart: % with negatives and mean clipped mass
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_ROOT = Path("/raid/den365/AgenticMedXAI_CVPR2026/outputs")

with open(OUTPUT_ROOT / "vase_vs_lvase_llava_1.5.json") as f:
    llava = json.load(f)
with open(OUTPUT_ROOT / "vase_vs_lvase_llava_med.json") as f:
    med = json.load(f)

# Check that per-token data exists
assert "clipped_mass_per_token" in llava["proof1_invalid_distributions"], \
    "Missing clipped_mass_per_token — re-run compare_vase_lvase.py first"

clip_llava = np.array(llava["proof1_invalid_distributions"]["clipped_mass_per_token"])
clip_med = np.array(med["proof1_invalid_distributions"]["clipped_mass_per_token"])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                gridspec_kw={"width_ratios": [1.4, 1.0],
                                             "wspace": 0.35})

colors = {"llava": "#2196F3", "med": "#E91E63"}

# === Panel (a): ECDF of clipped mass (excludes zeros for clarity) ===
for arr, color, label in [
    (clip_llava, colors["llava"], f"LLaVA-1.5 ($n$={len(clip_llava):,})"),
    (clip_med, colors["med"], f"LLaVA-Med ($n$={len(clip_med):,})"),
]:
    # Include all values (zeros = valid distributions)
    sorted_vals = np.sort(arr)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax0.plot(sorted_vals, ecdf, color=color, linewidth=1.5, label=label, zorder=3)

# Light shading for invalid region
ax0.axvspan(0.001, 0.52, alpha=0.05, color="red", zorder=1)

# Mark the jump: fraction with zero clipped mass
frac_valid_llava = np.mean(clip_llava == 0)
frac_valid_med = np.mean(clip_med == 0)

# Annotations — stacked vertically to avoid overlap
ax0.annotate(f"only {frac_valid_llava*100:.1f}% valid",
             xy=(0.002, frac_valid_llava), xytext=(0.15, 0.28),
             fontsize=7, color=colors["llava"],
             arrowprops=dict(arrowstyle="->", color=colors["llava"], lw=0.8))
ax0.annotate(f"only {frac_valid_med*100:.1f}% valid",
             xy=(0.002, frac_valid_med), xytext=(0.15, 0.45),
             fontsize=7, color=colors["med"],
             arrowprops=dict(arrowstyle="->", color=colors["med"], lw=0.8))

# Label the spike region
ax0.annotate("most mass\nnear 0.50", xy=(0.48, 0.55), xytext=(0.33, 0.7),
             fontsize=7, color="gray", ha="center", style="italic",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax0.set_xlabel("Clipped probability mass per token")
ax0.set_ylabel("Cumulative fraction")
ax0.set_title("(a) ECDF of clipped mass", fontweight="bold")
ax0.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
ax0.set_xlim(-0.02, 0.52)
ax0.set_ylim(0, 1.02)

# === Panel (b): Summary bar chart ===
models = ["LLaVA-1.5", "LLaVA-Med"]
neg_pcts = [
    llava["proof1_invalid_distributions"]["pct_with_negatives"],
    med["proof1_invalid_distributions"]["pct_with_negatives"],
]
clipped_mass_pct = [
    llava["proof1_invalid_distributions"]["mean_clipped_mass"] * 100,
    med["proof1_invalid_distributions"]["mean_clipped_mass"] * 100,
]

x = np.arange(len(models))
w = 0.30

bars1 = ax1.bar(x - w/2, neg_pcts, w, color="#F44336", alpha=0.85,
                label="Vectors with\nnegative entries (%)", zorder=3)
bars2 = ax1.bar(x + w/2, clipped_mass_pct, w, color="#FF9800", alpha=0.85,
                label="Mean clipped\nmass (%)", zorder=3)

for bar, val in zip(bars1, neg_pcts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
for bar, val in zip(bars2, clipped_mass_pct):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5)
ax1.set_ylim(0, 115)
ax1.set_ylabel("Percentage (%)")
ax1.set_title("(b) VASE invalidity summary", fontweight="bold")
ax1.legend(loc="center left", fontsize=6.5, framealpha=0.9,
           bbox_to_anchor=(0.02, 0.55))

fig.subplots_adjust(left=0.09, right=0.97, bottom=0.19, top=0.87)

out_path = OUTPUT_ROOT / "figures" / "vase_invalidity.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
print(f"Saved to {out_path} and {out_path.with_suffix('.png')}")

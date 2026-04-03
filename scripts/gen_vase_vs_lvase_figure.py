#!/usr/bin/env python3
"""Generate VASE vs L-VASE comparison figure for the paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_ROOT = Path("/raid/den365/AgenticMedXAI_CVPR2026/outputs")

# Load data
with open(OUTPUT_ROOT / "vase_vs_lvase_llava_1.5.json") as f:
    llava = json.load(f)
with open(OUTPUT_ROOT / "vase_vs_lvase_llava_med.json") as f:
    med = json.load(f)

# --- Setup ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                         gridspec_kw={"width_ratios": [1.2, 1.2, 1.0],
                                      "wspace": 0.42})

colors = {"llava": "#2196F3", "med": "#E91E63"}

# === Panel A: Scatter — VASE vs L-VASE per image ===
ax0 = axes[0]

for data, label, color, marker in [
    (llava, "LLaVA-1.5", colors["llava"], "o"),
    (med, "LLaVA-Med", colors["med"], "s"),
]:
    vase = data["proof2_scores"]["per_image_vase"]
    lvase = data["proof2_scores"]["per_image_lvase"]
    ax0.scatter(vase, lvase, c=color, marker=marker, s=25, alpha=0.7,
                edgecolors="white", linewidths=0.3, label=label, zorder=3)

# Identity line
lims = [0.3, 1.7]
ax0.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.5, zorder=1)
ax0.set_xlim(lims)
ax0.set_ylim([0.3, 1.15])
ax0.set_xlabel("VASE (prob. space)")
ax0.set_ylabel("L-VASE (logit space)")
ax0.set_title("(a) Per-image scores", fontweight="bold")
ax0.legend(loc="upper left", framealpha=0.9)

# Annotate: L-VASE always below VASE
ax0.annotate("L-VASE < VASE\n(valid by construction)",
             xy=(1.1, 0.38), fontsize=6.5, color="gray", ha="center",
             style="italic")

# === Panel B: Paired difference (VASE - L-VASE) ===
ax1 = axes[1]

# Compute differences and interleave
diff_llava = np.sort(np.array(llava["proof2_scores"]["per_image_vase"]) -
                     np.array(llava["proof2_scores"]["per_image_lvase"]))
diff_med = np.sort(np.array(med["proof2_scores"]["per_image_vase"]) -
                   np.array(med["proof2_scores"]["per_image_lvase"]))

y_llava = np.arange(len(diff_llava))
y_med = np.arange(len(diff_med))

ax1.barh(y_llava + 0.2, diff_llava, height=0.38, color=colors["llava"],
         alpha=0.7, label="LLaVA-1.5", zorder=3)
ax1.barh(y_med - 0.2, diff_med, height=0.38, color=colors["med"],
         alpha=0.7, label="LLaVA-Med", zorder=3)

ax1.axvline(0, color="black", linewidth=0.5)
ax1.set_xlabel("VASE $-$ L-VASE (inflation)")
ax1.set_ylabel("Image (sorted by inflation)")
ax1.set_title("(b) Score inflation by VASE", fontweight="bold")
ax1.legend(loc="lower right", framealpha=0.9, fontsize=7)

# === Panel C: Invalid distribution summary ===
ax2 = axes[2]

models = ["LLaVA-1.5", "LLaVA-Med"]
neg_pcts = [
    llava["proof1_invalid_distributions"]["pct_with_negatives"],
    med["proof1_invalid_distributions"]["pct_with_negatives"],
]
clipped_mass = [
    llava["proof1_invalid_distributions"]["mean_clipped_mass"] * 100,
    med["proof1_invalid_distributions"]["mean_clipped_mass"] * 100,
]
n_vectors = [
    llava["proof1_invalid_distributions"]["total_distributions"],
    med["proof1_invalid_distributions"]["total_distributions"],
]

x = np.arange(len(models))
w = 0.32

bars1 = ax2.bar(x - w/2, neg_pcts, w, color="#F44336", alpha=0.8,
                label="% with negatives", zorder=3)
bars2 = ax2.bar(x + w/2, clipped_mass, w, color="#FF9800", alpha=0.8,
                label="Mean clipped mass (%)", zorder=3)

# Value labels
for bar, val in zip(bars1, neg_pcts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
             f"{val:.1f}%", ha="center", va="top", fontsize=6.5,
             fontweight="bold", color="white")
for bar, val in zip(bars2, clipped_mass):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
             f"{val:.1f}%", ha="center", va="top", fontsize=6.5,
             fontweight="bold", color="white")

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8)
ax2.set_ylim(0, 108)
ax2.set_ylabel("Percentage")
ax2.set_title("(c) VASE invalidity", fontweight="bold")
ax2.legend(loc="center right", framealpha=0.9, fontsize=6.5)

# Add n= annotations below bars
for i, n in enumerate(n_vectors):
    ax2.text(i, 2, f"$n$={n:,}", ha="center", fontsize=6.5, color="gray")

fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.88)

out_path = OUTPUT_ROOT / "figures" / "vase_vs_lvase.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
print(f"Saved to {out_path} and {out_path.with_suffix('.png')}")

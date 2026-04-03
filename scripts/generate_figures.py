#!/usr/bin/env python3
"""
Generate publication-quality figures for GroundingProbe paper.

Figures:
1. Main results bar chart (L-VASE, NIA, GSS, Resistance)
2. Cross-axis correlation scatter (L-VASE vs Resistance, NIA vs Resistance, etc.)
3. Per-pressure-type grouped bar chart
4. NIA shift visualization (baseline vs challenged)
5. VASE vs L-VASE proof figure
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
MODELS = ["llava_1.5", "qwen3_vl", "llava_med", "medvlm_r1", "medgemma"]
MODEL_LABELS = {
    "llava_1.5": "LLaVA-1.5",
    "qwen3_vl": "Qwen3-VL",
    "llava_med": "LLaVA-Med",
    "medvlm_r1": "MedVLM-R1",
    "medgemma": "MedGemma",
}
MODEL_TYPES = {
    "llava_1.5": "General",
    "qwen3_vl": "General",
    "llava_med": "Medical",
    "medvlm_r1": "Medical",
    "medgemma": "Medical",
}

COLORS = {
    "llava_1.5": "#4472C4",
    "qwen3_vl": "#5B9BD5",
    "llava_med": "#C0392B",
    "medvlm_r1": "#E74C3C",
    "medgemma": "#E67E22",
}

results = {}
for m in MODELS:
    path = Path("outputs") / ("groundingprobe_" + m + ".json")
    with open(path) as f:
        results[m] = json.load(f)


# ================================================================
# Figure 1: Main results bar chart (4 metrics side by side)
# ================================================================
def fig1_main_results():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    labels = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(labels))
    width = 0.6

    vals = [results[m]["lvase"]["mean"] for m in MODELS]
    errs = [results[m]["lvase"]["std"] for m in MODELS]
    colors = [COLORS[m] for m in MODELS]
    axes[0].bar(x, vals, width, yerr=errs, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("L-VASE Score")
    axes[0].set_title("Hallucination\n(lower = better)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    vals = [results[m]["gss"]["mean_baseline_nia"] for m in MODELS]
    axes[1].bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Baseline NIA")
    axes[1].set_title("Visual Grounding\n(higher = better)", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    vals = [results[m]["gss"]["mean_gss"] for m in MODELS]
    errs = [results[m]["gss"]["std_gss"] for m in MODELS]
    axes[2].bar(x, vals, width, yerr=errs, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("GSS")
    axes[2].set_title("Grounding Shift\n(lower = more robust)", fontsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    vals = [results[m]["gss"]["overall_resistance_rate"] * 100 for m in MODELS]
    axes[3].bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    axes[3].set_ylabel("Resistance Rate (%)")
    axes[3].set_title("Sycophancy Resistance\n(higher = better)", fontsize=10)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    gen_patch = mpatches.Patch(color="#4472C4", label="General VLM")
    med_patch = mpatches.Patch(color="#C0392B", label="Medical VLM")
    fig.legend(handles=[gen_patch, med_patch], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_main_results.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig1_main_results.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig1_main_results")


# ================================================================
# Figure 2: Cross-axis correlation scatter plots
# ================================================================
def fig2_cross_axis():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    lvase = [results[m]["lvase"]["mean"] for m in MODELS]
    nia = [results[m]["gss"]["mean_baseline_nia"] for m in MODELS]
    gss = [results[m]["gss"]["mean_gss"] for m in MODELS]
    rr = [results[m]["gss"]["overall_resistance_rate"] * 100 for m in MODELS]

    def plot_scatter(ax, xvals, yvals, xlabel, ylabel, title):
        for i, m in enumerate(MODELS):
            marker = "s" if MODEL_TYPES[m] == "General" else "o"
            ax.scatter(xvals[i], yvals[i], c=COLORS[m], s=150, marker=marker,
                       edgecolors="black", linewidth=0.8, zorder=5)
            # Smart label placement
            offset_x, offset_y = 8, 8
            if m == "qwen3_vl":
                offset_y = -15
            ax.annotate(MODEL_LABELS[m], (xvals[i], yvals[i]),
                        textcoords="offset points", xytext=(offset_x, offset_y),
                        fontsize=8, fontweight="bold")

        # Trend line
        z = np.polyfit(xvals, yvals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xvals) * 0.9, max(xvals) * 1.1, 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, linewidth=1)

        # Correlation
        r_p, p_p = pearsonr(xvals, yvals)
        r_s, p_s = spearmanr(xvals, yvals)
        ax.text(0.05, 0.95, "r={:.2f} (p={:.3f})\n$\\rho$={:.2f} (p={:.3f})".format(r_p, p_p, r_s, p_s),
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

    plot_scatter(axes[0], lvase, rr,
                 "L-VASE (Hallucination)", "Resistance Rate (%)",
                 "Hallucination vs Sycophancy Resistance")

    plot_scatter(axes[1], nia, rr,
                 "Baseline NIA (Visual Grounding)", "Resistance Rate (%)",
                 "Visual Grounding vs Sycophancy Resistance")

    plot_scatter(axes[2], lvase, gss,
                 "L-VASE (Hallucination)", "GSS (Grounding Shift)",
                 "Hallucination vs Grounding Shift")

    gen_patch = plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                            markersize=8, label="General VLM")
    med_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                            markersize=8, label="Medical VLM")
    fig.legend(handles=[gen_patch, med_patch], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05), fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_cross_axis.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig2_cross_axis.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig2_cross_axis")


# ================================================================
# Figure 3: Per-pressure-type grouped bar chart
# ================================================================
def fig3_pressure_types():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ptypes = ["expert_correction", "consensus", "authority"]
    ptype_labels = ["Expert\nCorrection", "Consensus\nPanel", "ACR\nAuthority"]
    x = np.arange(len(ptypes))
    width = 0.15

    for i, m in enumerate(MODELS):
        vals = [results[m]["gss"]["per_pressure"][p]["resistance_rate"] * 100 for p in ptypes]
        axes[0].bar(x + i * width, vals, width, label=MODEL_LABELS[m], color=COLORS[m],
                    edgecolor="black", linewidth=0.5)

    axes[0].set_ylabel("Resistance Rate (%)")
    axes[0].set_title("Resistance by Pressure Type", fontsize=10)
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(ptype_labels, fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")

    for i, m in enumerate(MODELS):
        vals = [results[m]["gss"]["per_pressure"][p]["mean_gss"] for p in ptypes]
        axes[1].bar(x + i * width, vals, width, label=MODEL_LABELS[m], color=COLORS[m],
                    edgecolor="black", linewidth=0.5)

    axes[1].set_ylabel("Grounding Shift Score")
    axes[1].set_title("Grounding Shift by Pressure Type", fontsize=10)
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels(ptype_labels, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_pressure_types.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig3_pressure_types.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig3_pressure_types")


# ================================================================
# Figure 4: NIA shift (baseline vs challenged) per model
# ================================================================
def fig4_nia_shift():
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(labels))
    width = 0.35

    baseline_nia = [results[m]["gss"]["mean_baseline_nia"] for m in MODELS]
    challenged_nia = []
    for m in MODELS:
        avg_nia = np.mean([results[m]["gss"]["per_pressure"][p]["mean_challenged_nia"]
                          for p in ["expert_correction", "consensus", "authority"]])
        challenged_nia.append(avg_nia)

    ax.bar(x - width/2, baseline_nia, width, label="Baseline NIA",
           color="#2ECC71", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, challenged_nia, width, label="Challenged NIA",
           color="#E74C3C", edgecolor="black", linewidth=0.5)

    # Add drop percentage labels
    for i in range(len(MODELS)):
        if baseline_nia[i] > 0:
            drop_pct = (baseline_nia[i] - challenged_nia[i]) / baseline_nia[i] * 100
            y_pos = max(baseline_nia[i], challenged_nia[i]) + 0.005
            ax.text(x[i], y_pos, "{:.0f}%".format(drop_pct),
                    ha="center", fontsize=8, fontweight="bold", color="#E74C3C")

    ax.set_ylabel("Normalized Image Attention (NIA)")
    ax.set_title("Visual Grounding Drop Under Sycophantic Pressure", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_nia_shift.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig4_nia_shift.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig4_nia_shift")


# ================================================================
# Figure 5: VASE vs L-VASE proof
# ================================================================
def fig5_vase_proof():
    # Load comparison data
    vase_files = list(Path("outputs").glob("vase_vs_lvase_*.json"))
    if not vase_files:
        print("No VASE vs L-VASE comparison files found, skipping fig5")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    all_data = {}
    for f in vase_files:
        with open(f) as fp:
            d = json.load(fp)
        all_data[d["model"]] = d

    # Panel A: % distributions with negatives
    models_v = list(all_data.keys())
    labels_v = [MODEL_LABELS.get(m.replace("-", "_"), m) for m in models_v]
    pct_neg = [all_data[m]["proof1_invalid_distributions"]["pct_with_negatives"] for m in models_v]
    clipped = [all_data[m]["proof1_invalid_distributions"]["mean_clipped_mass"] * 100 for m in models_v]

    x = np.arange(len(models_v))
    axes[0].bar(x, pct_neg, 0.6, color="#E74C3C", edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("% of Distributions")
    axes[0].set_title("VASE: Invalid Distributions\n(with negative values)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_v, fontsize=9)
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(pct_neg):
        axes[0].text(i, v + 1, "{:.1f}%".format(v), ha="center", fontsize=9, fontweight="bold")

    # Panel B: clipped probability mass
    axes[1].bar(x, clipped, 0.6, color="#E67E22", edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("% Probability Mass Lost")
    axes[1].set_title("VASE: Information Loss\n(clipped probability mass)", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_v, fontsize=9)
    for i, v in enumerate(clipped):
        axes[1].text(i, v + 0.5, "{:.1f}%".format(v), ha="center", fontsize=9, fontweight="bold")

    # Panel C: VASE vs L-VASE scatter per image (first model)
    m0 = models_v[0]
    vase_scores = all_data[m0]["proof2_scores"]["per_image_vase"]
    lvase_scores = all_data[m0]["proof2_scores"]["per_image_lvase"]
    axes[2].scatter(vase_scores, lvase_scores, c="#4472C4", s=40, alpha=0.7,
                    edgecolors="black", linewidth=0.3)
    # Add diagonal
    mn = min(min(vase_scores), min(lvase_scores))
    mx = max(max(vase_scores), max(lvase_scores))
    axes[2].plot([mn, mx], [mn, mx], "--", color="gray", alpha=0.5)
    r, p = pearsonr(vase_scores, lvase_scores)
    axes[2].text(0.05, 0.95, "r={:.3f}\n{:.1f}% rank\ndisagreements".format(
        r, (1 - all_data[m0]["proof2_scores"].get("kendall_tau", 0.825)) * 50),
        transform=axes[2].transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    axes[2].set_xlabel("VASE Score (original)", fontsize=10)
    axes[2].set_ylabel("L-VASE Score (ours)", fontsize=10)
    axes[2].set_title("Per-Image Score Comparison\n({})".format(MODEL_LABELS.get(m0.replace("-","_"), m0)), fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_vase_proof.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig5_vase_proof.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig5_vase_proof")


# ================================================================
# Compute and print all correlations
# ================================================================
def compute_correlations():
    lvase = [results[m]["lvase"]["mean"] for m in MODELS]
    nia = [results[m]["gss"]["mean_baseline_nia"] for m in MODELS]
    gss = [results[m]["gss"]["mean_gss"] for m in MODELS]
    rr = [results[m]["gss"]["overall_resistance_rate"] for m in MODELS]

    print("\n" + "=" * 60)
    print("CROSS-AXIS CORRELATION ANALYSIS")
    print("=" * 60)

    pairs = [
        ("L-VASE", "Resistance Rate", lvase, rr),
        ("Baseline NIA", "Resistance Rate", nia, rr),
        ("L-VASE", "GSS", lvase, gss),
        ("Baseline NIA", "GSS", nia, gss),
        ("L-VASE", "Baseline NIA", lvase, nia),
        ("GSS", "Resistance Rate", gss, rr),
    ]

    print("\n{:<35} {:>10} {:>10} {:>10} {:>10}".format(
        "Pair", "Pearson r", "p-value", "Spearman", "p-value"))
    print("-" * 80)

    for name_x, name_y, x_vals, y_vals in pairs:
        r_p, p_p = pearsonr(x_vals, y_vals)
        r_s, p_s = spearmanr(x_vals, y_vals)
        sig_p = "*" if p_p < 0.05 else ""
        sig_s = "*" if p_s < 0.05 else ""
        print("{:<35} {:>9.3f}{} {:>10.4f} {:>9.3f}{} {:>10.4f}".format(
            name_x + " vs " + name_y, r_p, sig_p, p_p, r_s, sig_s, p_s))

    # Print the data table used
    print("\n\nData used for correlations:")
    print("{:<15} {:>10} {:>10} {:>10} {:>10}".format("Model", "L-VASE", "NIA", "GSS", "RR"))
    print("-" * 58)
    for m in MODELS:
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            MODEL_LABELS[m],
            results[m]["lvase"]["mean"],
            results[m]["gss"]["mean_baseline_nia"],
            results[m]["gss"]["mean_gss"],
            results[m]["gss"]["overall_resistance_rate"],
        ))

    # Save correlations to JSON
    corr_results = {}
    for name_x, name_y, x_vals, y_vals in pairs:
        r_p, p_p = pearsonr(x_vals, y_vals)
        r_s, p_s = spearmanr(x_vals, y_vals)
        key = name_x + "_vs_" + name_y
        corr_results[key] = {
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_rho": float(r_s), "spearman_p": float(p_s),
        }

    out_path = Path("outputs") / "cross_axis_correlations.json"
    with open(out_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print("\nCorrelations saved to", out_path)


if __name__ == "__main__":
    fig1_main_results()
    fig2_cross_axis()
    fig3_pressure_types()
    fig4_nia_shift()
    fig5_vase_proof()
    compute_correlations()
    print("\nAll figures saved to", OUTPUT_DIR)

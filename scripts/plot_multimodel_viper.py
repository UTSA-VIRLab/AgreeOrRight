#!/usr/bin/env python3
"""
Generate comparative visualizations for multi-model VIPER sycophancy audit.
Produces publication-quality figures for the CVPR 2026 manuscript.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_ROOT

FIGURES_DIR = OUTPUT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Pressure group ordering and colors
PRESSURE_GROUPS = {
    "strong_authority": [
        "senior_radiologist_disagrees", "panel_consensus", "acr_guidelines",
        "attending_correction",
    ],
    "moderate_authority": ["doctor_disagrees", "specialist_opinion"],
    "uncertainty_doubt": ["radiologist_not_sure", "doctor_confused", "someone_disagrees"],
    "social_pressure": ["most_people_say", "ai_consensus", "student_question"],
    "emotional_urgency": ["patient_safety", "legal_liability"],
}

GROUP_COLORS = {
    "strong_authority": "#d32f2f",
    "moderate_authority": "#f57c00",
    "uncertainty_doubt": "#fbc02d",
    "social_pressure": "#388e3c",
    "emotional_urgency": "#7b1fa2",
}

MODEL_SHORT_NAMES = {
    "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-NeXT",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL",
    "HuggingFaceM4/idefics2-8b": "IDEFICS2",
    "microsoft/Phi-3.5-vision-instruct": "Phi-3.5-V",
}


def load_all_viper_results():
    """Load all available VIPER result files."""
    results = {}
    for f in OUTPUT_ROOT.glob("viper_*.json"):
        if f.name == "viper_results.json":
            continue  # skip old single-model results
        with open(f) as fh:
            data = json.load(fh)
        model_id = data.get("model", f.stem)
        short_name = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        results[short_name] = data
    return results


def plot_heatmap(results: dict):
    """Plot Model × Template resistance rate heatmap."""
    all_templates = list(list(results.values())[0]["by_template"].keys())
    models = list(results.keys())

    matrix = np.zeros((len(models), len(all_templates)))
    for i, model in enumerate(models):
        for j, template in enumerate(all_templates):
            if template in results[model]["by_template"]:
                matrix[i, j] = results[model]["by_template"][template]["resistance_rate"]

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(all_templates)))
    ax.set_xticklabels([t.replace("_", "\n") for t in all_templates],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(all_templates)):
            val = matrix[i, j] * 100
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Resistance Rate (%)", shrink=0.8)
    ax.set_title("VIPER: Sycophancy Resistance Rate by Model × Injection Type", fontsize=13)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "viper_multimodel_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "viper_multimodel_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap")


def plot_group_bars(results: dict):
    """Plot grouped bar chart: Model × Pressure Group."""
    models = list(results.keys())
    groups = list(PRESSURE_GROUPS.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(groups))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        group_rates = []
        for group in groups:
            if group in results[model].get("by_pressure_group", {}):
                group_rates.append(
                    results[model]["by_pressure_group"][group]["resistance_rate"] * 100
                )
            else:
                group_rates.append(0)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, group_rates, width, label=model, alpha=0.85)
        for bar, rate in zip(bars, group_rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Pressure Group", fontsize=12)
    ax.set_ylabel("Resistance Rate (%)", fontsize=12)
    ax.set_title("VIPER: Sycophancy Resistance by Pressure Group Across Models", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in groups])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "viper_multimodel_groups.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "viper_multimodel_groups.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved group bars")


def plot_overall_comparison(results: dict):
    """Plot overall resistance rate comparison across models."""
    models = list(results.keys())
    overall_rr = [results[m]["overall"]["resistance_rate"] * 100 for m in models]

    # Sort by resistance rate
    sorted_pairs = sorted(zip(models, overall_rr), key=lambda x: x[1])
    models_sorted, rr_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.array(rr_sorted) / 100)
    bars = ax.barh(models_sorted, rr_sorted, color=colors, edgecolor="white", height=0.6)

    for bar, rate in zip(bars, rr_sorted):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", ha="left", va="center", fontweight="bold", fontsize=11)

    ax.set_xlabel("Overall Resistance Rate (%)", fontsize=12)
    ax.set_title("VIPER: Overall Sycophancy Resistance Comparison", fontsize=13)
    ax.set_xlim(0, max(rr_sorted) + 15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "viper_multimodel_overall.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "viper_multimodel_overall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved overall comparison")


def plot_radar(results: dict):
    """Radar/spider chart of resistance by pressure group per model."""
    groups = list(PRESSURE_GROUPS.keys())
    n_groups = len(groups)
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    model_colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (model, data), color in zip(results.items(), model_colors):
        values = []
        for group in groups:
            if group in data.get("by_pressure_group", {}):
                values.append(data["by_pressure_group"][group]["resistance_rate"] * 100)
            else:
                values.append(0)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([g.replace("_", "\n").title() for g in groups], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("VIPER: Resistance Profile by Pressure Group", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "viper_multimodel_radar.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "viper_multimodel_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved radar chart")


def generate_latex_table(results: dict):
    """Generate LaTeX table for the manuscript."""
    groups = list(PRESSURE_GROUPS.keys())
    models = list(results.keys())

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{VIPER Sycophancy Resistance Rates (\%) by model and pressure group. "
        r"Higher values indicate greater robustness to false authoritative corrections.}",
        r"\label{tab:viper_multimodel}",
        r"\begin{tabular}{@{}l" + "c" * len(groups) + "c@{}}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(
            [r"\textbf{" + g.replace("_", " ").title().replace(" ", r"\ ") + "}"
             for g in groups]
        ) + r" & \textbf{Overall} \\",
        r"\midrule",
    ]

    for model in models:
        data = results[model]
        row = [model]
        for group in groups:
            if group in data.get("by_pressure_group", {}):
                rr = data["by_pressure_group"][group]["resistance_rate"] * 100
                row.append(f"{rr:.1f}")
            else:
                row.append("--")
        overall = data["overall"]["resistance_rate"] * 100
        row.append(f"{overall:.1f}")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    table_str = "\n".join(lines)
    table_path = OUTPUT_ROOT / "viper_multimodel_table.tex"
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"Saved LaTeX table to {table_path}")
    return table_str


def main():
    results = load_all_viper_results()

    if not results:
        print("No multi-model VIPER results found yet.")
        return

    print(f"Found results for {len(results)} models: {list(results.keys())}")

    for model, data in results.items():
        overall = data["overall"]["resistance_rate"] * 100
        print(f"  {model}: Overall RR = {overall:.1f}%")

    plot_heatmap(results)
    plot_group_bars(results)
    plot_overall_comparison(results)
    plot_radar(results)
    latex = generate_latex_table(results)
    print("\nLaTeX table:\n")
    print(latex)


if __name__ == "__main__":
    main()

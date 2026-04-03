#!/usr/bin/env python3
"""Merge SLAKE evaluation shard files into complete per-model result files.

Shard files use local 0-based indices. This script remaps them to global
indices based on the dataset slice encoded in each filename, deduplicates
overlapping entries (keeping the one from the more specific/later shard),
recomputes aggregate statistics, and writes unified result files.
"""

import json
import glob
import re
import os
import sys
import numpy as np
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
N_TOTAL = 1061  # SLAKE test set size

# ── Shard definitions per model ──────────────────────────────────────────
# Each entry: (filename_pattern, global_offset)
# Offset is deduced from the filename slice or from log analysis.

SHARD_DEFS = {
    "medgemma": [
        ("eval_medgemma_slake.json",                     0),      # syco 0-399
        ("eval_medgemma_slake_slake_lvase_0_399.json",   0),      # lvase 0-399
        ("eval_medgemma_slake_slake_rem_shard0.json",    350),    # 350-491 both
        ("eval_medgemma_slake_slake_rem_shard1.json",    492),    # 492-633 partial lvase
        ("eval_medgemma_slake_slake_rem_shard2.json",    634),    # 634-775 partial lvase
        ("eval_medgemma_slake_slake_rem_shard3.json",    776),    # 776-917 partial lvase
        ("eval_medgemma_slake_slake_rem_shard4.json",    918),    # 918-1060 partial lvase
        ("eval_medgemma_slake_slake_ccs_492_499.json",   492),    # syco 492-499
        ("eval_medgemma_slake_slake_500_1060.json",      500),    # 500-1060 both
    ],
    "medvlm_r1": [
        ("eval_medvlm_r1_slake.json",                     0),    # syco 0-399
        ("eval_medvlm_r1_slake_slake_lvase_0_499.json",   0),    # lvase 0-499
        ("eval_medvlm_r1_slake_slake_ccs_400_499.json",   400),  # syco 400-499
        ("eval_medvlm_r1_slake_slake_500_1060.json",      500),  # 500-1060 both
        ("eval_medvlm_r1_slake_from540a.json",             540),  # partial lvase
        ("eval_medvlm_r1_slake_from540b.json",             540),  # partial lvase
    ],
    "qwen3_vl": [
        ("eval_qwen3_vl_slake.json",                       0),   # syco 0-399
        ("eval_qwen3_vl_slake_slake_lvase_0_399.json",     0),   # lvase 0-399
        ("eval_qwen3_vl_slake_slake_lvase_400_469.json",   400), # lvase 400-469
        ("eval_qwen3_vl_slake_slake_ccs_400_469.json",     400), # syco 400-469
        ("eval_qwen3_vl_slake_slake_470_499.json",         470), # 470-499 both
        ("eval_qwen3_vl_slake_slake_500_1060.json",        500), # 500-1060 both
    ],
}

# Model metadata
MODEL_META = {
    "medgemma": {
        "model_key": "medgemma",
        "model_id": "google/medgemma-4b-it",
        "short_name": "MedGemma",
    },
    "medvlm_r1": {
        "model_key": "medvlm-r1",
        "model_id": "TingchenWu/MedVLM-R1",
        "short_name": "MedVLM-R1",
    },
    "qwen3_vl": {
        "model_key": "qwen3-vl",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short_name": "Qwen3-VL",
    },
}


def load_shard(path):
    with open(path) as f:
        return json.load(f)


def merge_model(model_name):
    """Merge all shards for a single model, return unified result dict."""
    defs = SHARD_DEFS[model_name]
    meta = MODEL_META[model_name]

    # Collect per-image LVASE and per-case sycophancy keyed by global index
    lvase_by_idx = {}   # global_idx -> {image_idx, lvase_score}
    syco_by_idx = {}    # global_idx -> per_case dict

    for fname, offset in defs:
        path = OUTPUTS / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue

        data = load_shard(path)
        loaded_lvase = 0
        loaded_syco = 0

        # LVASE items
        if "lvase" in data and "per_image" in data["lvase"]:
            for item in data["lvase"]["per_image"]:
                local_idx = item["image_idx"]
                global_idx = local_idx + offset
                if global_idx < N_TOTAL:
                    # Keep entry (later shards overwrite earlier ones on conflict)
                    lvase_by_idx[global_idx] = {
                        "image_idx": global_idx,
                        "lvase_score": item["lvase_score"],
                    }
                    loaded_lvase += 1

        # Sycophancy items
        if "sycophancy" in data and "per_case" in data["sycophancy"]:
            for item in data["sycophancy"]["per_case"]:
                local_idx = item["case_idx"]
                global_idx = local_idx + offset
                if global_idx < N_TOTAL:
                    new_item = dict(item)
                    new_item["case_idx"] = global_idx
                    syco_by_idx[global_idx] = new_item
                    loaded_syco += 1

        if loaded_lvase or loaded_syco:
            print(f"  {fname}: +{loaded_lvase} lvase, +{loaded_syco} syco (offset={offset})")

    # Sort by global index
    lvase_list = [lvase_by_idx[i] for i in sorted(lvase_by_idx)]
    syco_list = [syco_by_idx[i] for i in sorted(syco_by_idx)]

    # Compute LVASE aggregates
    lvase_scores = [x["lvase_score"] for x in lvase_list]
    lvase_agg = {}
    if lvase_scores:
        lvase_agg = {
            "n_images": len(lvase_scores),
            "n_samples": 5,  # consistent with existing files
            "alpha": 0.5,
            "mean": float(np.mean(lvase_scores)),
            "std": float(np.std(lvase_scores)),
            "per_image": lvase_list,
        }

    # Compute sycophancy aggregates
    syco_agg = {}
    if syco_list:
        n_cases = len(syco_list)
        # Per-pressure aggregates
        pressure_types = ["expert_correction", "consensus", "authority"]
        per_pressure = {}
        for pt in pressure_types:
            resisted = []
            ccs_vals = []
            for item in syco_list:
                if "pressures" in item and pt in item["pressures"]:
                    p = item["pressures"][pt]
                    resisted.append(p.get("resisted", False))
                    ccs_vals.append(p.get("ccs_contribution", 0))
            if resisted:
                per_pressure[pt] = {
                    "resistance_rate": float(np.mean(resisted)),
                    "mean_ccs": float(np.mean(ccs_vals)),
                    "n_cases": len(resisted),
                }

        # Overall stats
        all_resisted = []
        all_ccs = []
        all_conf = []
        for item in syco_list:
            conf = item.get("baseline_confidence", 0)
            all_conf.append(conf)
            if "pressures" in item:
                for pt in pressure_types:
                    if pt in item["pressures"]:
                        all_resisted.append(item["pressures"][pt].get("resisted", False))
                        all_ccs.append(item["pressures"][pt].get("ccs_contribution", 0))

        overall_rr = float(np.mean(all_resisted)) if all_resisted else 0.0
        overall_ccs = float(np.mean(all_ccs)) if all_ccs else 0.0
        mean_conf = float(np.mean(all_conf)) if all_conf else 0.0

        # Confidence-resistance correlation
        conf_resist_corr = 0.0
        conf_resist_pval = 1.0
        try:
            from scipy.stats import pearsonr
            per_case_resist = []
            per_case_conf = []
            for item in syco_list:
                c = item.get("baseline_confidence", 0)
                r_vals = []
                if "pressures" in item:
                    for pt in pressure_types:
                        if pt in item["pressures"]:
                            r_vals.append(item["pressures"][pt].get("resisted", False))
                if r_vals:
                    per_case_resist.append(float(np.mean(r_vals)))
                    per_case_conf.append(c)
            if len(per_case_resist) > 2:
                corr, pval = pearsonr(per_case_conf, per_case_resist)
                conf_resist_corr = float(corr)
                conf_resist_pval = float(pval)
        except Exception:
            pass

        syco_agg = {
            "n_cases": n_cases,
            "overall_resistance_rate": overall_rr,
            "overall_ccs": overall_ccs,
            "mean_baseline_confidence": mean_conf,
            "confidence_resistance_corr": conf_resist_corr,
            "confidence_resistance_pval": conf_resist_pval,
            "per_pressure": per_pressure,
            "per_case": syco_list,
        }

    # Build final result
    result = {
        "model_key": meta["model_key"],
        "model_id": meta["model_id"],
        "short_name": meta["short_name"],
        "dataset": "slake",
        "split": "test",
        "n_images": N_TOTAL,
    }
    if lvase_agg:
        result["lvase"] = lvase_agg
    if syco_agg:
        result["sycophancy"] = syco_agg

    return result, len(lvase_by_idx), len(syco_by_idx)


def main():
    for model_name in ["medgemma", "medvlm_r1", "qwen3_vl"]:
        print(f"\n{'='*60}")
        print(f"Merging {MODEL_META[model_name]['short_name']} SLAKE shards...")
        print(f"{'='*60}")

        result, n_lvase, n_syco = merge_model(model_name)

        # Report coverage
        missing_lvase = N_TOTAL - n_lvase
        missing_syco = N_TOTAL - n_syco
        print(f"\n  Coverage: LVASE {n_lvase}/{N_TOTAL}, Sycophancy {n_syco}/{N_TOTAL}")
        if missing_lvase:
            all_idx = set(range(N_TOTAL))
            have = {x["image_idx"] for x in result.get("lvase", {}).get("per_image", [])}
            missing = sorted(all_idx - have)
            print(f"  Missing LVASE indices ({missing_lvase}): {missing[:20]}{'...' if len(missing)>20 else ''}")
        if missing_syco:
            all_idx = set(range(N_TOTAL))
            have = {x["case_idx"] for x in result.get("sycophancy", {}).get("per_case", [])}
            missing = sorted(all_idx - have)
            print(f"  Missing SYCO indices ({missing_syco}): {missing[:20]}{'...' if len(missing)>20 else ''}")

        # Back up original and write merged
        out_path = OUTPUTS / f"eval_{model_name}_slake.json"
        bak_path = OUTPUTS / f"eval_{model_name}_slake.json.bak"
        if out_path.exists() and not bak_path.exists():
            os.rename(out_path, bak_path)
            print(f"  Backed up original to {bak_path.name}")

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote merged results to {out_path.name}")

        # Summary stats
        if "lvase" in result:
            lv = result["lvase"]
            print(f"  LVASE: mean={lv['mean']:.4f}, std={lv['std']:.4f}")
        if "sycophancy" in result:
            sy = result["sycophancy"]
            print(f"  Sycophancy: resist={sy['overall_resistance_rate']:.4f}, CCS={sy['overall_ccs']:.4f}")


if __name__ == "__main__":
    main()

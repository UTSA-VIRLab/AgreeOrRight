#!/usr/bin/env python3
"""Merge shard results from parallel runs into a single result file."""

import argparse
import json
import glob
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_ROOT


def merge(model_key, dataset_key, n_shards):
    safe_name = model_key.replace("/", "_").replace("-", "_")
    pattern = OUTPUT_ROOT / f"eval_{safe_name}_{dataset_key}_shard*.json"
    files = sorted(glob.glob(str(pattern)))

    if not files:
        print(f"No shard files found matching {pattern}")
        return

    print(f"Found {len(files)} shard files")

    merged_lvase_images = []
    merged_syc_cases = []
    first = None

    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if first is None:
            first = d

        if "lvase" in d and "per_image" in d["lvase"]:
            merged_lvase_images.extend(d["lvase"]["per_image"])
        if "sycophancy" in d and "per_case" in d["sycophancy"]:
            merged_syc_cases.extend(d["sycophancy"]["per_case"])

    result = {
        "model_key": first["model_key"],
        "model_id": first["model_id"],
        "short_name": first["short_name"],
        "dataset": dataset_key,
        "split": first["split"],
        "n_images": len(merged_lvase_images) or len(merged_syc_cases),
    }

    # Merge L-VASE
    if merged_lvase_images:
        scores = [r["lvase_score"] for r in merged_lvase_images]
        result["lvase"] = {
            "n_images": len(scores),
            "n_samples": first.get("lvase", {}).get("n_samples", 5),
            "alpha": first.get("lvase", {}).get("alpha", 0.5),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "per_image": merged_lvase_images,
        }
        print(f"  L-VASE: {len(scores)} images, mean={np.mean(scores):.4f}")

    # Merge Sycophancy
    if merged_syc_cases:
        from scipy.stats import pointbiserialr

        all_resist = [c["pressures"][p]["resisted"]
                      for c in merged_syc_cases for p in c["pressures"]]
        all_ccs = [c["pressures"][p]["ccs_contribution"]
                   for c in merged_syc_cases for p in c["pressures"]]
        all_conf = [c["baseline_confidence"] for c in merged_syc_cases]

        # Per pressure
        pressure_types = list(merged_syc_cases[0]["pressures"].keys())
        per_pressure = {}
        for ptype in pressure_types:
            resisted = [c["pressures"][ptype]["resisted"]
                        for c in merged_syc_cases if ptype in c["pressures"]]
            ccs_vals = [c["pressures"][ptype]["ccs_contribution"]
                        for c in merged_syc_cases if ptype in c["pressures"]]
            per_pressure[ptype] = {
                "resistance_rate": float(np.mean(resisted)),
                "mean_ccs": float(np.mean(ccs_vals)),
                "n_cases": len(resisted),
            }

        # Correlation
        case_resist = []
        case_conf = []
        for c in merged_syc_cases:
            r = np.mean([c["pressures"][p]["resisted"] for p in c["pressures"]])
            case_resist.append(r)
            case_conf.append(c["baseline_confidence"])
        try:
            resist_binary = [1 if r > 0.5 else 0 for r in case_resist]
            if len(set(resist_binary)) > 1:
                corr, p_val = pointbiserialr(resist_binary, case_conf)
            else:
                corr, p_val = 0.0, 1.0
        except Exception:
            corr, p_val = 0.0, 1.0

        result["sycophancy"] = {
            "n_cases": len(merged_syc_cases),
            "overall_resistance_rate": float(np.mean(all_resist)),
            "overall_ccs": float(np.mean(all_ccs)),
            "mean_baseline_confidence": float(np.mean(all_conf)),
            "confidence_resistance_corr": float(corr),
            "confidence_resistance_pval": float(p_val),
            "per_pressure": per_pressure,
            "per_case": merged_syc_cases,
        }
        print(f"  Sycophancy: {len(merged_syc_cases)} cases, resist={np.mean(all_resist)*100:.1f}%, CCS={np.mean(all_ccs):.4f}")

    out_path = OUTPUT_ROOT / f"eval_{safe_name}_{dataset_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Merged result saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-shards", type=int, default=8)
    args = parser.parse_args()
    merge(args.model, args.dataset, args.n_shards)

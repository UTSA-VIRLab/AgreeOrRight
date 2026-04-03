#!/usr/bin/env python3
"""Compare old (400-image) vs new (1061-image) SLAKE results."""
import json

models = ['medgemma', 'medvlm_r1', 'qwen3_vl']
meta = {'medgemma': 'MedGemma', 'medvlm_r1': 'MedVLM-R1', 'qwen3_vl': 'Qwen3-VL'}

for m in models:
    with open(f'outputs/eval_{m}_slake.json.bak') as f:
        old = json.load(f)
    with open(f'outputs/eval_{m}_slake.json') as f:
        new = json.load(f)

    old_n = old.get('n_images', 0)
    new_n = new.get('n_images', 0)

    print(f'=== {meta[m]} (n={old_n} -> {new_n}) ===')

    # LVASE
    old_lv = old.get('lvase', {})
    new_lv = new.get('lvase', {})
    old_lv_mean = old_lv.get('mean', None)
    new_lv_mean = new_lv.get('mean', None)
    old_lv_std = old_lv.get('std', None)
    new_lv_std = new_lv.get('std', None)
    old_lv_n = old_lv.get('n_images', len(old_lv.get('per_image', [])))
    new_lv_n = new_lv.get('n_images', len(new_lv.get('per_image', [])))

    print(f'  LVASE:')
    if old_lv_mean is not None:
        delta = new_lv_mean - old_lv_mean
        print(f'    Old ({old_lv_n:>4}): mean={old_lv_mean:.4f} +/- {old_lv_std:.4f}')
        print(f'    New ({new_lv_n:>4}): mean={new_lv_mean:.4f} +/- {new_lv_std:.4f}')
        pct = delta / old_lv_mean * 100 if old_lv_mean else 0
        print(f'    Delta:       {delta:+.4f} ({pct:+.1f}%)')
    else:
        print(f'    Old: no LVASE in main file (was in shards only)')
        print(f'    New ({new_lv_n:>4}): mean={new_lv_mean:.4f} +/- {new_lv_std:.4f}')

    # Sycophancy
    old_sy = old.get('sycophancy', {})
    new_sy = new.get('sycophancy', {})
    old_sy_n = old_sy.get('n_cases', 0)
    new_sy_n = new_sy.get('n_cases', 0)

    print(f'  Sycophancy:')
    for metric, label in [('overall_resistance_rate', 'Resistance Rate'),
                           ('overall_ccs', 'CCS'),
                           ('mean_baseline_confidence', 'Mean Confidence')]:
        old_v = old_sy.get(metric, None)
        new_v = new_sy.get(metric, None)
        if old_v is not None and new_v is not None:
            delta = new_v - old_v
            if abs(old_v) > 1e-9:
                pct = f' ({delta/old_v*100:+.1f}%)'
            else:
                pct = ''
            print(f'    {label}:')
            print(f'      Old ({old_sy_n:>4}): {old_v:.4f}')
            print(f'      New ({new_sy_n:>4}): {new_v:.4f}')
            print(f'      Delta:       {delta:+.4f}{pct}')

    # Per-pressure breakdown
    print(f'  Per-Pressure:')
    print(f'    {"Type":>20}  {"RR old":>8} {"RR new":>8} {"dRR":>8}  {"CCS old":>8} {"CCS new":>8} {"dCCS":>8}')
    for pt in ['expert_correction', 'consensus', 'authority']:
        old_pp = old_sy.get('per_pressure', {}).get(pt, {})
        new_pp = new_sy.get('per_pressure', {}).get(pt, {})
        old_rr = old_pp.get('resistance_rate', None)
        new_rr = new_pp.get('resistance_rate', None)
        old_ccs = old_pp.get('mean_ccs', None)
        new_ccs = new_pp.get('mean_ccs', None)
        if old_rr is not None and new_rr is not None:
            print(f'    {pt:>20}  {old_rr:8.4f} {new_rr:8.4f} {new_rr-old_rr:+8.4f}  {old_ccs:8.4f} {new_ccs:8.4f} {new_ccs-old_ccs:+8.4f}')

    print()

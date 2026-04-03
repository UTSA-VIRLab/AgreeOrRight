import json, numpy as np, sys

base = '/raid/den365/AgenticMedXAI_CVPR2026/outputs/'

def compute_R_CCS(per_case):
    all_resist = [c['pressures'][p]['resisted'] for c in per_case for p in c['pressures']]
    all_ccs = [c['pressures'][p]['ccs_contribution'] for c in per_case for p in c['pressures']]
    R = float(np.mean(all_resist)) if all_resist else 0.0
    CCS = float(np.mean(all_ccs)) if all_ccs else 0.0
    return R, CCS

def compute_per_pressure_R(per_case):
    ptypes = {}
    for c in per_case:
        for p, v in c['pressures'].items():
            if p not in ptypes:
                ptypes[p] = []
            ptypes[p].append(v['resisted'])
    return {p: float(np.mean(v))*100 for p, v in ptypes.items()}

def compute_confidence(per_case):
    return float(np.mean([c['baseline_confidence'] for c in per_case]))

def csi(lv, R, ccs):
    return (max(1-lv, 0.01) * max(R, 0.01) * max(1-ccs, 0.01)) ** (1/3)

results = {}

# ==================== VQA-RAD (N=451) ====================
print("Computing VQA-RAD...")
vqa_rad = {}
for model, fname in [('LLaVA-1.5', 'eval_llava_1.5_vqa_rad.json'),
                      ('Qwen3-VL', 'eval_qwen3_vl_vqa_rad.json'),
                      ('IDEFICS2', 'eval_idefics2_vqa_rad.json'),
                      ('LLaVA-Med', 'eval_llava_med_vqa_rad.json'),
                      ('MedVLM-R1', 'eval_medvlm_r1_vqa_rad.json'),
                      ('MedGemma', 'eval_medgemma_vqa_rad.json')]:
    d = json.load(open(base + fname))
    lv = float(np.mean([x['lvase_score'] for x in d['lvase']['per_image']]))
    pc = d['sycophancy']['per_case']
    R, CCS = compute_R_CCS(pc)
    CSI = csi(lv, R, CCS)
    pp = compute_per_pressure_R(pc)
    conf = compute_confidence(pc)
    vqa_rad[model] = {'lv': lv, 'R': R, 'CCS': CCS, 'CSI': CSI, 'pp': pp, 'conf': conf}
results['vqa_rad'] = vqa_rad

# ==================== SLAKE (N=500) ====================
print("Computing SLAKE...")
slake = {}

# Full 1061 files, take first 500
for model, fname in [('LLaVA-1.5', 'eval_llava_1.5_slake.json'),
                      ('IDEFICS2', 'eval_idefics2_slake.json'),
                      ('LLaVA-Med', 'eval_llava_med_slake.json')]:
    d = json.load(open(base + fname))
    pc = d['sycophancy']['per_case'][:500]
    R, CCS = compute_R_CCS(pc)
    pi = d['lvase']['per_image'][:500]
    lv = float(np.mean([x['lvase_score'] for x in pi]))
    CSI = csi(lv, R, CCS)
    pp = compute_per_pressure_R(pc)
    conf = compute_confidence(pc)
    slake[model] = {'lv': lv, 'R': R, 'CCS': CCS, 'CSI': CSI, 'pp': pp, 'conf': conf}

# Qwen3-VL
d_q_main = json.load(open(base + 'eval_qwen3_vl_slake.json'))
d_q_470 = json.load(open(base + 'eval_qwen3_vl_slake_slake_470_499.json'))
d_q_gap = json.load(open(base + 'eval_qwen3_vl_slake_slake_ccs_400_469.json'))
pi_q_470 = d_q_470['lvase']['per_image'][:30]
lv_q = (400*0.3037 + sum(x['lvase_score'] for x in pi_q_470)) / 430
all_pc_q = list(d_q_main['sycophancy']['per_case'][:400]) + list(d_q_gap['sycophancy']['per_case'][:70]) + list(d_q_470['sycophancy']['per_case'][:30])
R_q, CCS_q = compute_R_CCS(all_pc_q)
CSI_q = csi(lv_q, R_q, CCS_q)
pp_q = compute_per_pressure_R(all_pc_q)
conf_q = compute_confidence(all_pc_q)
slake['Qwen3-VL'] = {'lv': lv_q, 'R': R_q, 'CCS': CCS_q, 'CSI': CSI_q, 'pp': pp_q, 'conf': conf_q}

# MedGemma
d1 = json.load(open(base + 'eval_medgemma_slake.json'))
d2 = json.load(open(base + 'eval_medgemma_slake_slake_rem_shard0.json'))
d3 = json.load(open(base + 'eval_medgemma_slake_slake_ccs_492_499.json'))
all_pc_mg = list(d1['sycophancy']['per_case'][:400]) + list(d2['sycophancy']['per_case'][:92]) + list(d3['sycophancy']['per_case'][:8])
R_mg, CCS_mg = compute_R_CCS(all_pc_mg)
pi_s0 = d2['lvase']['per_image'][:100]
lv_mg = (400*0.4767 + sum(x['lvase_score'] for x in pi_s0)) / 500
CSI_mg = csi(lv_mg, R_mg, CCS_mg)
pp_mg = compute_per_pressure_R(all_pc_mg)
conf_mg = compute_confidence(all_pc_mg)
slake['MedGemma'] = {'lv': lv_mg, 'R': R_mg, 'CCS': CCS_mg, 'CSI': CSI_mg, 'pp': pp_mg, 'conf': conf_mg}

# MedVLM-R1
d_mv1 = json.load(open(base + 'eval_medvlm_r1_slake.json'))
d_mv2 = json.load(open(base + 'eval_medvlm_r1_slake_slake_ccs_400_499.json'))
all_pc_mv = list(d_mv1['sycophancy']['per_case'][:400]) + list(d_mv2['sycophancy']['per_case'][:100])
R_mv, CCS_mv = compute_R_CCS(all_pc_mv)
lv_mv = 0.7054
CSI_mv = csi(lv_mv, R_mv, CCS_mv)
pp_mv = compute_per_pressure_R(all_pc_mv)
conf_mv = compute_confidence(all_pc_mv)
slake['MedVLM-R1'] = {'lv': lv_mv, 'R': R_mv, 'CCS': CCS_mv, 'CSI': CSI_mv, 'pp': pp_mv, 'conf': conf_mv}

results['slake'] = slake

# ==================== PathVQA (N=200) ====================
print("Computing PathVQA...")
pathvqa = {}

# L40S shard sums (precomputed for idx 10-149)
l40s_sums = {
    'LLaVA-1.5': {'lv_sum': 147.669798, 'lv_n': 140, 'r_sum': 4, 'ccs_sum': 314.663551, 'p_n': 420},
    'IDEFICS2':  {'lv_sum': 133.873263, 'lv_n': 145, 'r_sum': 43, 'ccs_sum': 305.882588, 'p_n': 435},
    'MedVLM-R1': {'lv_sum': 119.373738, 'lv_n': 140, 'r_sum': 55, 'ccs_sum': 251.892022, 'p_n': 420},
}

for model in ['LLaVA-1.5', 'IDEFICS2', 'MedVLM-R1']:
    prefix = {'LLaVA-1.5': 'llava_1.5', 'IDEFICS2': 'idefics2', 'MedVLM-R1': 'medvlm_r1'}[model]

    # idx 0-9/0-4 from H200
    d_h200 = json.load(open(base + f'eval_{prefix}_pathvqa.json'))
    h200_lv = d_h200['lvase']['per_image']
    h200_pc = d_h200.get('sycophancy', {}).get('per_case', [])

    s = l40s_sums[model]

    lv_sum_0_149 = sum(x['lvase_score'] for x in h200_lv) + s['lv_sum']
    lv_n_0_149 = len(h200_lv) + s['lv_n']
    r_sum_0_149 = sum(1 for c in h200_pc for p in c.get('pressures',{}).values() if p.get('resisted')) + s['r_sum']
    ccs_sum_0_149 = sum(p['ccs_contribution'] for c in h200_pc for p in c.get('pressures',{}).values()) + s['ccs_sum']
    p_n_0_149 = sum(len(c.get('pressures',{})) for c in h200_pc) + s['p_n']

    # idx 250-299
    d_250 = json.load(open(base + f'eval_{prefix}_pathvqa_pathvqa_250_399.json'))
    lv_250 = d_250['lvase']['per_image'][:50]
    lv_sum_250 = sum(x['lvase_score'] for x in lv_250)

    if model == 'IDEFICS2':
        ccs_250_pc = d_250['sycophancy']['per_case'][:50]
    else:
        d_ccs_250 = json.load(open(base + f'eval_{prefix}_pathvqa_pathvqa_ccs_250_299.json'))
        ccs_250_pc = d_ccs_250['sycophancy']['per_case'][:50]

    r_sum_250 = sum(1 for c in ccs_250_pc for p in c['pressures'].values() if p['resisted'])
    ccs_sum_250 = sum(p['ccs_contribution'] for c in ccs_250_pc for p in c['pressures'].values())
    p_n_250 = sum(len(c['pressures']) for c in ccs_250_pc)

    total_lv = (lv_sum_0_149 + lv_sum_250) / (lv_n_0_149 + 50)
    total_R = (r_sum_0_149 + r_sum_250) / (p_n_0_149 + p_n_250)
    total_CCS = (ccs_sum_0_149 + ccs_sum_250) / (p_n_0_149 + p_n_250)
    CSI = csi(total_lv, total_R, total_CCS)

    # Per-pressure from available per_case
    pp_pc = list(h200_pc) + list(ccs_250_pc)
    pp = compute_per_pressure_R(pp_pc)
    conf = compute_confidence(pp_pc)
    pathvqa[model] = {'lv': total_lv, 'R': total_R, 'CCS': total_CCS, 'CSI': CSI, 'pp': pp, 'conf': conf}

# LLaVA-Med
d_lm = json.load(open(base + 'eval_llava_med_pathvqa.json'))
pc_combined = list(d_lm['sycophancy']['per_case'][:150]) + list(d_lm['sycophancy']['per_case'][250:300])
R_lm, CCS_lm = compute_R_CCS(pc_combined)
pi_combined = d_lm['lvase']['per_image'][:150] + d_lm['lvase']['per_image'][250:300]
lv_lm = float(np.mean([x['lvase_score'] for x in pi_combined]))
CSI_lm = csi(lv_lm, R_lm, CCS_lm)
pp_lm = compute_per_pressure_R(pc_combined)
conf_lm = compute_confidence(pc_combined)
pathvqa['LLaVA-Med'] = {'lv': lv_lm, 'R': R_lm, 'CCS': CCS_lm, 'CSI': CSI_lm, 'pp': pp_lm, 'conf': conf_lm}

# Qwen3-VL
d_q_main = json.load(open(base + 'eval_qwen3_vl_pathvqa.json'))
q_lv_list = list(d_q_main['lvase']['per_image'])
q_pc_list = list(d_q_main['sycophancy']['per_case'])
for i in range(4):
    d = json.load(open(base + f'eval_qwen3_vl_pathvqa_pathvqa_h200_s{i}.json'))
    q_lv_list.extend(d['lvase']['per_image'])
    q_pc_list.extend(d['sycophancy']['per_case'])
d_q_250_lv = json.load(open(base + 'eval_qwen3_vl_pathvqa_pathvqa_250_399.json'))
q_lv_list.extend(d_q_250_lv['lvase']['per_image'][:50])
d_q_250_ccs = json.load(open(base + 'eval_qwen3_vl_pathvqa_pathvqa_ccs_250_299.json'))
q_pc_list.extend(d_q_250_ccs['sycophancy']['per_case'][:50])

lv_q = float(np.mean([x['lvase_score'] for x in q_lv_list]))
R_q, CCS_q = compute_R_CCS(q_pc_list)
CSI_q = csi(lv_q, R_q, CCS_q)
pp_q = compute_per_pressure_R(q_pc_list)
conf_q = compute_confidence(q_pc_list)
pathvqa['Qwen3-VL'] = {'lv': lv_q, 'R': R_q, 'CCS': CCS_q, 'CSI': CSI_q, 'pp': pp_q, 'conf': conf_q}

# MedGemma
d_mg_s0 = json.load(open(base + 'eval_medgemma_pathvqa_shard0.json'))
d_mg_main = json.load(open(base + 'eval_medgemma_pathvqa.json'))
d_mg_ccs_10 = json.load(open(base + 'eval_medgemma_pathvqa_pathvqa_ccs_10_149.json'))
d_mg_250_lv = json.load(open(base + 'eval_medgemma_pathvqa_pathvqa_250_399.json'))
d_mg_ccs_250 = json.load(open(base + 'eval_medgemma_pathvqa_pathvqa_ccs_250_294.json'))
d_mg_295 = json.load(open(base + 'eval_medgemma_pathvqa_pathvqa_295_299.json'))

mg_lv = d_mg_s0['lvase']['per_image'][:150] + d_mg_250_lv['lvase']['per_image'][:45] + d_mg_295['lvase']['per_image'][:5]
mg_pc = list(d_mg_main['sycophancy']['per_case'][:10]) + list(d_mg_ccs_10['sycophancy']['per_case'][:140]) + list(d_mg_ccs_250['sycophancy']['per_case'][:45]) + list(d_mg_295['sycophancy']['per_case'][:5])

lv_mg = float(np.mean([x['lvase_score'] for x in mg_lv]))
R_mg, CCS_mg = compute_R_CCS(mg_pc)
CSI_mg = csi(lv_mg, R_mg, CCS_mg)
pp_mg = compute_per_pressure_R(mg_pc)
conf_mg = compute_confidence(mg_pc)
pathvqa['MedGemma'] = {'lv': lv_mg, 'R': R_mg, 'CCS': CCS_mg, 'CSI': CSI_mg, 'pp': pp_mg, 'conf': conf_mg}

results['pathvqa'] = pathvqa

# ==================== PRINT TABLES ====================
models_order = ['LLaVA-1.5', 'Qwen3-VL', 'IDEFICS2', 'LLaVA-Med', 'MedVLM-R1', 'MedGemma']

print("\n" + "="*80)
print("TABLE 1: Main Results")
print("="*80)
for ds_name, ds_label in [('vqa_rad','VQA-RAD (451)'), ('slake','SLAKE (500)'), ('pathvqa','PathVQA (200)')]:
    print(f"\n  {ds_label}")
    print(f"  {'Model':12s} | {'L-VASE':>7s} | {'R':>6s} | {'CCS':>6s} | {'CSI':>6s}")
    print("  " + "-"*48)
    ds = results[ds_name]
    for m in models_order:
        v = ds[m]
        print(f"  {m:12s} | {v['lv']:7.3f} | {v['R']:6.3f} | {v['CCS']:6.3f} | {v['CSI']:6.3f}")

print("\n" + "="*80)
print("TABLE 2: Per-pressure resistance (%) + confidence")
print("="*80)
for ds_name, ds_label in [('vqa_rad','VQA-RAD'), ('slake','SLAKE'), ('pathvqa','PathVQA')]:
    print(f"\n  {ds_label}")
    print(f"  {'Model':12s} | {'Expert':>7s} | {'Cons.':>7s} | {'Auth.':>7s} | {'Conf.':>6s}")
    print("  " + "-"*50)
    ds = results[ds_name]
    for m in models_order:
        v = ds[m]
        pp = v['pp']
        e = pp.get('expert_correction', 0)
        c = pp.get('consensus', 0)
        a = pp.get('authority', 0)
        print(f"  {m:12s} | {e:7.1f} | {c:7.1f} | {a:7.1f} | {v['conf']:6.3f}")

# Save
def to_native(obj):
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_native(v) for v in obj]
    return obj

with open(base + 'final_results_all.json', 'w') as f:
    json.dump(to_native(results), f, indent=2)
print("\nSaved to final_results_all.json")

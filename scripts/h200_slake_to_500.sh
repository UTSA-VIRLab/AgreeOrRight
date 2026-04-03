#!/bin/bash
# Push SLAKE from N=470 to N=500 for all 6 models
# Run after Qwen3-VL CCS SLAKE (400) finishes on cuda:5
source /opt/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate med_reasoning
cd /raid/den365/AgenticMedXAI_CVPR2026

echo "=== SLAKE → N=500: 3 jobs on 3 GPUs ==="

# GPU 0: Qwen3-VL full (L-VASE+CCS), idx 470-499 (30 imgs)
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:0 --dataset slake \
  --n-images 30 --start-idx 470 --shard-tag slake_470_499 \
  > outputs/logs/qwen3_vl_slake_470_499.log 2>&1 &
echo "GPU 0: Qwen3-VL full 470-499 (30 imgs)"

# GPU 3: MedVLM-R1 CCS-only, idx 400-499 (100 imgs)
nohup python experiments/run_evaluation.py \
  --model medvlm-r1 --device cuda:3 --dataset slake \
  --n-images 100 --start-idx 400 --ccs-only --shard-tag slake_ccs_400_499 \
  > outputs/logs/medvlm_r1_slake_ccs_400_499.log 2>&1 &
echo "GPU 3: MedVLM-R1 CCS-only 400-499 (100 imgs)"

# GPU 5: MedGemma CCS-only, idx 492-499 (8 imgs)
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:5 --dataset slake \
  --n-images 8 --start-idx 492 --ccs-only --shard-tag slake_ccs_492_499 \
  > outputs/logs/medgemma_slake_ccs_492_499.log 2>&1 &
echo "GPU 5: MedGemma CCS-only 492-499 (8 imgs)"

echo ""
echo "Launched 3 jobs. Bottleneck: Qwen3-VL ~45 min"
echo "Monitor: tail -f outputs/logs/*slake_*_4*.log"

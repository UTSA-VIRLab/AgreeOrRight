#!/bin/bash
# Batch 2: MedGemma CCS-only for idx 250-399
# Run after Batch 1 finishes
source ~/miniconda3/etc/profile.d/conda.sh
conda activate med_reasoning
cd /home/den365/AgenticMedXAI_CVPR2026

echo "=== Batch 2: MedGemma CCS-only 250-399 ==="

# GPU 0: MedGemma CCS-only, idx 250-399 (150 imgs)
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:0 --dataset pathvqa \
  --n-images 150 --start-idx 250 --ccs-only --shard-tag pathvqa_ccs_250_399 \
  > outputs/logs/medgemma_pathvqa_ccs_250_399.log 2>&1 &
echo "GPU 0: MedGemma CCS-only 250-399"

echo ""
echo "Batch 2 launched. Monitor:"
echo "  tail -f outputs/logs/medgemma_pathvqa_ccs_250_399.log"

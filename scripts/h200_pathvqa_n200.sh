#!/bin/bash
# PathVQA N=200: fill gaps for idx 250-299
# Run AFTER current Qwen3-VL and MedGemma jobs finish
source /opt/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate med_reasoning
cd /raid/den365/AgenticMedXAI_CVPR2026

echo "=== PathVQA N=200: 5 jobs on GPUs 1-5 ==="

# GPU 1: MedGemma L-VASE+CCS for idx 295-299 (5 imgs to complete 50)
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:1 --dataset pathvqa \
  --n-images 5 --start-idx 295 --shard-tag pathvqa_295_299 \
  > outputs/logs/medgemma_pathvqa_295_299.log 2>&1 &
echo "GPU 1: MedGemma full 295-299 (5 imgs)"

# GPU 2: LLaVA-1.5 CCS-only, idx 250-299 (50 imgs)
nohup python experiments/run_evaluation.py \
  --model llava-1.5 --device cuda:2 --dataset pathvqa \
  --n-images 50 --start-idx 250 --ccs-only --shard-tag pathvqa_ccs_250_299 \
  > outputs/logs/llava15_pathvqa_ccs_250_299.log 2>&1 &
echo "GPU 2: LLaVA-1.5 CCS-only 250-299 (50 imgs)"

# GPU 3: MedVLM-R1 CCS-only, idx 250-299 (50 imgs)
nohup python experiments/run_evaluation.py \
  --model medvlm-r1 --device cuda:3 --dataset pathvqa \
  --n-images 50 --start-idx 250 --ccs-only --shard-tag pathvqa_ccs_250_299 \
  > outputs/logs/medvlm_r1_pathvqa_ccs_250_299.log 2>&1 &
echo "GPU 3: MedVLM-R1 CCS-only 250-299 (50 imgs)"

# GPU 4: Qwen3-VL CCS-only, idx 250-299 (50 imgs)
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:4 --dataset pathvqa \
  --n-images 50 --start-idx 250 --ccs-only --shard-tag pathvqa_ccs_250_299 \
  > outputs/logs/qwen3_vl_pathvqa_ccs_250_299.log 2>&1 &
echo "GPU 4: Qwen3-VL CCS-only 250-299 (50 imgs)"

# GPU 5: MedGemma CCS-only, idx 250-294 (45 imgs, already have L-VASE)
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:5 --dataset pathvqa \
  --n-images 45 --start-idx 250 --ccs-only --shard-tag pathvqa_ccs_250_294 \
  > outputs/logs/medgemma_pathvqa_ccs_250_294.log 2>&1 &
echo "GPU 5: MedGemma CCS-only 250-294 (45 imgs)"

echo ""
echo "Launched 5 jobs. After completion, combine for N=200."
echo "Monitor: tail -f outputs/logs/*pathvqa_*25*.log"

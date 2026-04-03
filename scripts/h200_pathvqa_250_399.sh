#!/bin/bash
# PathVQA idx 250-399 (150 imgs) on H200 GPUs 1-5
# 5 models: LLaVA-1.5, IDEFICS2, MedVLM-R1, Qwen3-VL, MedGemma
# LLaVA-Med already has full 1000 images — skip
source /opt/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate med_reasoning
cd /raid/den365/AgenticMedXAI_CVPR2026

echo "=== PathVQA 250-399: 5 models on GPUs 1-5 ==="

# GPU 1: LLaVA-1.5 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model llava-1.5 --device cuda:1 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/llava15_pathvqa_250_399.log 2>&1 &
echo "GPU 1: LLaVA-1.5 full 250-399"

# GPU 2: IDEFICS2 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model idefics2 --device cuda:2 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/idefics2_pathvqa_250_399.log 2>&1 &
echo "GPU 2: IDEFICS2 full 250-399"

# GPU 3: MedVLM-R1 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model medvlm-r1 --device cuda:3 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/medvlm_r1_pathvqa_250_399.log 2>&1 &
echo "GPU 3: MedVLM-R1 full 250-399"

# GPU 4: Qwen3-VL full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:4 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/qwen3_vl_pathvqa_250_399.log 2>&1 &
echo "GPU 4: Qwen3-VL full 250-399"

# GPU 5: MedGemma full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:5 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/medgemma_pathvqa_250_399.log 2>&1 &
echo "GPU 5: MedGemma full 250-399"

echo ""
echo "Launched 5 jobs on GPUs 1-5. Monitor:"
echo "  tail -f outputs/logs/*pathvqa_250_399.log"

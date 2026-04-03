#!/bin/bash
# Batch 1: 7 jobs on 8 GPUs
# 4 full runs (idx 250-399) + 3 CCS-only (idx 0-149 missing CCS)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate med_reasoning
cd /home/den365/AgenticMedXAI_CVPR2026

echo "=== Batch 1: 4 full runs (250-399) + 3 CCS-only (0-149) ==="

# GPU 0: LLaVA-1.5 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model llava-1.5 --device cuda:0 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/llava15_pathvqa_250_399.log 2>&1 &
echo "GPU 0: LLaVA-1.5 full 250-399"

# GPU 1: IDEFICS2 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model idefics2 --device cuda:1 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/idefics2_pathvqa_250_399.log 2>&1 &
echo "GPU 1: IDEFICS2 full 250-399"

# GPU 2: MedVLM-R1 full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model medvlm-r1 --device cuda:2 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/medvlm_r1_pathvqa_250_399.log 2>&1 &
echo "GPU 2: MedVLM-R1 full 250-399"

# GPU 3: Qwen3-VL full, idx 250-399
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:3 --dataset pathvqa \
  --n-images 150 --start-idx 250 --shard-tag pathvqa_250_399 \
  > outputs/logs/qwen3_vl_pathvqa_250_399.log 2>&1 &
echo "GPU 3: Qwen3-VL full 250-399"

# GPU 4: IDEFICS2 CCS-only, idx 0-4 (5 imgs)
nohup python experiments/run_evaluation.py \
  --model idefics2 --device cuda:4 --dataset pathvqa \
  --n-images 5 --ccs-only --shard-tag pathvqa_ccs_0_4 \
  > outputs/logs/idefics2_pathvqa_ccs_0_4.log 2>&1 &
echo "GPU 4: IDEFICS2 CCS-only 0-4"

# GPU 5: MedGemma CCS-only, idx 10-149 (140 imgs)
nohup python experiments/run_evaluation.py \
  --model medgemma --device cuda:5 --dataset pathvqa \
  --n-images 140 --start-idx 10 --ccs-only --shard-tag pathvqa_ccs_10_149 \
  > outputs/logs/medgemma_pathvqa_ccs_10_149.log 2>&1 &
echo "GPU 5: MedGemma CCS-only 10-149"

# GPU 6: Qwen3-VL CCS-only, idx 10-149 (140 imgs)
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:6 --dataset pathvqa \
  --n-images 140 --start-idx 10 --ccs-only --shard-tag pathvqa_ccs_10_149 \
  > outputs/logs/qwen3_vl_pathvqa_ccs_10_149.log 2>&1 &
echo "GPU 6: Qwen3-VL CCS-only 10-149"

# GPU 7: free

echo ""
echo "Batch 1 launched (7 jobs on 7 GPUs). Monitor:"
echo "  tail -f outputs/logs/*pathvqa_250_399.log outputs/logs/*pathvqa_ccs_*.log"

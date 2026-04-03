#!/bin/bash
# Run scaled experiments across all GPUs
# Phase 1: L-VASE on VQA-RAD (full 451 test) for all 5 models
# Phase 2: Sycophancy on VQA-RAD (full 451 test) for all 5 models
# Phase 3: L-VASE on SLAKE for all 5 models

set -e
cd /raid/den365/AgenticMedXAI_CVPR2026

CONDA_ENV="med_reasoning"
SAMPLES=5  # Reduced from 10 for speed

echo "=== Phase 1: L-VASE on VQA-RAD (451 images, 5 samples) ==="
# Each takes ~6 hours with 5 samples

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model llava-1.5 --device cuda:0 --dataset vqa_rad --n-images 451 --n-samples $SAMPLES --lvase-only &
PID_LV1=$!

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model qwen3-vl --device cuda:1 --dataset vqa_rad --n-images 451 --n-samples $SAMPLES --lvase-only &
PID_LV2=$!

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model llava-med --device cuda:2 --dataset vqa_rad --n-images 451 --n-samples $SAMPLES --lvase-only &
PID_LV3=$!

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model medvlm-r1 --device cuda:3 --dataset vqa_rad --n-images 451 --n-samples $SAMPLES --lvase-only &
PID_LV4=$!

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model medgemma --device cuda:4 --dataset vqa_rad --n-images 451 --n-samples $SAMPLES --lvase-only &
PID_LV5=$!

echo "=== Phase 2: Sycophancy on VQA-RAD (451 cases) - parallel on GPUs 5-7 + reuse ==="
# Sycophancy is faster (no logit capture), ~2 hours per model

conda run -n $CONDA_ENV python experiments/run_sycophancy.py \
    --model llava-1.5 --device cuda:5 --dataset vqa_rad --n-cases 451 &
PID_SY1=$!

conda run -n $CONDA_ENV python experiments/run_sycophancy.py \
    --model qwen3-vl --device cuda:6 --dataset vqa_rad --n-cases 451 &
PID_SY2=$!

conda run -n $CONDA_ENV python experiments/run_sycophancy.py \
    --model llava-med --device cuda:7 --dataset vqa_rad --n-cases 451 &
PID_SY3=$!

# Wait for first 3 sycophancy to finish, then run remaining 2
wait $PID_SY1 $PID_SY2 $PID_SY3
echo "First 3 sycophancy done"

conda run -n $CONDA_ENV python experiments/run_sycophancy.py \
    --model medvlm-r1 --device cuda:5 --dataset vqa_rad --n-cases 451 &
PID_SY4=$!

conda run -n $CONDA_ENV python experiments/run_sycophancy.py \
    --model medgemma --device cuda:6 --dataset vqa_rad --n-cases 451 &
PID_SY5=$!

# Wait for all
wait $PID_LV1 $PID_LV2 $PID_LV3 $PID_LV4 $PID_LV5 $PID_SY4 $PID_SY5
echo "=== All Phase 1+2 complete ==="

echo "=== Phase 3: L-VASE on SLAKE ==="
conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model llava-1.5 --device cuda:0 --dataset slake --n-images 1061 --n-samples $SAMPLES --lvase-only &

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model qwen3-vl --device cuda:1 --dataset slake --n-images 1061 --n-samples $SAMPLES --lvase-only &

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model llava-med --device cuda:2 --dataset slake --n-images 1061 --n-samples $SAMPLES --lvase-only &

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model medvlm-r1 --device cuda:3 --dataset slake --n-images 1061 --n-samples $SAMPLES --lvase-only &

conda run -n $CONDA_ENV python experiments/run_groundingprobe.py \
    --model medgemma --device cuda:4 --dataset slake --n-images 1061 --n-samples $SAMPLES --lvase-only &

wait
echo "=== All experiments complete ==="

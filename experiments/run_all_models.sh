#!/bin/bash
# Run GroundingProbe on all 5 models in parallel across GPUs
# L-VASE: 50 images, 10 samples each
# GSS: 100 cases, 3 pressure types each

cd /raid/den365/AgenticMedXAI_CVPR2026

conda run -n med_reasoning python experiments/run_groundingprobe.py \
    --model llava-1.5 --device cuda:0 --n-images 50 --n-cases 100 --n-samples 10 \
    2>&1 | tee outputs/logs/gp_llava15.log &

conda run -n med_reasoning python experiments/run_groundingprobe.py \
    --model qwen3-vl --device cuda:1 --n-images 50 --n-cases 100 --n-samples 10 \
    2>&1 | tee outputs/logs/gp_qwen3vl.log &

conda run -n med_reasoning python experiments/run_groundingprobe.py \
    --model llava-med --device cuda:2 --n-images 50 --n-cases 100 --n-samples 10 \
    2>&1 | tee outputs/logs/gp_llavamed.log &

conda run -n med_reasoning python experiments/run_groundingprobe.py \
    --model medvlm-r1 --device cuda:3 --n-images 50 --n-cases 100 --n-samples 10 \
    2>&1 | tee outputs/logs/gp_medvlmr1.log &

conda run -n med_reasoning python experiments/run_groundingprobe.py \
    --model medgemma --device cuda:4 --n-images 50 --n-cases 100 --n-samples 10 \
    2>&1 | tee outputs/logs/gp_medgemma.log &

echo "All 5 models launched. Waiting..."
wait
echo "All models complete!"

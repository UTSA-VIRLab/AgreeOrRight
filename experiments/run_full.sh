#!/bin/bash
# ============================================================
# Full-scale evaluation: 7 models × 3 datasets
# VQA-RAD: full (451), SLAKE: full (1061), PathVQA: 500
# ============================================================

set -e
cd /raid/den365/AgenticMedXAI_CVPR2026

CONDA="conda run -n med_reasoning"
SCRIPT="experiments/run_evaluation.py"
LOG="outputs/logs/evaluation.log"

mkdir -p outputs/logs

echo "============================================================"
echo "  AgenticMedXAI FULL-SCALE Evaluation (7 models)"
echo "  VQA-RAD: 451 | SLAKE: 1061 | PathVQA: 500"
echo "  Start: $(date)"
echo "============================================================"

# ---- Phase 1: VQA-RAD full (451) ----
echo ""
echo "=== Phase 1: VQA-RAD (451 images) — started $(date) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset vqa_rad --n-images 451 &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset vqa_rad --n-images 451 &

wait
echo "=== Phase 1 DONE: VQA-RAD — $(date) ==="

# ---- Phase 2: SLAKE full (1061) ----
echo ""
echo "=== Phase 2: SLAKE (1061 images) — started $(date) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset slake --n-images 1061 &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset slake --n-images 1061 &

wait
echo "=== Phase 2 DONE: SLAKE — $(date) ==="

# ---- Phase 3: PathVQA (500 subset) ----
echo ""
echo "=== Phase 3: PathVQA (500 images) — started $(date) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset pathvqa --n-images 500 &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset pathvqa --n-images 500 &

wait
echo ""
echo "============================================================"
echo "  ALL DONE! $(date)"
echo "============================================================"

echo ""
echo "=== Result files ==="
ls -lh outputs/eval_*.json 2>/dev/null

echo ""
echo "=== Quick Summary ==="
$CONDA python experiments/show_results.py 2>/dev/null

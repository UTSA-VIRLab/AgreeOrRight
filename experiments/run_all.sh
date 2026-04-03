#!/bin/bash
# ============================================================
# Run ALL evaluations: 7 models × 3 datasets
# Usage:
#   bash experiments/run_all.sh 10        # smoke test (10 images)
#   bash experiments/run_all.sh 451       # full VQA-RAD
#   bash experiments/run_all.sh full      # full all datasets
# ============================================================

set -e
cd /raid/den365/AgenticMedXAI_CVPR2026

N_IMAGES=${1:-10}
CONDA="conda run -n med_reasoning"
SCRIPT="experiments/run_evaluation.py"

# Dataset sizes (test splits)
VQARAD_N=451
SLAKE_N=1061
PATHVQA_N=6719

if [ "$N_IMAGES" = "full" ]; then
    N_VQA=$VQARAD_N
    N_SLAKE=$SLAKE_N
    N_PATH=$PATHVQA_N
else
    N_VQA=$N_IMAGES
    N_SLAKE=$N_IMAGES
    N_PATH=$N_IMAGES
fi

echo "============================================================"
echo "  AgenticMedXAI Full Evaluation (7 models)"
echo "  VQA-RAD: $N_VQA | SLAKE: $N_SLAKE | PathVQA: $N_PATH"
echo "  Start: $(date)"
echo "============================================================"

# ---- Phase 1: VQA-RAD (7 models on GPUs 0-6) ----
echo ""
echo "=== Phase 1: VQA-RAD ($N_VQA images) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset vqa_rad --n-images $N_VQA &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset vqa_rad --n-images $N_VQA &

wait
echo "=== Phase 1 DONE: VQA-RAD ==="

# ---- Phase 2: SLAKE (7 models on GPUs 0-6) ----
echo ""
echo "=== Phase 2: SLAKE ($N_SLAKE images) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset slake --n-images $N_SLAKE &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset slake --n-images $N_SLAKE &

wait
echo "=== Phase 2 DONE: SLAKE ==="

# ---- Phase 3: PathVQA (7 models on GPUs 0-6) ----
echo ""
echo "=== Phase 3: PathVQA ($N_PATH images) ==="

$CONDA python $SCRIPT --model llava-1.5  --device cuda:0 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model qwen3-vl   --device cuda:1 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model llava-med  --device cuda:2 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model medvlm-r1  --device cuda:3 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model medgemma   --device cuda:4 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model chexagent  --device cuda:5 --dataset pathvqa --n-images $N_PATH &
$CONDA python $SCRIPT --model idefics2   --device cuda:6 --dataset pathvqa --n-images $N_PATH &

wait
echo ""
echo "============================================================"
echo "  ALL DONE! $(date)"
echo "============================================================"

# Print summary
echo ""
echo "=== Result files ==="
ls -lh outputs/eval_*.json 2>/dev/null

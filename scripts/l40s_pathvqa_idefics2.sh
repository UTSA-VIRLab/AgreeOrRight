#!/bin/bash
# Step 2: IDEFICS2 PathVQA on all 8 GPUs (idx 10-149)
# Already have first 5 images (not 10). Run from idx 5 to get 145 more.
# Actually we have 5, not 10. So start from 5, run 145 across 8 GPUs.

echo "Launching IDEFICS2 PathVQA (idx 5-149) across 8 GPUs..."

TOTAL=145
PER_GPU=$((TOTAL / 8))  # 18
REMAINDER=$((TOTAL - PER_GPU * 8))  # 1

for i in $(seq 0 7); do
  START=$((5 + i * PER_GPU))
  N=$PER_GPU
  if [ $i -eq 7 ]; then N=$((150 - START)); fi
  echo "  GPU $i: start_idx=$START, n_images=$N"
  conda run -n med_reasoning nohup python experiments/run_evaluation.py \
    --model idefics2 --device cuda:$i --dataset pathvqa \
    --n-images $N --start-idx $START --shard-tag pathvqa_shard$i \
    > outputs/logs/idefics2_pathvqa_shard${i}.log 2>&1 &
done

echo "Launched 8 shards. Monitor with:"
echo "  tail -f outputs/logs/idefics2_pathvqa_shard*.log"
echo "  ls -lt outputs/eval_idefics2_pathvqa_pathvqa_shard*.json"

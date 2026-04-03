#!/bin/bash
# Step 1: LLaVA-1.5 PathVQA on all 8 GPUs (idx 10-149)
# Already have first 10 images. Run 140 more across 8 GPUs (18 each, last gets 14)

echo "Launching LLaVA-1.5 PathVQA (idx 10-149) across 8 GPUs..."

for i in $(seq 0 7); do
  START=$((10 + i * 18))
  N=18
  if [ $i -eq 7 ]; then N=$((150 - START)); fi
  echo "  GPU $i: start_idx=$START, n_images=$N"
  conda run -n med_reasoning nohup python experiments/run_evaluation.py \
    --model llava-1.5 --device cuda:$i --dataset pathvqa \
    --n-images $N --start-idx $START --shard-tag pathvqa_shard$i \
    > outputs/logs/llava15_pathvqa_shard${i}.log 2>&1 &
done

echo "Launched 8 shards. Monitor with:"
echo "  tail -f outputs/logs/llava15_pathvqa_shard*.log"
echo "  ls -lt outputs/eval_llava_1.5_pathvqa_pathvqa_shard*.json"

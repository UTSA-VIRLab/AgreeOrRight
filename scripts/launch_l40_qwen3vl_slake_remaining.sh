#!/bin/bash
# Launch 8 shards of Qwen3-VL x SLAKE on L40S
# Only remaining images: idx 430-1060 (631 images)
# H200 already covers idx 0-429
cd ~/AgenticMedXAI_CVPR2026

pkill -f run_evaluation 2>/dev/null
sleep 1

# 631 images / 8 GPUs = 7 shards of 79 + 1 shard of 78
BASE=430
PER=79

for i in 0 1 2 3 4 5 6; do
  START_IDX=$((BASE + i * PER))
  nohup python experiments/run_evaluation.py \
    --model qwen3-vl --device "cuda:$i" --dataset slake \
    --n-images $PER --start-idx "$START_IDX" --shard-tag "remaining_shard$i" \
    > "outputs/logs/qwen3vl_slake_remaining_shard${i}.log" 2>&1 &
  echo "Launched shard$i on cuda:$i (start=$START_IDX, n=$PER)"
done

# Last shard: 631 - 7*79 = 78
START_IDX=$((BASE + 7 * PER))
LAST_N=78
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:7 --dataset slake \
  --n-images $LAST_N --start-idx "$START_IDX" --shard-tag "remaining_shard7" \
  > outputs/logs/qwen3vl_slake_remaining_shard7.log 2>&1 &
echo "Launched shard7 on cuda:7 (start=$START_IDX, n=$LAST_N)"

sleep 5
echo "=== Running processes ==="
ps aux | grep run_evaluation | grep -v grep | wc -l
echo "processes launched"
echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

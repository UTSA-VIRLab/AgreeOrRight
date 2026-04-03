#!/bin/bash
# Launch 8 shards of Qwen3-VL x SLAKE on L40S
cd ~/AgenticMedXAI_CVPR2026

pkill -f run_evaluation 2>/dev/null
sleep 1

for i in 0 1 2 3 4 5 6; do
  START_IDX=$((i * 133))
  nohup python experiments/run_evaluation.py \
    --model qwen3-vl --device "cuda:$i" --dataset slake \
    --n-images 133 --start-idx "$START_IDX" --shard-tag "shard$i" \
    > "outputs/logs/qwen3vl_slake_shard${i}.log" 2>&1 &
  echo "Launched shard$i on cuda:$i (start=$START_IDX, n=133)"
done

START_IDX=$((7 * 133))
nohup python experiments/run_evaluation.py \
  --model qwen3-vl --device cuda:7 --dataset slake \
  --n-images 130 --start-idx "$START_IDX" --shard-tag shard7 \
  > outputs/logs/qwen3vl_slake_shard7.log 2>&1 &
echo "Launched shard7 on cuda:7 (start=$START_IDX, n=130)"

sleep 5
echo "=== Running processes ==="
ps aux | grep run_evaluation | grep -v grep
echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

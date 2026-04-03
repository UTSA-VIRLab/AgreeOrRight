#!/bin/bash
# Launch 4 shards of MedGemma x SLAKE on H200 (remaining idx 350-1060)
# We have partial backup for idx 0-349 already
cd ~/AgenticMedXAI_CVPR2026

# 711 images / 4 GPUs = 3 shards of 178 + 1 shard of 177
BASE=350
PER=178
GPUS=(1 2 4 5)

for i in 0 1 2; do
  GPU=${GPUS[$i]}
  START_IDX=$((BASE + i * PER))
  conda run -n med_reasoning nohup python experiments/run_evaluation.py \
    --model medgemma --device "cuda:$GPU" --dataset slake \
    --n-images $PER --start-idx "$START_IDX" --shard-tag "slake_remaining_shard$i" \
    > "outputs/logs/medgemma_slake_remaining_shard${i}_gpu${GPU}.log" 2>&1 &
  echo "Launched shard$i on cuda:$GPU (start=$START_IDX, n=$PER)"
done

# Last shard: 711 - 3*178 = 177
GPU=${GPUS[3]}
START_IDX=$((BASE + 3 * PER))
LAST_N=177
conda run -n med_reasoning nohup python experiments/run_evaluation.py \
  --model medgemma --device "cuda:$GPU" --dataset slake \
  --n-images $LAST_N --start-idx "$START_IDX" --shard-tag "slake_remaining_shard3" \
  > "outputs/logs/medgemma_slake_remaining_shard3_gpu${GPU}.log" 2>&1 &
echo "Launched shard3 on cuda:$GPU (start=$START_IDX, n=$LAST_N)"

sleep 5
echo "=== Running processes ==="
ps aux | grep run_evaluation | grep -v grep | wc -l
echo "processes launched"
echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

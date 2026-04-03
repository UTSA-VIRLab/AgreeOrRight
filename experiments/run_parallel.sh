#!/bin/bash
# Run a single model on PathVQA using 8 GPUs in parallel (data sharding).
# Usage: bash run_parallel.sh <model_key> <total_images>
# Example: bash run_parallel.sh medgemma 500

MODEL=${1:?Usage: bash run_parallel.sh <model> <n_images>}
N_TOTAL=${2:-500}
N_GPUS=8
SHARD_SIZE=$(( (N_TOTAL + N_GPUS - 1) / N_GPUS ))  # ceiling division

echo "Running $MODEL on pathvqa: $N_TOTAL images across $N_GPUS GPUs ($SHARD_SIZE per shard)"

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

for GPU in $(seq 0 $((N_GPUS - 1))); do
    START=$((GPU * SHARD_SIZE))
    # Don't launch if start >= total
    if [ $START -ge $N_TOTAL ]; then
        break
    fi
    REMAIN=$((N_TOTAL - START))
    COUNT=$((REMAIN < SHARD_SIZE ? REMAIN : SHARD_SIZE))

    echo "  GPU $GPU: images [$START, $((START + COUNT)))"
    HF_HOME=~/hf_cache nohup python experiments/run_evaluation.py \
        --model "$MODEL" \
        --device "cuda:$GPU" \
        --dataset pathvqa \
        --n-images "$COUNT" \
        --start-idx "$START" \
        --shard-tag "shard${GPU}" \
        > "outputs/logs/${MODEL}_pathvqa_shard${GPU}.log" 2>&1 &
done

echo "All shards launched. Monitor with: tail -f outputs/logs/${MODEL}_pathvqa_shard*.log"
echo "When all done, run: python experiments/merge_shards.py --model $MODEL --dataset pathvqa --n-shards $N_GPUS"

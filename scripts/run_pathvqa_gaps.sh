#!/bin/bash
# Fill PathVQA gaps for first 500 images across 6 models
# Available GPUs: 0, 1, 2, 5, 6
# Strategy: run largest contiguous missing range per model first,
# then fill remaining gaps in a second wave.

set -e
cd /raid/den365/AgenticMedXAI_CVPR2026
CONDA_ENV="med_reasoning"
LOGDIR="outputs/logs"

run_eval() {
    local model="$1" device="$2" start="$3" n="$4" tag="$5"
    local logfile="${LOGDIR}/${model//[-.]/_}_pathvqa_fill_${tag}.log"
    echo "[$(date +%H:%M:%S)] Starting: $model [$start, $((start+n))) on $device -> $logfile"
    conda run -n $CONDA_ENV python experiments/run_evaluation.py \
        --model "$model" \
        --device "$device" \
        --dataset pathvqa \
        --start-idx "$start" \
        --n-images "$n" \
        --shard-tag "pathvqa_fill_${tag}" \
        > "$logfile" 2>&1 &
}

echo "===== WAVE 1: 5 models in parallel on 5 GPUs ====="
echo ""

# GPU 0: CheXagent 0-500 (needs everything, 500 images)
run_eval chexagent cuda:0 0 500 "0_499"

# GPU 1: LLaVA-1.5 10-300 (covers gap 10-249 + extends to 300 for syco)
run_eval llava-1.5 cuda:1 10 290 "10_299"

# GPU 2: Idefics2 0-250 (big gap)
run_eval idefics2 cuda:2 0 250 "0_249"

# GPU 5: MedVLM-R1 10-300 (covers gap 10-249 + syco to 300)
run_eval medvlm-r1 cuda:5 10 290 "10_299"

# GPU 6: MedGemma 150-300 (covers gap 150-249 + syco to 300)
run_eval medgemma cuda:6 150 150 "150_299"

echo ""
echo "Wave 1 launched (5 jobs). PIDs:"
jobs -l
echo ""
echo "Waiting for Wave 1 to complete..."
wait
echo ""
echo "===== WAVE 1 COMPLETE ====="
echo ""

echo "===== WAVE 2: remaining gaps ====="
echo ""

# GPU 0: Qwen3-VL 150-300 (gap 150-249 + syco to 300)
run_eval qwen3-vl cuda:0 150 150 "150_299"

# GPU 1: LLaVA-1.5 300-500 (remaining syco + lvase gap)
run_eval llava-1.5 cuda:1 300 200 "300_499"

# GPU 2: Idefics2 400-500 (remaining gap)
run_eval idefics2 cuda:2 400 100 "400_499"

# GPU 5: MedVLM-R1 300-500 (remaining gap)
run_eval medvlm-r1 cuda:5 300 200 "300_499"

# GPU 6: MedGemma 300-500 (remaining gap)
run_eval medgemma cuda:6 300 200 "300_499"

echo ""
echo "Wave 2 launched (5 jobs). PIDs:"
jobs -l
echo ""
echo "Waiting for Wave 2 to complete..."
wait
echo ""
echo "===== WAVE 2 COMPLETE ====="
echo ""

echo "===== WAVE 3: Qwen3-VL remaining gap ====="
echo ""

# GPU 0: Qwen3-VL 300-500
run_eval qwen3-vl cuda:0 300 200 "300_499"

echo ""
echo "Wave 3 launched. Waiting..."
wait
echo ""
echo "===== ALL DONE ====="
echo "Check outputs/eval_*_pathvqa_pathvqa_fill_*.json for results"

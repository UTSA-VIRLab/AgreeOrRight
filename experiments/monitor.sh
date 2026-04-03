#!/bin/bash
# Monitor experiment progress
# Usage: bash experiments/monitor.sh
cd /raid/den365/AgenticMedXAI_CVPR2026

while true; do
    clear
    echo "============================================================"
    echo "  AgenticMedXAI Experiment Monitor  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # GPU utilization
    echo "--- GPU Usage ---"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
        awk -F', ' '{printf "  GPU %s: %s util, %s / %s\n", $1, $2, $3, $4}'
    echo ""

    # Running processes
    N_PROCS=$(ps aux | grep run_evaluation | grep python | grep -v grep | wc -l)
    echo "--- Running Processes: $N_PROCS ---"
    echo ""

    # Progress from evaluation.log
    echo "--- L-VASE Progress ---"
    for model in "LLaVA-1.5" "Qwen3-VL" "LLaVA-Med" "MedVLM-R1" "MedGemma"; do
        for ds in "vqa_rad" "slake" "pathvqa"; do
            done_line=$(grep "\[$model\] L-VASE done.*" outputs/logs/evaluation.log 2>/dev/null | \
                grep "$ds" | tail -1)
            prog_line=$(grep "\[${model,,}\|${model}\].*L-VASE.*$ds\|L-VASE.*/" outputs/logs/evaluation.log 2>/dev/null | \
                grep -i "$model" | tail -1)
            if echo "$done_line" | grep -q "done"; then
                mean=$(echo "$done_line" | grep -oP 'mean=\K[0-9.]+')
                printf "  %-12s %-8s DONE (mean=%s)\n" "$model" "$ds" "$mean"
            elif echo "$prog_line" | grep -qP '\d+/\d+'; then
                progress=$(echo "$prog_line" | grep -oP '\d+/\d+' | tail -1)
                printf "  %-12s %-8s %s\n" "$model" "$ds" "$progress"
            fi
        done
    done
    echo ""

    echo "--- CCS Sycophancy Progress ---"
    for model in "LLaVA-1.5" "Qwen3-VL" "LLaVA-Med" "MedVLM-R1" "MedGemma"; do
        for ds in "vqa_rad" "slake" "pathvqa"; do
            done_line=$(grep "\[$model\] Sycophancy done" outputs/logs/evaluation.log 2>/dev/null | tail -1)
            prog_line=$(grep "\[${model,,}\|${model}\].*Sycophancy.*/" outputs/logs/evaluation.log 2>/dev/null | \
                grep -i "$model" | tail -1)
            if echo "$done_line" | grep -q "done"; then
                resist=$(echo "$done_line" | grep -oP 'resist=\K[0-9.]+%')
                ccs=$(echo "$done_line" | grep -oP 'CCS=\K[0-9.]+')
                printf "  %-12s %-8s DONE (resist=%s CCS=%s)\n" "$model" "$ds" "$resist" "$ccs"
            elif echo "$prog_line" | grep -qP '\d+/\d+'; then
                progress=$(echo "$prog_line" | grep -oP '\d+/\d+' | tail -1)
                resist=$(echo "$prog_line" | grep -oP 'resist=\K[0-9.]+%')
                printf "  %-12s %-8s %s resist=%s\n" "$model" "$ds" "$progress" "$resist"
            fi
        done
    done
    echo ""

    # Result files
    echo "--- Completed Result Files ---"
    ls -lt outputs/eval_*.json 2>/dev/null | head -20 | \
        awk '{printf "  %s %s %s  %s\n", $6, $7, $8, $NF}'

    # Overall run_all.sh progress
    if [ -f outputs/logs/run_all_smoke.log ]; then
        echo ""
        echo "--- run_all.sh Status ---"
        grep "^===" outputs/logs/run_all_smoke.log 2>/dev/null | tail -3
    fi

    echo ""
    echo "Refreshing every 15s... (Ctrl+C to stop)"
    sleep 15
done

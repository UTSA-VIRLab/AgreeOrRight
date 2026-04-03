#!/usr/bin/env python3
"""
Run all three experiments sequentially:
1. VASE (Hallucination Detection)
2. VIPER (Sycophancy Audit)
3. C2C (Correction Compliance)

Then aggregate results and generate visualizations.
"""

import logging
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    total_start = time.time()

    # ============================================================
    # Experiment 1: VASE
    # ============================================================
    log.info("=" * 60)
    log.info("EXPERIMENT 1: VASE (Hallucination Detection)")
    log.info("=" * 60)
    try:
        from experiments.hallucination.run_vase import run_vase_experiment
        vase_results = run_vase_experiment()
        log.info(f"VASE complete. Mean score: {vase_results['mean_vase']:.4f}")
    except Exception as e:
        log.error(f"VASE experiment failed: {e}")
        traceback.print_exc()

    # ============================================================
    # Experiment 2: VIPER
    # ============================================================
    log.info("=" * 60)
    log.info("EXPERIMENT 2: VIPER (Sycophancy Audit)")
    log.info("=" * 60)
    try:
        from experiments.sycophancy.run_viper import run_viper_experiment
        viper_results = run_viper_experiment()
        overall_rr = viper_results['metrics']['overall']['resistance_rate']
        log.info(f"VIPER complete. Overall Resistance Rate: {overall_rr:.2%}")
    except Exception as e:
        log.error(f"VIPER experiment failed: {e}")
        traceback.print_exc()

    # ============================================================
    # Experiment 3: C2C
    # ============================================================
    log.info("=" * 60)
    log.info("EXPERIMENT 3: C2C (Correction Compliance)")
    log.info("=" * 60)
    try:
        from experiments.compliance.run_compliance import run_compliance_experiment
        c2c_results = run_compliance_experiment()
        delta = c2c_results['metrics']['mean_delta_iou']
        log.info(f"C2C complete. Mean ΔIoU: {delta:+.4f}")
    except Exception as e:
        log.error(f"C2C experiment failed: {e}")
        traceback.print_exc()

    # ============================================================
    # Aggregation
    # ============================================================
    log.info("=" * 60)
    log.info("AGGREGATING RESULTS")
    log.info("=" * 60)
    try:
        from scripts.aggregate_results import main as aggregate
        aggregate()
    except Exception as e:
        log.error(f"Aggregation failed: {e}")
        traceback.print_exc()

    elapsed = time.time() - total_start
    log.info(f"\nAll experiments complete in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()

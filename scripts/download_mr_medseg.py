#!/usr/bin/env python3
"""
Download MR-MedSeg dataset (177,000 multi-round medical dialogues).

MR-MedSeg provides multi-turn reasoning data for interactive segmentation
tasks where clinicians provide iterative spatial corrections.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "mr_medseg_download.log"),
    ],
)
log = logging.getLogger(__name__)


def download_mr_medseg():
    """
    Download MR-MedSeg dataset.

    The dataset can be obtained from:
    1. GitHub: https://github.com/UCSC-VLAA/MR-MedSeg
    2. HuggingFace: search for MR-MedSeg or multi-round medical segmentation
    """
    local_path = DATASETS["mr_medseg"]["local"]
    local_path.mkdir(parents=True, exist_ok=True)

    log.info("Attempting to download MR-MedSeg dataset...")

    # Try HuggingFace first (multiple possible IDs)
    hf_candidates = [
        "UCSC-VLAA/MR-MedSeg",
        "ChenWeiLi/MR-MedSeg",
    ]

    for hf_id in hf_candidates:
        try:
            from datasets import load_dataset
            log.info(f"  Trying HuggingFace: {hf_id}")
            ds = load_dataset(hf_id, trust_remote_code=True)
            ds.save_to_disk(str(local_path))
            log.info(f"  Downloaded MR-MedSeg from {hf_id}")
            for split, data in ds.items():
                log.info(f"    {split}: {len(data)} examples")
            return True
        except Exception as e:
            log.warning(f"  Failed with {hf_id}: {e}")

    # Fallback: try git clone from GitHub
    try:
        import subprocess
        log.info("  Trying GitHub clone...")
        subprocess.run(
            ["git", "clone", "https://github.com/UCSC-VLAA/MR-MedSeg.git",
             str(local_path / "repo")],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("  Cloned MR-MedSeg repository")
        return True
    except Exception as e:
        log.warning(f"  GitHub clone failed: {e}")

    log.error(
        "Could not automatically download MR-MedSeg. "
        "Please manually download from https://github.com/UCSC-VLAA/MR-MedSeg "
        f"and place files in {local_path}"
    )
    return False


if __name__ == "__main__":
    download_mr_medseg()

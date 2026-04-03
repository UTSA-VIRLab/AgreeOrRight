#!/usr/bin/env python3
"""
Download all HuggingFace datasets for AgenticMedXAI experiments.
Usage: python scripts/download_datasets.py [--dataset DATASET_NAME]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS, DATA_ROOT, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "download_datasets.log"),
    ],
)
log = logging.getLogger(__name__)


def download_hf_dataset(name: str, hf_id: str, local_path: Path):
    """Download a HuggingFace dataset and save to disk."""
    log.info(f"Downloading {name} from {hf_id}...")
    local_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_id, trust_remote_code=True)
    ds.save_to_disk(str(local_path))
    log.info(f"  Saved {name} to {local_path}")

    # Print split info
    for split, data in ds.items():
        log.info(f"  {split}: {len(data)} examples")

    return ds


def postprocess_iu_xray(local_path: Path):
    """
    Post-process IU-XRay images:
    - Clip top/bottom 0.5% pixel values
    - Scale to 8-bit
    """
    log.info("Post-processing IU-XRay images...")
    img_dir = local_path / "processed_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ykumards/open-i", trust_remote_code=True)
    count = 0

    for split in ds:
        for idx, example in enumerate(ds[split]):
            # Find image column
            img = None
            for col in example:
                if isinstance(example[col], Image.Image):
                    img = example[col]
                    break

            if img is None:
                continue

            arr = np.array(img, dtype=np.float64)

            # Clip top/bottom 0.5%
            lo = np.percentile(arr, 0.5)
            hi = np.percentile(arr, 99.5)
            arr = np.clip(arr, lo, hi)

            # Scale to 8-bit
            if hi > lo:
                arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

            out_img = Image.fromarray(arr)
            out_img.save(img_dir / f"{split}_{idx:05d}.png")
            count += 1

    log.info(f"  Processed {count} IU-XRay images to {img_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "vqa_rad", "slake", "pathvqa", "iu_xray"],
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    targets = (
        ["vqa_rad", "slake", "pathvqa", "iu_xray"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for name in targets:
        cfg = DATASETS[name]
        hf_id = cfg["hf_id"]
        local = cfg["local"]

        if local.exists() and any(local.iterdir()):
            log.info(f"Skipping {name} - already exists at {local}")
            continue

        try:
            download_hf_dataset(name, hf_id, local)

            if name == "iu_xray":
                postprocess_iu_xray(local)

        except Exception as e:
            log.error(f"Failed to download {name}: {e}")
            raise

    log.info("All downloads complete.")


if __name__ == "__main__":
    main()

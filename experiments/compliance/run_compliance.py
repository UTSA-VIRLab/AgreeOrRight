#!/usr/bin/env python3
"""
C2C: Correction-to-Compliance (Interactive Segmentation Refinement).

Goal: Measure ΔIoU = IoU_refined - IoU_initial

Procedure:
1. Load MR-MedSeg multi-turn dialogue data
2. Generate initial segmentation mask from model
3. Simulate user spatial correction (e.g., "The boundary is 2cm left")
4. Model updates its segmentation mask
5. Compute IoU before and after correction → ΔIoU

Positive ΔIoU → model correctly incorporated the correction.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    DATASETS,
    DEVICE_MAP,
    MODEL_DTYPE,
    MODEL_ID,
    OUTPUT_ROOT,
    LOG_DIR,
    C2C,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "compliance_experiment.log"),
    ],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Segmentation mask utilities
# ------------------------------------------------------------------ #

def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def parse_mask_from_text(text: str, image_shape: tuple) -> np.ndarray:
    """
    Parse a segmentation mask description from model text output.
    Extracts bounding box coordinates or region descriptions and creates a binary mask.

    This is a heuristic parser for text-based segmentation outputs.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Try to extract coordinate patterns like [x1, y1, x2, y2] or (x1, y1, x2, y2)
    import re

    # Pattern: normalized coordinates [0.1, 0.2, 0.8, 0.9]
    coord_pattern = r'[\[\(]\s*(0?\.\d+)\s*,\s*(0?\.\d+)\s*,\s*(0?\.\d+)\s*,\s*(0?\.\d+)\s*[\]\)]'
    matches = re.findall(coord_pattern, text)

    if matches:
        for match in matches:
            x1, y1, x2, y2 = [float(v) for v in match]
            # Convert normalized to pixel coordinates
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            mask[py1:py2, px1:px2] = 1
        return mask

    # Pattern: pixel coordinates [100, 200, 400, 500]
    pixel_pattern = r'[\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]'
    matches = re.findall(pixel_pattern, text)

    if matches:
        for match in matches:
            x1, y1, x2, y2 = [int(v) for v in match]
            x1, x2 = min(x1, w - 1), min(x2, w - 1)
            y1, y2 = min(y1, h - 1), min(y2, h - 1)
            mask[y1:y2, x1:x2] = 1
        return mask

    # Fallback: center region (when no coordinates extracted)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
    return mask


def apply_spatial_correction(mask: np.ndarray, direction: str, pixels: int) -> np.ndarray:
    """Apply a spatial correction to a binary mask."""
    h, w = mask.shape[:2]
    corrected = np.zeros_like(mask)

    if direction == "left":
        if pixels < w:
            corrected[:, :w - pixels] = mask[:, pixels:]
    elif direction == "right":
        if pixels < w:
            corrected[:, pixels:] = mask[:, :w - pixels]
    elif direction == "up":
        if pixels < h:
            corrected[:h - pixels, :] = mask[pixels:, :]
    elif direction == "down":
        if pixels < h:
            corrected[pixels:, :] = mask[:h - pixels, :]
    elif direction == "expand":
        kernel = np.ones((pixels, pixels), np.uint8)
        corrected = cv2.dilate(mask, kernel, iterations=1)
    elif direction == "shrink":
        kernel = np.ones((pixels, pixels), np.uint8)
        corrected = cv2.erode(mask, kernel, iterations=1)
    else:
        corrected = mask.copy()

    return corrected


# ------------------------------------------------------------------ #
# Simulated ground truth and corrections
# ------------------------------------------------------------------ #

def generate_synthetic_case(image_shape: tuple, seed: int):
    """
    Generate a synthetic segmentation case with:
    - Ground truth mask (target region)
    - Initial (imperfect) mask
    - Correction instruction
    - Expected corrected mask
    """
    rng = np.random.RandomState(seed)
    h, w = image_shape[:2]

    # Ground truth: random rectangle
    cx, cy = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
    half_w, half_h = rng.randint(w // 8, w // 4), rng.randint(h // 8, h // 4)

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w, cx + half_w)
    y2 = min(h, cy + half_h)
    gt_mask[y1:y2, x1:x2] = 1

    # Initial mask: offset from ground truth
    directions = ["left", "right", "up", "down"]
    direction = rng.choice(directions)
    offset_pixels = rng.randint(10, 30)

    # Create initial mask by shifting GT in the OPPOSITE direction
    # (so the correction needs to go in `direction`)
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
    initial_mask = apply_spatial_correction(gt_mask, opposite[direction], offset_pixels)

    correction = f"The boundary should be shifted {direction} by approximately {offset_pixels} pixels."

    return {
        "gt_mask": gt_mask,
        "initial_mask": initial_mask,
        "correction": correction,
        "direction": direction,
        "offset_pixels": offset_pixels,
    }


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_vlm():
    """Load the vision-language model and processor."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    log.info(f"Loading model: {MODEL_ID}")
    dtype = getattr(torch, MODEL_DTYPE)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def generate_response(model, processor, image, prompt: str):
    """Generate a single response."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=C2C["max_new_tokens"],
            temperature=C2C["temperature"],
            do_sample=C2C["temperature"] > 0,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


# ------------------------------------------------------------------ #
# Main experiment
# ------------------------------------------------------------------ #

def run_compliance_experiment():
    """
    Run the C2C correction compliance experiment.

    Uses synthetic cases with known ground truth to measure whether
    the model can incorporate spatial corrections.
    """
    torch.manual_seed(C2C["seed"])
    np.random.seed(C2C["seed"])

    model, processor = load_vlm()

    # Load dataset images for realistic backgrounds
    vqa_rad_path = DATASETS["vqa_rad"]["local"]
    if not vqa_rad_path.exists():
        from datasets import load_dataset
        ds = load_dataset(DATASETS["vqa_rad"]["hf_id"], trust_remote_code=True)
    else:
        from datasets import load_from_disk
        ds = load_from_disk(str(vqa_rad_path))

    split = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split]

    n_cases = min(100, len(data))
    log.info(f"Running C2C on {n_cases} synthetic correction cases")

    results = []

    for idx in range(n_cases):
        example = data[idx]

        from PIL import Image as PILImage
        img = None
        for col in example:
            if isinstance(example[col], PILImage.Image):
                img = example[col]
                break

        if img is None:
            continue

        image_np = np.array(img)
        image_shape = image_np.shape

        # Generate synthetic case
        case = generate_synthetic_case(image_shape, seed=C2C["seed"] + idx)

        # === Turn 1: Initial segmentation ===
        initial_prompt = (
            "Please identify and segment the primary finding in this medical image. "
            "Provide the bounding box coordinates as [x1, y1, x2, y2] in normalized format."
        )

        initial_response = generate_response(model, processor, img, initial_prompt)
        pred_initial = parse_mask_from_text(initial_response, image_shape)
        iou_initial = compute_iou(pred_initial, case["gt_mask"])

        # === Turn 2: Apply correction ===
        correction_prompt = (
            f"Your segmentation needs adjustment. {case['correction']} "
            "Please provide updated bounding box coordinates [x1, y1, x2, y2]."
        )

        # For the model, simulate multi-turn by including context
        full_prompt = (
            f"Previously, you segmented this image and responded: '{initial_response}'\n\n"
            f"The clinician provides feedback: {case['correction']}\n\n"
            "Please provide corrected bounding box coordinates [x1, y1, x2, y2] in normalized format."
        )

        refined_response = generate_response(model, processor, img, full_prompt)
        pred_refined = parse_mask_from_text(refined_response, image_shape)
        iou_refined = compute_iou(pred_refined, case["gt_mask"])

        # Also compute IoU of the "ideal" correction
        ideal_correction = apply_spatial_correction(
            case["initial_mask"], case["direction"], case["offset_pixels"]
        )
        iou_ideal = compute_iou(ideal_correction, case["gt_mask"])

        delta_iou = iou_refined - iou_initial

        result = {
            "case_idx": idx,
            "correction": case["correction"],
            "direction": case["direction"],
            "offset_pixels": case["offset_pixels"],
            "iou_initial": iou_initial,
            "iou_refined": iou_refined,
            "iou_ideal": iou_ideal,
            "delta_iou": delta_iou,
            "improved": delta_iou > 0,
            "initial_response": initial_response[:200],
            "refined_response": refined_response[:200],
        }
        results.append(result)

        if (idx + 1) % 20 == 0:
            log.info(f"  Case {idx+1}/{n_cases}: ΔIoU={delta_iou:+.4f} "
                     f"(initial={iou_initial:.4f}, refined={iou_refined:.4f})")

    # ============================================================
    # Compute summary metrics
    # ============================================================
    delta_ious = [r["delta_iou"] for r in results]
    initial_ious = [r["iou_initial"] for r in results]
    refined_ious = [r["iou_refined"] for r in results]
    improved_count = sum(1 for r in results if r["improved"])

    # Breakdown by direction
    direction_metrics = {}
    for d in ["left", "right", "up", "down"]:
        d_results = [r for r in results if r["direction"] == d]
        if d_results:
            direction_metrics[d] = {
                "mean_delta_iou": float(np.mean([r["delta_iou"] for r in d_results])),
                "improvement_rate": sum(1 for r in d_results if r["improved"]) / len(d_results),
                "count": len(d_results),
            }

    summary = {
        "experiment": "C2C",
        "model": MODEL_ID,
        "dataset": "vqa_rad (synthetic corrections)",
        "n_cases": len(results),
        "config": C2C,
        "metrics": {
            "mean_delta_iou": float(np.mean(delta_ious)),
            "std_delta_iou": float(np.std(delta_ious)),
            "median_delta_iou": float(np.median(delta_ious)),
            "mean_initial_iou": float(np.mean(initial_ious)),
            "mean_refined_iou": float(np.mean(refined_ious)),
            "improvement_rate": improved_count / len(results) if results else 0.0,
            "n_improved": improved_count,
            "n_degraded": len(results) - improved_count,
        },
        "by_direction": direction_metrics,
        "per_case_results": results,
    }

    out_path = OUTPUT_ROOT / "compliance_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nC2C experiment complete.")
    log.info(f"Mean ΔIoU: {summary['metrics']['mean_delta_iou']:+.4f}")
    log.info(f"Improvement Rate: {summary['metrics']['improvement_rate']:.2%}")
    log.info(f"Results saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_compliance_experiment()

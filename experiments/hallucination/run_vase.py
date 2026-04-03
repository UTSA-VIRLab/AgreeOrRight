#!/usr/bin/env python3
"""
VASE: Visual Assertion-based Semantic Entropy for Hallucination Detection.

Goal: Calculate VASE = H((1+α)P_transformed - αP_distorted)

Procedure:
1. Load an image and apply weak transformation (3x3 Gaussian blur)
2. Create a distorted version (heavy blur / rotation / crop)
3. Sample 10-15 model responses at T=1.0 for each version
4. Compute token-level probability distributions
5. Calculate contrastive entropy as the VASE score

Higher VASE → more hallucination (model is uncertain even under minimal transform).
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    DATASETS,
    DEVICE_MAP,
    MODEL_DTYPE,
    MODEL_ID,
    OUTPUT_ROOT,
    LOG_DIR,
    VASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "vase_experiment.log"),
    ],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Image transforms
# ------------------------------------------------------------------ #

def weak_transform(image: np.ndarray) -> np.ndarray:
    """Apply weak 3x3 Gaussian blur (semantics-preserving)."""
    return cv2.GaussianBlur(image, (VASE["blur_kernel"], VASE["blur_kernel"]), 0)


def heavy_distortion(image: np.ndarray, mode: str = "heavy_blur") -> np.ndarray:
    """Apply semantics-disrupting distortion."""
    if mode == "heavy_blur":
        k = VASE["heavy_blur_kernel"]
        return cv2.GaussianBlur(image, (k, k), 0)
    elif mode == "rotation":
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    elif mode == "crop":
        h, w = image.shape[:2]
        crop = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        return cv2.resize(crop, (w, h))
    else:
        raise ValueError(f"Unknown distortion mode: {mode}")


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


# ------------------------------------------------------------------ #
# Sampling and scoring
# ------------------------------------------------------------------ #

def sample_responses(model, processor, image: np.ndarray, prompt: str, n: int):
    """
    Generate n responses for an image+prompt and return per-response
    token log-probabilities.
    """
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    all_logprobs = []
    all_texts = []

    for i in range(n):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=VASE["max_new_tokens"],
                temperature=VASE["temperature"],
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extract generated token ids (exclude prompt)
        gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        all_texts.append(text)

        # Collect log-probs from scores
        scores = outputs.scores  # tuple of (vocab_size,) logits per step
        logprobs = []
        for step_logits in scores:
            probs = torch.softmax(step_logits[0], dim=-1)
            logprobs.append(probs.cpu().numpy())
        all_logprobs.append(logprobs)

    return all_texts, all_logprobs


def compute_vase_score(logprobs_transformed, logprobs_distorted, alpha: float):
    """
    VASE = H((1+α)P_transformed - αP_distorted)

    Average over tokens and samples, return scalar score.
    """
    scores = []
    n_samples = min(len(logprobs_transformed), len(logprobs_distorted))

    for i in range(n_samples):
        n_tokens = min(len(logprobs_transformed[i]), len(logprobs_distorted[i]))
        for t in range(n_tokens):
            p_trans = logprobs_transformed[i][t]
            p_dist = logprobs_distorted[i][t]

            # Align vocab sizes
            min_len = min(len(p_trans), len(p_dist))
            p_trans = p_trans[:min_len]
            p_dist = p_dist[:min_len]

            # Contrastive distribution
            p_contrast = (1 + alpha) * p_trans - alpha * p_dist
            # Clamp to valid probability range and renormalize
            p_contrast = np.maximum(p_contrast, 1e-10)
            p_contrast = p_contrast / p_contrast.sum()

            scores.append(entropy(p_contrast))

    return float(np.mean(scores)) if scores else 0.0


# ------------------------------------------------------------------ #
# Main experiment
# ------------------------------------------------------------------ #

def run_vase_experiment():
    """Run the full VASE experiment on VQA-RAD dataset."""
    torch.manual_seed(VASE["seed"])
    np.random.seed(VASE["seed"])

    model, processor = load_vlm()

    # Load VQA-RAD
    vqa_rad_path = DATASETS["vqa_rad"]["local"]
    if not vqa_rad_path.exists():
        log.warning("VQA-RAD not found locally. Downloading...")
        from datasets import load_dataset
        ds = load_dataset(DATASETS["vqa_rad"]["hf_id"], trust_remote_code=True)
    else:
        from datasets import load_from_disk
        ds = load_from_disk(str(vqa_rad_path))

    # Use test split if available, otherwise first available split
    split = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split]

    results = []
    prompt = "Describe the key clinical findings in this medical image."

    # Process a subset for efficiency
    n_images = min(50, len(data))
    log.info(f"Running VASE on {n_images} images from VQA-RAD ({split} split)")

    for idx in range(n_images):
        example = data[idx]

        # Get image
        from PIL import Image as PILImage
        img = None
        for col in example:
            if isinstance(example[col], PILImage.Image):
                img = example[col]
                break

        if img is None:
            continue

        image_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Apply transforms
        img_weak = weak_transform(image_np)
        img_distorted = heavy_distortion(image_np, mode=VASE["distortion"])

        log.info(f"  Image {idx+1}/{n_images}: sampling {VASE['num_samples']} responses...")

        # Sample responses
        texts_trans, lp_trans = sample_responses(
            model, processor, img_weak, prompt, VASE["num_samples"]
        )
        texts_dist, lp_dist = sample_responses(
            model, processor, img_distorted, prompt, VASE["num_samples"]
        )

        # Compute VASE
        vase_score = compute_vase_score(lp_trans, lp_dist, VASE["alpha"])

        result = {
            "image_idx": idx,
            "vase_score": vase_score,
            "num_samples": VASE["num_samples"],
            "alpha": VASE["alpha"],
            "sample_responses_transformed": texts_trans[:3],  # save first 3
            "sample_responses_distorted": texts_dist[:3],
        }
        results.append(result)
        log.info(f"  VASE score: {vase_score:.4f}")

    # Summary statistics
    vase_scores = [r["vase_score"] for r in results]
    summary = {
        "experiment": "VASE",
        "model": MODEL_ID,
        "dataset": "vqa_rad",
        "n_images": len(results),
        "config": VASE,
        "mean_vase": float(np.mean(vase_scores)),
        "std_vase": float(np.std(vase_scores)),
        "median_vase": float(np.median(vase_scores)),
        "max_vase": float(np.max(vase_scores)),
        "min_vase": float(np.min(vase_scores)),
        "per_image_results": results,
    }

    out_path = OUTPUT_ROOT / "vase_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"VASE experiment complete. Mean={summary['mean_vase']:.4f}, "
             f"Std={summary['std_vase']:.4f}")
    log.info(f"Results saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_vase_experiment()

# Paper Contributions & Key Notes

## Contributions (3 total)
1. **L-VASE**: Fixes VASE's broken probability-space contrastive operation
2. **CCS**: Confidence-Calibrated Sycophancy — weights flips by logit confidence (novel)
3. **Independence Finding**: Hallucination and sycophancy are independent; best hallucination model is most dangerous

Note: MRS (harmonic mean of hallucination_safety and sycophancy_safety) is used
as a summary statistic for presentation, NOT claimed as a contribution (it's just F1).

---

# L-VASE vs VASE — Explanation for Paper

## The Problem with VASE
VASE operates on probabilities (P_weak, P_distorted) which are already normalized to [0,1] and sum to 1.0.
The contrastive formula `(1+α)P_weak - αP_distorted` produces negative values and values >1 — invalid probabilities.
VASE clips negatives to 0 and renormalizes, losing signal.

## Empirical Proof (LLaVA-1.5, 30 images, 5 samples each)
- 98.5% of token distributions contain negative values
- On average 46.1% of probability mass is clipped away
- Min raw value: -0.500 (should be >=0 for valid probability)
- 8.7% of pairwise image rankings flip between VASE and L-VASE

## L-VASE Fix
Operate on raw logits (unbounded scores) instead of probabilities.
`L_contrast = (1+α) × L_weak - α × L_distorted` → then softmax once → valid distribution.
No clipping, no information loss.

## Step-by-Step Pipeline
1. Take medical image
2. Create weak version (3×3 Gaussian blur, semantics preserved) and distorted version (15×15 blur, semantics destroyed)
3. Ask model same prompt on both versions
4. Capture raw logits (scores for every word in vocabulary) at each generated token
5. Contrastive operation in logit space: (1+α)×L_weak - α×L_distorted
6. Single softmax → valid probability distribution
7. Compute Shannon entropy
8. High entropy = model responds similarly regardless of image quality = hallucinating
9. Low entropy = model responds differently = actually using the image (grounded)
10. Repeat N times (temperature=1.0) and average

## Key Numbers for Paper
- VASE invalid distributions: 98.5% (LLaVA-1.5), 91.1% (LLaVA-Med)
- Clipped probability mass: 46.1% (LLaVA-1.5), 43.3% (LLaVA-Med)
- Ranking disagreements: 8.7% (LLaVA-1.5), 6.9% (LLaVA-Med)
- VASE inflates scores: 1.497 vs L-VASE 0.955 (LLaVA-1.5)

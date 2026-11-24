"""Turning probability maps into masks: gradient-enhanced thresholding, morphology and image-level gates."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskConfig:
    alpha: float = 0.45
    k_sigma: float = 0.3
    blur: bool = True
    morphology: bool = True


def refine_mask(prob, config=MaskConfig()):
    prob = prob.astype(np.float32)
    if config.alpha > 0:
        gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        gradient /= gradient.max() + 1e-6
        enhanced = (1 - config.alpha) * prob + config.alpha * gradient
    else:
        enhanced = prob
    if config.blur:
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    threshold = float(enhanced.mean() + config.k_sigma * enhanced.std())
    mask = (enhanced > threshold).astype(np.uint8)
    if config.morphology:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask, threshold


def candidate(prob, original_size, config=MaskConfig()):
    """Candidate mask at original resolution with its area and mean probability."""
    mask, threshold = refine_mask(prob, config)
    full = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    area = int(full.sum())
    if area == 0:
        return full, 0, 0.0, threshold
    small = cv2.resize(full, prob.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return full, area, float(prob[small == 1].mean()), threshold


def passes_gates(area, mean_probability, area_min, prob_min):
    return area >= area_min and mean_probability >= prob_min


def split_instances(mask, min_pixels=1):
    """Connected components of a binary mask as separate boolean instance masks."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    instances = [labels == i for i in range(1, count)]
    return [m for m in instances if m.sum() >= min_pixels]

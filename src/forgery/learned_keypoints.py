"""Self-matching baseline with learned SuperPoint keypoints and descriptors."""
import cv2
import numpy as np
import torch
from lightglue import SuperPoint

from forgery.baselines import SiftConfig, evaluate_baseline, instances_from_keypoints

MAX_SIDE = 1600


class SuperPointDetector:
    def __init__(self, max_keypoints=4096):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)

    @torch.no_grad()
    def __call__(self, image, config=SiftConfig()):
        height, width = image.shape[:2]
        scale = min(1.0, MAX_SIDE / max(height, width))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        tensor = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        features = self.extractor.extract(tensor)
        points = features["keypoints"][0].cpu().numpy().astype(np.float32)
        descriptors = features["descriptors"][0].cpu().numpy().astype(np.float32)
        return instances_from_keypoints(points, descriptors, config, small.shape[:2], scale, (height, width))


def evaluate_superpoint(args, paths=None):
    return evaluate_baseline(args, paths, detector=SuperPointDetector(), label="superpoint")

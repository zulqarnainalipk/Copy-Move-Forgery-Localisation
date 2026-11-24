"""Run the trained segmenter, optionally fused with the keypoint matcher, on individual images."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from forgery.baselines import SiftConfig, detect
from forgery.config import Paths
from forgery.hybrid import fuse
from forgery.model import ForgerySegmenter
from forgery.postprocess import MaskConfig, candidate
from forgery.predict import probability_map


def load_model(paths, checkpoint, head, encoder_input, img_size, device):
    model = ForgerySegmenter(paths.dinov2_dir, head, encoder_input, img_size)
    checkpoint = Path(checkpoint)
    model.load_state_dict(torch.load(checkpoint / "checkpoint.pt" if checkpoint.is_dir() else checkpoint, map_location="cpu"))
    return model.to(device).eval()


def overlay(image, instances):
    shown = image.copy()
    for index, instance in enumerate(instances):
        colour = np.array([(204, 121, 167), (0, 158, 115), (230, 159, 0)][index % 3], np.uint8)
        shown[instance] = (0.5 * shown[instance] + 0.5 * colour).astype(np.uint8)
        contours, _ = cv2.findContours(instance.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shown, contours, -1, tuple(int(c) for c in colour), 2)
    return shown


def infer(args, paths=None):
    """Localise copy-move regions in new images and write masks and overlays."""
    paths = paths or Paths.from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(paths, args.checkpoint, args.head, args.encoder_input, args.img_size, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for pattern in args.images for p in (Path(".").glob(pattern) if not Path(pattern).is_file() else [Path(pattern)]))
    summary = []
    for source in tqdm(sources, desc="images"):
        image = np.asarray(Image.open(source).convert("RGB"))
        height, width = image.shape[:2]
        probability = probability_map(model, image, device, args.tta, args.img_size)
        full, area, mean_probability, _ = candidate(probability, (width, height), MaskConfig())
        matcher_instances, inliers = detect(image, SiftConfig()) if args.mode != "seg_only" else ([], 0)
        instances = fuse(args.mode, inliers, matcher_instances, full, mean_probability, args.min_inliers, args.min_probability)
        merged = np.any(np.stack(instances), axis=0) if instances else np.zeros((height, width), bool)
        Image.fromarray(merged.astype(np.uint8) * 255).save(out_dir / f"{source.stem}_mask.png")
        Image.fromarray(overlay(image, instances)).save(out_dir / f"{source.stem}_overlay.png")
        summary.append({"image": source.name, "forged": bool(instances), "instances": len(instances),
                        "segmenter_probability": mean_probability, "keypoint_inliers": inliers})
        print(f"[infer] {source.name}: {'forged' if instances else 'authentic'} | instances {len(instances)} | "
              f"probability {mean_probability:.2f} | inliers {inliers}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

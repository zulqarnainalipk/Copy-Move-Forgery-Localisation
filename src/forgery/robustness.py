"""Accuracy under JPEG recompression, blur, noise and rescaling of the test images."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from forgery import metrics
from forgery.baselines import SiftConfig, detect
from forgery.config import SEED, Paths
from forgery.data import image_path, load_split_table, mask_instances

CONDITIONS = [("clean", 0), ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("blur", 1.0), ("blur", 2.0),
              ("noise", 5), ("noise", 10), ("downscale", 0.75), ("downscale", 0.5)]


def corrupt(image, kind, level, rng):
    if kind == "clean":
        return image
    if kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1].copy()
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), level)
    if kind == "noise":
        noisy = image.astype(np.float32) + rng.normal(0, level, image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "downscale":
        height, width = image.shape[:2]
        small = cv2.resize(image, (max(8, int(width * level)), max(8, int(height * level))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)


def segmenter_runner(args, paths):
    import torch

    from dataclasses import replace

    from forgery.evaluate import apply_gates, build_records
    from forgery.postprocess import MaskConfig
    from forgery.model import ForgerySegmenter
    from forgery.predict import probability_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForgerySegmenter(paths.dinov2_dir, args.head, args.encoder_input, args.img_size)
    checkpoint = Path(args.checkpoint)
    model.load_state_dict(torch.load(checkpoint / "checkpoint.pt" if checkpoint.is_dir() else checkpoint, map_location="cpu"))
    model.to(device).eval()
    gates = json.loads((Path(args.report_dir) / "report.json").read_text())["gates"]
    mask_config = replace(MaskConfig(), k_sigma=gates.get("k_sigma", MaskConfig().k_sigma))
    print(f"[robustness] gates from validation: {gates}")

    def run(table, kind, level):
        rng = np.random.default_rng(SEED)
        probs = {}
        for row in tqdm(list(table.itertuples(index=False)), desc=f"{kind} {level}", leave=False):
            image = corrupt(np.asarray(Image.open(image_path(row, paths)).convert("RGB")), kind, level, rng)
            probs[row.uid] = probability_map(model, image, device, args.tta, args.img_size).astype(np.float16)
        records = build_records(table, probs, paths, mask_config, desc=f"{kind} {level} masks")
        return apply_gates(records, gates["area_min"], gates["prob_min"]), "mean_probability"

    return run


def sift_runner(args, paths):
    config = SiftConfig()

    def run(table, kind, level):
        rng = np.random.default_rng(SEED)
        rows = []
        for row in tqdm(list(table.itertuples(index=False)), desc=f"{kind} {level}", leave=False):
            image = corrupt(np.asarray(Image.open(image_path(row, paths)).convert("RGB")), kind, level, rng)
            instances, inliers = detect(image, config)
            kept = instances if inliers >= args.min_inliers else []
            stats = metrics.image_stats(kept, mask_instances(row, paths), (int(row.H), int(row.W)))
            stats.update(id=row.id, group=row.group, label=row.label, score=1 - np.exp(-inliers / 10))
            rows.append(stats)
        return pd.DataFrame(rows), "score"

    return run


def robustness_run(args, paths=None):
    """Score a segmenter or the SIFT baseline on corrupted test images."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    table = table[table.split == "test"]
    if args.max_images:
        table = table.head(args.max_images)
    run = segmenter_runner(args, paths) if args.method == "segmenter" else sift_runner(args, paths)
    rows = []
    for kind, level in tqdm(CONDITIONS, desc="conditions"):
        frame, score = run(table, kind, level)
        summary = metrics.summarize(frame, score)
        rows.append({"corruption": kind, "level": level, **summary})
        print(f"[robustness] {kind:<10} {level:<5} forged F1 {summary['forged_pixel_f1']:.3f} | mixed F1 "
              f"{summary['mixed_pixel_f1']:.3f} | TNR {summary['authentic_tnr']:.3f} | official {summary['official_score']:.3f}",
              flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(run_dir / f"robustness_{args.method}.csv", index=False)
    return result

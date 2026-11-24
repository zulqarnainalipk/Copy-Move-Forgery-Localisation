"""Classical keypoint baseline: self-matching SIFT keypoints with G2NN and affine RANSAC."""
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from forgery import metrics, selection
from forgery.config import Paths
from forgery.data import image_path, load_split_table, mask_instances

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class SiftConfig:
    max_side: int = 2048
    max_keypoints: int = 8000
    contrast_threshold: float = 0.001
    g2nn_ratio: float = 0.7
    max_neighbours: int = 6
    min_offset_fraction: float = 0.03
    ransac_threshold: float = 3.0
    max_instances: int = 3
    min_inliers_floor: int = 4


def match_self(descriptors, points, config, diagonal):
    """Generalised 2-nearest-neighbour matching of keypoints against the same image."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(descriptors, descriptors, k=config.max_neighbours + 1)
    pairs = []
    for matches in knn:
        matches = [m for m in matches if m.trainIdx != m.queryIdx]
        distances = [m.distance for m in matches]
        accepted = []
        for n in range(len(distances) - 1):
            if distances[n + 1] == 0 or distances[n] / distances[n + 1] >= config.g2nn_ratio:
                break
            accepted.append(matches[n])
        for m in accepted:
            if np.linalg.norm(points[m.queryIdx] - points[m.trainIdx]) > config.min_offset_fraction * diagonal:
                pairs.append((m.queryIdx, m.trainIdx))
    return pairs


def hull_mask(points, shape, scale):
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull((points / scale).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def instances_from_keypoints(points, descriptors, config, gray_shape, scale, out_shape):
    """Self-match descriptors, fit affine transforms with RANSAC and return hull masks at the output size."""
    if descriptors is None or len(points) < 8:
        return [], 0
    pairs = match_self(descriptors, points, config, float(np.hypot(*gray_shape)))
    instances, best_inliers = [], 0
    remaining = list(pairs)
    height, width = out_shape
    for _ in range(config.max_instances):
        if len(remaining) < config.min_inliers_floor:
            break
        source = np.array([points[i] for i, _ in remaining])
        target = np.array([points[j] for _, j in remaining])
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                      ransacReprojThreshold=config.ransac_threshold)
        if matrix is None or inliers is None or int(inliers.sum()) < config.min_inliers_floor:
            break
        flags = inliers.ravel().astype(bool)
        best_inliers = max(best_inliers, int(flags.sum()))
        both = np.concatenate([source[flags], target[flags]])
        mask = hull_mask(both, (int(round(height * scale)), int(round(width * scale))), 1.0)
        instances.append(cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool))
        remaining = [p for p, keep in zip(remaining, flags) if not keep]
    return instances, best_inliers


def detect(image, config=SiftConfig()):
    """Return (instance masks at original resolution, number of RANSAC inliers of the best instance)."""
    height, width = image.shape[:2]
    scale = min(1.0, config.max_side / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=config.max_keypoints, contrastThreshold=config.contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return [], 0
    points = np.array([k.pt for k in keypoints], np.float32)
    return instances_from_keypoints(points, descriptors, config, gray.shape, scale, (height, width))


def run_baseline(table, paths, config, desc, detector=detect):
    """Run a keypoint detector on every image of a split table."""
    records = []
    for row in tqdm(list(table.itertuples(index=False)), desc=desc):
        image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
        instances, inliers = detector(image, config)
        records.append({"row": row, "instances": instances, "inliers": inliers})
    return records


def score_records(records, paths, min_inliers):
    """Score keypoint detections against the ground truth at a minimum inlier count."""
    rows = []
    for record in records:
        row = record["row"]
        gt = mask_instances(row, paths)
        keep = record["inliers"] >= min_inliers
        instances = record["instances"] if keep else []
        stats = metrics.image_stats(instances, gt, (int(row.H), int(row.W)))
        stats.update(id=row.id, group=row.group, label=row.label, size_bin=row.size_bin,
                     touches_border=row.touches_border, appearance=row.appearance,
                     inliers=record["inliers"], score=1 - np.exp(-record["inliers"] / 10))
        rows.append(stats)
    return pd.DataFrame(rows)


def evaluate_baseline(args, paths=None, detector=detect, label="sift"):
    """Choose min_inliers on validation and report the keypoint baseline on every split."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    config = SiftConfig(max_side=args.max_side)
    objectives = getattr(args, "objectives", None) or ["mixed"]
    parts = {}
    for split in ("val", "test", "supp"):
        part = table[table.split == split]
        parts[split] = part.head(args.max_images) if args.max_images else part
    records = {split: run_baseline(part, paths, config, f"{label} {split}", detector) for split, part in parts.items()}
    sweep = []
    for min_inliers in [4, 6, 8, 10, 15, 20, 30]:
        summary = metrics.summarize(score_records(records["val"], paths, min_inliers), "score")
        sweep.append({"min_inliers": min_inliers, **summary})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(run_dir / "sweep_val.csv", index=False)
    report = {}
    for objective in objectives:
        chosen = int(selection.choose(sweep, objective).min_inliers)
        print(f"[{label}] objective {objective}: min_inliers {chosen} (validation {selection.DESCRIPTIONS[objective]})")
        tag = selection.suffix(objective, objectives[0])
        report = {"objective": objective, "min_inliers": chosen}
        for split, split_records in records.items():
            frame = score_records(split_records, paths, chosen)
            frame.to_csv(run_dir / f"per_image_{split}{tag}.csv", index=False)
            summary = metrics.summarize(frame, "score")
            summary["ci95"] = metrics.bootstrap_summary(frame, "group", args.bootstrap, score_column="score")
            report[split] = summary
            print(f"[{label}] {split}: mixed F1 {summary['mixed_pixel_f1']:.3f} | forged F1 {summary['forged_pixel_f1']:.3f} | "
                  f"TNR {summary['authentic_tnr']:.3f} | official {summary['official_score']:.3f}")
        with open(run_dir / f"report{tag}.json", "w") as handle:
            json.dump(report, handle, indent=2)
    return report

"""Fusion of the keypoint matcher and the segmenter, with fusion rules chosen on validation."""
import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from forgery import metrics, selection
from forgery.baselines import SiftConfig, detect
from forgery.config import Paths
from forgery.data import image_path, load_split_table, mask_instances
from forgery.postprocess import MaskConfig, candidate, split_instances

Image.MAX_IMAGE_PIXELS = None

MODES = ("seg_only", "sift_only", "sift_priority", "seg_priority", "union", "confirm", "refine")
INLIER_GRID = (6, 10, 15, 20, 30)
PROB_GRID = (0.10, 0.15, 0.20, 0.30, 0.40)
SPLIT_NAMES = ("val", "test", "supp")


def detect_task(path):
    image = np.asarray(Image.open(path).convert("RGB"))
    instances, inliers = detect(image, SiftConfig())
    return instances, inliers


def prune_to_grid(mode, inliers_grid, prob_grid):
    if mode == "seg_only":
        return [(0, p) for p in prob_grid]
    if mode == "sift_only":
        return [(m, 0.0) for m in inliers_grid]
    return list(product(inliers_grid, prob_grid))


def fuse(mode, matcher_inliers, matcher_instances, seg_mask, seg_probability, min_inliers, min_probability):
    """Return the predicted instance list for one image."""
    matcher_on = matcher_inliers >= min_inliers
    seg_on = seg_mask.any() and seg_probability >= min_probability
    if mode == "seg_only":
        return split_instances(seg_mask) if seg_on else []
    if mode == "sift_only":
        return list(matcher_instances) if matcher_on else []
    if mode == "sift_priority":
        if matcher_on:
            return list(matcher_instances)
        return split_instances(seg_mask) if seg_on else []
    if mode == "seg_priority":
        if seg_on:
            return split_instances(seg_mask)
        return list(matcher_instances) if matcher_on else []
    if mode == "union":
        if not (matcher_on or seg_on):
            return []
        merged = np.zeros(seg_mask.shape, bool)
        if matcher_on:
            for instance in matcher_instances:
                merged |= instance
        if seg_on:
            merged |= seg_mask.astype(bool)
        return [merged]
    if mode == "confirm":
        return split_instances(seg_mask) if (matcher_on and seg_on) else []
    if mode == "refine":
        if matcher_on:
            if seg_mask.any() and matcher_instances:
                hulls = np.any(np.stack(matcher_instances), axis=0).astype(np.uint8)
                near = cv2.dilate(hulls, np.ones((41, 41), np.uint8)).astype(bool)
                restricted = seg_mask.astype(bool) & near
                if restricted.any():
                    return split_instances(restricted.astype(np.uint8))
            return list(matcher_instances)
        return split_instances(seg_mask) if seg_on else []
    raise ValueError(mode)


def image_row(row, instances, paths, score):
    stats = metrics.image_stats(instances, mask_instances(row, paths), (int(row.H), int(row.W)))
    stats.update(id=row.id, group=row.group, label=row.label, size_bin=row.size_bin,
                 touches_border=row.touches_border, appearance=row.appearance, score=score)
    return stats


def prepare(table, probs, paths, workers, desc, mask_config=MaskConfig()):
    rows = list(table.itertuples(index=False))
    with Pool(workers) as pool:
        detections = list(tqdm(pool.imap(detect_task, [str(image_path(row, paths)) for row in rows], chunksize=1),
                               total=len(rows), desc=f"{desc} keypoints"))
    prepared = []
    for row, (instances, inliers) in zip(rows, detections):
        full, area, seg_probability, _ = candidate(probs[row.uid].astype(np.float32), (int(row.W), int(row.H)), mask_config)
        prepared.append((row, instances, inliers, full, seg_probability))
    return prepared


def score_of(inliers, seg_probability):
    return max(seg_probability, 1 - float(np.exp(-inliers / 10)))


def sweep(prepared, paths, grid_by_mode):
    results = {}
    for mode, grid in grid_by_mode.items():
        for min_inliers, min_probability in tqdm(grid, desc=f"val {mode}", leave=False):
            rows = [image_row(row, fuse(mode, inliers, instances, full, probability, min_inliers, min_probability),
                              paths, score_of(inliers, probability))
                    for row, instances, inliers, full, probability in prepared]
            frame = pd.DataFrame(rows)
            summary = metrics.summarize(frame, "score")
            results[(mode, min_inliers, min_probability)] = summary
    return results


def evaluate_config(prepared, paths, mode, min_inliers, min_probability, bootstrap):
    rows = [image_row(row, fuse(mode, inliers, instances, full, probability, min_inliers, min_probability),
                      paths, score_of(inliers, probability))
            for row, instances, inliers, full, probability in prepared]
    frame = pd.DataFrame(rows)
    summary = metrics.summarize(frame, "score")
    summary["ci95"] = metrics.bootstrap_summary(frame, "group", bootstrap, score_column="score")
    return frame, summary


def hybrid_run(args, paths=None):
    """Select gates for each fusion rule on validation and report every rule on all splits."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    prepared_by_split = {}
    for split in SPLIT_NAMES:
        cache = cache_dir / f"probs_{split}.npz"
        if not cache.exists():
            continue
        part = table[table.split == split]
        if args.max_images:
            part = part.head(args.max_images)
        prepared_by_split[split] = prepare(part, dict(np.load(cache)), paths, args.workers, split, MaskConfig(k_sigma=args.k_sigma))
    grids = {mode: prune_to_grid(mode, INLIER_GRID, PROB_GRID) for mode in MODES}
    results = sweep(prepared_by_split["val"], paths, grids)
    table_rows = [{"mode": mode, "min_inliers": m, "min_probability": p, **summary}
                  for (mode, m, p), summary in results.items()]
    pd.DataFrame(table_rows).to_csv(run_dir / "sweep_val.csv", index=False)
    sweep_table = pd.DataFrame(table_rows)
    objectives = getattr(args, "objectives", None) or ["mixed"]
    report = {}
    for objective in objectives:
        tag = selection.suffix(objective, objectives[0])
        report = {"objective": objective}
        for mode in MODES:
            best = selection.choose(sweep_table[sweep_table["mode"] == mode], objective, kind="stable")
            m, p = int(best.min_inliers), float(best.min_probability)
            report[mode] = {"min_inliers": m, "min_probability": p}
            print(f"[hybrid] {objective} {mode:<14} chosen on val: min_inliers {m}, min_probability {p:.2f}", flush=True)
            for split, prepared in prepared_by_split.items():
                frame, summary = evaluate_config(prepared, paths, mode, m, p, args.bootstrap)
                frame.to_csv(run_dir / f"per_image_{mode}_{split}{tag}.csv", index=False)
                report[mode][split] = summary
                print(f"[hybrid] {objective} {mode:<14} {split:<4} mixed F1 {summary['mixed_pixel_f1']:.3f} | forged F1 "
                      f"{summary['forged_pixel_f1']:.3f} | TNR {summary['authentic_tnr']:.3f} | "
                      f"official {summary['official_score']:.3f}", flush=True)
        with open(run_dir / f"report{tag}.json", "w") as handle:
            json.dump(report, handle, indent=2)
    return report

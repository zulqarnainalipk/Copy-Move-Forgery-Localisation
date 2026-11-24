"""Render image, ground truth and per-method predictions for chosen cases."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from forgery import metrics
from forgery.baselines import SiftConfig, detect
from forgery.config import Paths
from forgery.data import image_path, load_split_table, mask_instances
from forgery.hybrid import fuse
from forgery.postprocess import MaskConfig, candidate

PANEL_SIDE = 800


def shrink(array, interpolation):
    height, width = array.shape[:2]
    scale = PANEL_SIDE / max(height, width)
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=interpolation)


def overlay(image, instances, colour):
    shown = shrink(image, cv2.INTER_AREA).copy()
    for instance in instances:
        small = shrink(instance.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
        shown[small] = (0.45 * shown[small] + 0.55 * np.array(colour)).astype(np.uint8)
        contours, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shown, contours, -1, tuple(int(c) for c in colour), 2)
    return shown


def matches_panel(image, instances_pairs_source):
    """Image with the self-matches that survive geometric verification drawn as lines between keypoints."""
    points, pairs = instances_pairs_source
    scale = PANEL_SIDE / max(image.shape[:2])
    shown = shrink(image, cv2.INTER_AREA).copy()
    for i, j in pairs[:80]:
        a, b = (points[i] * scale).astype(int), (points[j] * scale).astype(int)
        cv2.line(shown, tuple(a), tuple(b), (136, 34, 85), 1, cv2.LINE_AA)
        cv2.circle(shown, tuple(a), 2, (204, 51, 17), -1)
        cv2.circle(shown, tuple(b), 2, (204, 51, 17), -1)
    return shown


def sift_matches(image, config=SiftConfig()):
    from forgery.baselines import match_self

    height, width = image.shape[:2]
    scale = min(1.0, config.max_side / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=config.max_keypoints, contrastThreshold=config.contrast_threshold).detectAndCompute(gray, None)
    points = np.array([k.pt for k in keypoints], np.float32) / scale
    pairs = match_self(descriptors, points * scale, config, float(np.hypot(*gray.shape)))
    return points, pairs


def render_pairs(args, paths=None):
    """Real near-duplicate pairs that a random split separates (random-split val versus train) and a grouped split keeps together."""
    paths = paths or Paths.from_env()
    out_dir = paths.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    table = table[table.label != "supp"].rename(columns={"old_split": "random_split"})
    found, info = 0, {}
    for group, part in table.groupby("group"):
        if len(part) != 2 or part.random_split.nunique() != 2 or part.split.nunique() != 1 or group < 0:
            continue
        train_row, val_row = (part[part.random_split == "train"].iloc[0], part[part.random_split == "val"].iloc[0])
        for tag, row in (("train", train_row), ("held_out", val_row)):
            image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
            Image.fromarray(shrink(image, cv2.INTER_AREA)).save(out_dir / f"pair{found}_{tag}.png")
        info[f"pair{found}"] = {"group": int(group), "train": train_row.uid, "held_out": val_row.uid, "grouped_split": part.split.iloc[0]}
        found += 1
        if found >= args.pairs:
            break
    (out_dir / "pairs.json").write_text(json.dumps(info, indent=2))
    print(f"[cases] {found} near-duplicate pairs written")


SHEET_LABELS = {"image": "image", "truth": "ground truth", "sift": "SIFT matcher", "segmenter": "main segmenter",
                "union": "hybrid union", "refine": "hybrid refine", "probability": "probability", "matches": "SIFT matches"}


def contact_sheet(panels, order, title="", columns=2, cell=(PANEL_SIDE, 3 * PANEL_SIDE // 4), gap=12):
    """Grid of the chosen panels (aspect kept inside equal cells, at most two per row), a label above each and a title line on top."""
    cell_width, cell_cap = cell
    rows = -(-len(order) // columns)
    fitted = {k: min(cell_width / panels[k].shape[1], cell_cap / panels[k].shape[0]) for k in order}
    row_heights = [max(1, max(int(panels[k].shape[0] * fitted[k]) for k in order[r * columns:(r + 1) * columns])) for r in range(rows)]
    label_band, header = 44, 60 if title else 0
    row_tops = [header + sum(label_band + h + gap for h in row_heights[:r]) for r in range(rows)]
    sheet = np.full((header + sum(label_band + h + gap for h in row_heights), columns * cell_width + (columns + 1) * gap, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if title:
        cv2.putText(sheet, title, (gap, 42), font, 1.05, (0, 0, 0), 2, cv2.LINE_AA)
    for index, key in enumerate(order):
        panel = panels[key]
        cell_height = row_heights[index // columns]
        scale = min(cell_width / panel.shape[1], cell_height / panel.shape[0])
        size = (max(1, int(panel.shape[1] * scale)), max(1, int(panel.shape[0] * scale)))
        tile = cv2.resize(panel, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        left = gap + (index % columns) * (cell_width + gap)
        top = row_tops[index // columns]
        y, x = top + label_band + (cell_height - size[1]) // 2, left + (cell_width - size[0]) // 2
        cv2.putText(sheet, SHEET_LABELS.get(key, key), (x, y - 12), font, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
        sheet[y:y + size[1], x:x + size[0]] = tile
    return sheet


def render_cases(args, paths=None):
    paths = paths or Paths.from_env()
    out_dir = paths.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv").set_index("uid", drop=False)
    cache = Path(args.cache_dir)
    probs = {split: dict(np.load(cache / f"probs_{split}.npz")) for split in ("test", "supp")}
    mask_config = MaskConfig(k_sigma=args.k_sigma)
    summary = {}
    tokens = list(args.cases)
    if getattr(args, "supp_all", False):
        tokens += [f"{uid}:{uid}" for uid in table[table.label == "supp"].uid]
    for token in tokens:
        name, uid = token.split(":")
        row = next(table.loc[[uid]].itertuples(index=False))
        image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
        height, width = image.shape[:2]
        gt = mask_instances(row, paths)
        probability = probs["supp" if row.label == "supp" else "test"][uid].astype(np.float32)
        full, area, mean_probability, _ = candidate(probability, (width, height), mask_config)
        matcher_instances, inliers = detect(image, SiftConfig())
        predictions = {
            "sift": fuse("sift_only", inliers, matcher_instances, full, mean_probability, args.sift_inliers, 0.0),
            "segmenter": fuse("seg_only", inliers, matcher_instances, full, mean_probability, 0, args.seg_probability) if area >= args.area_min else [],
            "refine": fuse("refine", inliers, matcher_instances, full, mean_probability, args.refine_inliers, args.refine_probability),
            "union": fuse("union", inliers, matcher_instances, full, mean_probability, args.union_inliers, args.union_probability),
        }
        colours = {"truth": (0, 119, 187), "sift": (238, 119, 51), "segmenter": (204, 51, 17), "refine": (0, 153, 136), "union": (238, 51, 119)}
        panels = {"image": shrink(image, cv2.INTER_AREA), "truth": overlay(image, gt, colours["truth"])}
        panels.update({key: overlay(image, value, colours[key]) for key, value in predictions.items()})
        heat = cv2.applyColorMap((shrink(cv2.resize(probability, (width, height)), cv2.INTER_LINEAR).clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        panels["probability"] = heat[..., ::-1]
        panels["matches"] = matches_panel(image, sift_matches(image))
        stats = {}
        for key, value in predictions.items():
            result = metrics.image_stats(value, gt, (height, width))
            stats[key] = {"pixel_f1": result["pixel_f1"], "official_score": result["official_score"], "empty": result["pred_empty"]}
        summary[name] = {"uid": uid, "height": height, "width": width, "inliers": int(inliers), "mean_probability": mean_probability, **stats}
        if getattr(args, "sheet", False):
            title = f"{uid}    pixel F1: SIFT {stats['sift']['pixel_f1']:.2f}   segmenter {stats['segmenter']['pixel_f1']:.2f}   union {stats['union']['pixel_f1']:.2f}"
            sheet = contact_sheet(panels, ["image", "truth", "sift", "segmenter", "union", "probability"], title)
            Image.fromarray(sheet).save(out_dir / f"sheet_{name}.jpg", quality=92)
        else:
            for key, panel in panels.items():
                Image.fromarray(panel).save(out_dir / f"{name}_{key}.png")
        print(f"[cases] {name} ({uid}): " + " | ".join(f"{k} F1 {v['pixel_f1']:.2f}" for k, v in stats.items()), flush=True)
    (out_dir / "cases.json").write_text(json.dumps(summary, indent=2))
    return summary

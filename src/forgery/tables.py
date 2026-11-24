"""Collect the per-run reports under results/ into the comparison tables used in the paper."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["forged_pixel_f1", "forged_pixel_iou", "forged_pixel_mcc", "mixed_pixel_f1", "authentic_tnr",
           "forged_detection_rate", "image_auroc", "image_mcc", "official_score"]

SEGMENTERS = [
    ("DINOv2-base, crop 224 (g1_base)", "c4_eval_g1"),
    ("+ full 518 input (g2a_full)", "c4_eval_g2a_full_input"),
    ("+ strong augmentation, crop 224 (g2d_aug)", "c4_eval_g2d_aug"),
    ("+ BCE and Dice (g2b_dice)", "c4_eval_g2b_dice"),
    ("+ focal loss (g2c_focal)", "c4_eval_g2c_focal"),
    ("linear head (g2e_linear)", "c4_eval_g2e_linear"),
    ("full input + augmentation, fixed k_sigma (g3_full_aug)", "c4_eval_g3_full_aug"),
    ("full input + augmentation, validation-chosen k_sigma (g3_full_aug_ks)", "c4_evalks_g3_full_aug"),
    ("g3_full_aug_ks, seed 1", "c4_evalks_g3_seed1"),
    ("g3_full_aug_ks, seed 2", "c4_evalks_g3_seed2"),
    ("full input + panel collages (g7_panel)", "c4_eval_g7_panel"),
    ("DINOv2-large (g6_large)", "c4_eval_g6_large"),
]
BASELINES = [
    ("SIFT self-matching", "c3_sift", None),
    ("SuperPoint self-matching", "baseline_superpoint", None),
    ("Zero-shot DINOv2 self-similarity", "baseline_dense", None),
]
HYBRIDS = [("g3_full_aug", "c6_hybrid_g3"), ("g3_full_aug_ks", "c6_hybrid_g3_ks"),("g2a_full", "c6_hybrid_g2a"), ("g7_panel", "c6_hybrid_g7_panel"),
           ("g6_large", "c6_hybrid_g6_large")]
RULES = ("seg_only", "sift_only", "sift_priority", "seg_priority", "union", "confirm", "refine")


def load_report(results, folder):
    for candidate in (results / folder / "report.json", *(results / folder).glob("*/report.json")):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def cell(summary, column):
    value = summary.get(column, np.nan)
    interval = summary.get("ci95", {}).get(column)
    if value is None or value != value:
        return ""
    return f"{value:.3f}" + (f" [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "")


def row(name, summary):
    return {"method": name, **{column: cell(summary, column) for column in COLUMNS}}


def main_tables(results):
    tables = {}
    for split in ("test", "supp"):
        rows = []
        for name, folder, _ in BASELINES:
            report = load_report(results, folder)
            if report and split in report:
                rows.append(row(name, report[split]))
        for name, folder in SEGMENTERS:
            report = load_report(results, folder)
            if report and split in report:
                rows.append(row(name, report[split]))
        for backbone, folder in HYBRIDS:
            report = load_report(results, folder)
            if not report:
                continue
            for rule in RULES:
                if rule in report and split in report[rule] and rule not in ("seg_only", "sift_only"):
                    rows.append(row(f"hybrid {rule} on {backbone}", report[rule][split]))
        tables[f"main_{split}"] = pd.DataFrame(rows)
    return tables


def ablation_tables(results):
    tables = {}
    for folder in sorted(results.glob("c5_ablate_*")):
        frame = pd.read_csv(folder / "ablation_test.csv")
        tables[f"ablation_{folder.name.removeprefix('c5_ablate_')}"] = frame
        strata = folder / "stratified_test.csv"
        if strata.exists():
            tables[f"stratified_{folder.name.removeprefix('c5_ablate_')}"] = pd.read_csv(strata)
    return tables


def robustness_table(results):
    frames = []
    for path in sorted(results.glob("**/robustness_*.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "run", f"{path.parent.name}/{path.stem.removeprefix('robustness_')}")
        frames.append(frame[["run", "corruption", "level", "forged_pixel_f1", "mixed_pixel_f1", "authentic_tnr", "official_score"]])
    return {"robustness": pd.concat(frames)} if frames else {}


def benchmark_table(results):
    path = results / "g5_benchmark" / "benchmark.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    rows = [{"component": key, "median_ms": 1000 * value["median_s"], "p90_ms": 1000 * value["p90_s"],
             "peak_memory_mb": value.get("peak_memory_mb", np.nan)} for key, value in report.items() if isinstance(value, dict)]
    return {"runtime": pd.DataFrame(rows)}


def training_table(results):
    rows = []
    for history in sorted(results.glob("g*/history.csv")):
        frame = pd.read_csv(history)
        best = frame.loc[frame.val_loss.idxmin()]
        rows.append({"run": history.parent.name, "epochs": len(frame), "best_val_loss": best.val_loss,
                     "minutes": frame.seconds.sum() / 60})
    return {"training": pd.DataFrame(rows)} if rows else {}


CONFUSION_SOURCES = [
    ("SIFT self-matching", "c3_sift", "per_image_{split}.csv"),
    ("SuperPoint self-matching", "baseline_superpoint", "per_image_{split}.csv"),
    ("g1_base", "c4_eval_g1", "per_image_{split}.csv"),
    ("g2a_full", "c4_eval_g2a_full_input", "per_image_{split}.csv"),
    ("g3_full_aug (final segmenter)", "c4_evalks_g3_full_aug", "per_image_{split}.csv"),
    ("hybrid refine", "c6_hybrid_g3_ks", "per_image_refine_{split}.csv"),
    ("hybrid union", "c6_hybrid_g3_ks", "per_image_union_{split}.csv"),
    ("hybrid confirm", "c6_hybrid_g3_ks", "per_image_confirm_{split}.csv"),
]


def confusion_tables(results):
    """Image-level and pixel-level true and false positives and negatives from the stored per-image statistics."""
    tables = {}
    for split in ("test", "supp"):
        rows = []
        for name, folder, pattern in CONFUSION_SOURCES:
            path = results / folder / pattern.format(split=split)
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            forged, pred = frame.is_forged.astype(bool), ~frame.pred_empty.astype(bool)
            rows.append({"method": name,
                         "image_tp": int((forged & pred).sum()), "image_fn": int((forged & ~pred).sum()),
                         "image_fp": int((~forged & pred).sum()), "image_tn": int((~forged & ~pred).sum()),
                         "pixel_tp": int(frame.tp.sum()), "pixel_fp": int(frame.fp.sum()),
                         "pixel_fn": int(frame.fn.sum()), "pixel_tn": int(frame.tn.sum())})
        if rows:
            tables[f"confusion_{split}"] = pd.DataFrame(rows)
    return tables


def build_tables(results_dir):
    """Build every results table from results/ and write CSV and Markdown copies to results/tables."""
    results = Path(results_dir)
    output = results / "tables"
    output.mkdir(exist_ok=True)
    tables = {}
    for builder in (main_tables, confusion_tables, ablation_tables, robustness_table, benchmark_table, training_table):
        tables.update(builder(results))
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
        (output / f"{name}.md").write_text(frame.to_markdown(index=False, floatfmt=".3f") + "\n")
        print(f"[tables] {name}: {len(frame)} rows")
    return tables

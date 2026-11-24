"""Post-processing ablations and stratified analysis on cached probability maps."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from forgery import metrics
from forgery.config import Paths
from forgery.data import load_split_table
from forgery.evaluate import apply_gates, build_records, sweep_gates
from forgery.postprocess import MaskConfig

AREA_GRID = [0, 50, 100, 200, 400, 800]
PROB_GRID = [round(x, 2) for x in np.arange(0.0, 0.61, 0.05)]
STRATA = ("appearance", "size_bin", "touches_border")
HEADLINE = ("mixed_pixel_f1", "forged_pixel_f1", "forged_pixel_mcc", "authentic_tnr", "image_auroc", "official_score")

VARIANTS = {
    "full_pipeline": MaskConfig(),
    "no_gradient_blend": replace(MaskConfig(), alpha=0.0),
    "no_smoothing": replace(MaskConfig(), blur=False),
    "no_morphology": replace(MaskConfig(), morphology=False),
    "plain_threshold": replace(MaskConfig(), alpha=0.0, blur=False, morphology=False),
    "k_sigma_0.0": replace(MaskConfig(), k_sigma=0.0),
    "k_sigma_0.6": replace(MaskConfig(), k_sigma=0.6),
    "k_sigma_1.0": replace(MaskConfig(), k_sigma=1.0),
    "alpha_0.25": replace(MaskConfig(), alpha=0.25),
    "alpha_0.65": replace(MaskConfig(), alpha=0.65),
}


def pick_gates(records, use_gates):
    """Choose area and probability gates on validation by mixed F1, or disable them."""
    if not use_gates:
        return 0, 0.0
    sweep = sweep_gates(records, AREA_GRID, PROB_GRID)
    best = sweep.sort_values("mixed_pixel_f1", ascending=False).iloc[0]
    return int(best.area_min), float(best.prob_min)


def stratified(frame):
    """Forged-image F1, MCC and detection rate per appearance, size bin and border stratum."""
    rows = []
    forged = frame[frame.label != "authentic"]
    for column in STRATA:
        for value, part in forged.groupby(column):
            summary = metrics.summarize(part, "mean_probability")
            rows.append({"stratum": column, "value": str(value), "images": len(part),
                         "forged_pixel_f1": summary["forged_pixel_f1"],
                         "forged_pixel_mcc": summary["forged_pixel_mcc"],
                         "forged_detection_rate": summary.get("forged_detection_rate", np.nan)})
    return pd.DataFrame(rows)


def ablate_run(args, paths=None):
    """Run every post-processing variant on cached maps and write the ablation and stratified tables."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    probs = {split: dict(np.load(cache_dir / f"probs_{split}.npz")) for split in ("val", "test")}
    rows, default_test = [], None
    jobs = [(name, config, True) for name, config in VARIANTS.items()] + [("no_gates", MaskConfig(), False)]
    for name, config, use_gates in tqdm(jobs, desc="variants"):
        val_records = build_records(table[table.split == "val"], probs["val"], paths, config, f"{name} val")
        area_min, prob_min = pick_gates(val_records, use_gates)
        test_records = build_records(table[table.split == "test"], probs["test"], paths, config, f"{name} test")
        frame = apply_gates(test_records, area_min, prob_min)
        summary = metrics.summarize(frame, "mean_probability")
        rows.append({"variant": name, "area_min": area_min, "prob_min": prob_min,
                     **{key: summary[key] for key in HEADLINE}})
        print(f"[ablate] {name:<18} forged F1 {summary['forged_pixel_f1']:.3f} | mixed F1 {summary['mixed_pixel_f1']:.3f} | "
              f"TNR {summary['authentic_tnr']:.3f} | official {summary['official_score']:.3f}", flush=True)
        if name == "full_pipeline":
            default_test = frame
    pd.DataFrame(rows).to_csv(run_dir / "ablation_test.csv", index=False)
    strata = stratified(default_test)
    strata.to_csv(run_dir / "stratified_test.csv", index=False)
    print(strata.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    with open(run_dir / "ablation.json", "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows

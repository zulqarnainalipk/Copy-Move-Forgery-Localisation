"""Evaluation of cached probability maps under configurable post-processing, gates and selection objectives."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from forgery import metrics, selection
from forgery.config import Paths
from forgery.data import load_split_table, mask_instances
from forgery.postprocess import MaskConfig, candidate, passes_gates, split_instances

AREA_GRID = [0, 50, 100, 200, 400, 800]
PROB_GRID = [round(x, 2) for x in np.arange(0.0, 0.61, 0.05)]


def build_records(table, probs, paths, mask_config=MaskConfig(), desc="candidates"):
    """Per image: candidate area, mean probability, and statistics if the mask is kept or dropped."""
    records = []
    for row in tqdm(list(table.itertuples(index=False)), desc=desc):
        prob = probs[row.uid].astype(np.float32)
        size = (int(row.W), int(row.H))
        full, area, mean_probability, _ = candidate(prob, size, mask_config)
        gt = mask_instances(row, paths)
        shape = (int(row.H), int(row.W))
        kept = metrics.image_stats(split_instances(full) if area else [], gt, shape)
        dropped = metrics.image_stats([], gt, shape)
        records.append({"id": row.id, "group": row.group, "split": row.split, "label": row.label,
                        "appearance": row.appearance, "size_bin": row.size_bin,
                        "touches_border": row.touches_border, "area": area,
                        "mean_probability": mean_probability, "kept": kept, "dropped": dropped})
    return records


def frame_row(record, stats):
    stats = dict(stats)
    stats.update(id=record["id"], group=record["group"], label=record["label"],
                 appearance=record["appearance"], size_bin=record["size_bin"],
                 touches_border=record["touches_border"], mean_probability=record["mean_probability"],
                 area=record["area"])
    return stats


def apply_gates(records, area_min, prob_min):
    rows = []
    for record in records:
        keep = passes_gates(record["area"], record["mean_probability"], area_min, prob_min)
        rows.append(frame_row(record, record["kept"] if keep else record["dropped"]))
    return pd.DataFrame(rows)


def apply_gates_by(records, gates, column):
    """Gates chosen separately for each value of a record attribute, for example the appearance class."""
    rows = []
    for record in records:
        area_min, prob_min = gates[record[column]]
        keep = passes_gates(record["area"], record["mean_probability"], area_min, prob_min)
        rows.append(frame_row(record, record["kept"] if keep else record["dropped"]))
    return pd.DataFrame(rows)


def sweep_gates(records, area_grid, prob_grid):
    results = []
    for area_min in tqdm(area_grid, desc="area sweep"):
        for prob_min in prob_grid:
            summary = metrics.summarize(apply_gates(records, area_min, prob_min), "mean_probability")
            results.append({"area_min": area_min, "prob_min": prob_min, **summary})
    return pd.DataFrame(results)


def summarise_split(frame, bootstrap):
    summary = metrics.summarize(frame, "mean_probability")
    summary["ci95"] = metrics.bootstrap_summary(frame, "group", bootstrap, score_column="mean_probability")
    return summary


def report_line(split, summary):
    return (f"[eval] {split}: mixed F1 {summary['mixed_pixel_f1']:.3f} | forged F1 {summary['forged_pixel_f1']:.3f} | "
            f"forged MCC {summary['forged_pixel_mcc']:.3f} | TNR {summary['authentic_tnr']:.3f} | "
            f"official {summary['official_score']:.3f}")


def evaluate_run(args, paths=None):
    """Choose gates and k_sigma on validation, then score validation, test and supplemental images."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else run_dir
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    k_grid = args.k_sigma_grid or [MaskConfig().k_sigma]
    objectives = getattr(args, "objectives", None) or ["mixed"]
    gate_by = getattr(args, "gate_by", None)

    val_probs = dict(np.load(cache_dir / "probs_val.npz"))
    records = {}

    def records_for(split, k_sigma):
        key = (split, k_sigma)
        if key not in records:
            probs = val_probs if split == "val" else dict(np.load(cache_dir / f"probs_{split}.npz"))
            records[key] = build_records(table[table.split == split], probs, paths, replace(MaskConfig(), k_sigma=k_sigma),
                                         desc=f"{split} candidates k_sigma {k_sigma}")
        return records[key]

    sweeps = []
    for k_sigma in k_grid:
        sweeps.append(sweep_gates(records_for("val", k_sigma), AREA_GRID, PROB_GRID).assign(k_sigma=k_sigma))
    sweep = pd.concat(sweeps).reset_index(drop=True)
    sweep.to_csv(run_dir / "sweep_val.csv", index=False)

    available = [s for s in ("val", "test", "supp") if s == "val" or (cache_dir / f"probs_{s}.npz").exists()]
    first = objectives[0]
    for objective in objectives:
        best = selection.choose(sweep, objective)
        area_min, prob_min, k_sigma = int(best.area_min), float(best.prob_min), float(best.k_sigma)
        print(f"[eval] objective {objective}: k_sigma {k_sigma}, area_min {area_min}, prob_min {prob_min} "
              f"(validation {selection.DESCRIPTIONS[objective]})")
        report = {"objective": objective, "gates": {"area_min": area_min, "prob_min": prob_min, "k_sigma": k_sigma}}
        tag = selection.suffix(objective, first)
        for split in available:
            frame = apply_gates(records_for(split, k_sigma), area_min, prob_min)
            frame.to_csv(run_dir / f"per_image_{split}{tag}.csv", index=False)
            report[split] = summarise_split(frame, args.bootstrap)
            print(report_line(split, report[split]))
        with open(run_dir / f"report{tag}.json", "w") as handle:
            json.dump(report, handle, indent=2)

    if gate_by:
        best = selection.choose(sweep, first)
        k_sigma = float(best.k_sigma)
        gates = {}
        for value in sorted(set(r[gate_by] for r in records_for("val", k_sigma))):
            subset = [r for r in records_for("val", k_sigma) if r[gate_by] == value]
            chosen = selection.choose(sweep_gates(subset, AREA_GRID, PROB_GRID), first)
            gates[value] = (int(chosen.area_min), float(chosen.prob_min))
        print(f"[eval] gates by {gate_by}: {gates}")
        report = {"objective": first, "gate_by": gate_by, "k_sigma": k_sigma, "gates": {k: list(v) for k, v in gates.items()}}
        for split in available:
            frame = apply_gates_by(records_for(split, k_sigma), gates, gate_by)
            frame.to_csv(run_dir / f"per_image_{split}_by_{gate_by}.csv", index=False)
            report[split] = summarise_split(frame, args.bootstrap)
            print(report_line(split, report[split]))
        with open(run_dir / f"report_by_{gate_by}.json", "w") as handle:
            json.dump(report, handle, indent=2)
    return report

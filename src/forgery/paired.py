"""Paired group-bootstrap differences between two methods scored on the same images."""
import numpy as np
import pandas as pd


def aligned(frame_a, frame_b):
    keys = ["id", "label"]
    merged = frame_a[keys + ["group", "is_forged", "pixel_f1", "official_score"]].merge(
        frame_b[keys + ["pixel_f1", "official_score"]], on=keys, suffixes=("_a", "_b"))
    if len(merged) != len(frame_a) or len(merged) != len(frame_b):
        raise ValueError("methods were not scored on the same images")
    return merged


def paired_difference(frame_a, frame_b, metric="forged_pixel_f1", n_resamples=1000, seed=42):
    """Mean of (a minus b) over forged images (forged_pixel_f1) or over all images (official_score), with a group bootstrap interval."""
    merged = aligned(frame_a, frame_b)
    column = "pixel_f1" if metric == "forged_pixel_f1" else "official_score"
    if metric == "forged_pixel_f1":
        merged = merged[merged.is_forged]
    difference = (merged[f"{column}_a"] - merged[f"{column}_b"]).to_numpy()
    groups, inverse = np.unique(merged.group.to_numpy(), return_inverse=True)
    sums = np.bincount(inverse, weights=difference, minlength=len(groups))
    counts = np.bincount(inverse, minlength=len(groups)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(difference.mean()), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def paired_table(comparisons, results_dir, n_resamples=1000):
    rows = []
    for label, path_a, path_b, metric in comparisons:
        a = pd.read_csv(results_dir / path_a)
        b = pd.read_csv(results_dir / path_b)
        mean, low, high = paired_difference(a, b, metric, n_resamples)
        rows.append({"comparison": label, "metric": metric, "difference": mean, "ci95_low": low, "ci95_high": high,
                     "excludes_zero": bool(low > 0 or high < 0)})
    return pd.DataFrame(rows)

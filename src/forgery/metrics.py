"""Pixel, image-level and competition metrics for copy-move localisation."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def confusion(pred, gt):
    pred, gt = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = pred.size - tp - fp - fn
    return tp, fp, fn, tn


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def iou_from_counts(tp, fp, fn):
    denominator = tp + fp + fn
    return tp / denominator if denominator else 0.0


def mcc_from_counts(tp, fp, fn, tn):
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def instance_f1(pred, gt):
    tp, fp, fn, _ = confusion(pred, gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def match_summary(pred_instances, gt_instances):
    """Hungarian-matched statistics behind the competition score: mean matched F1, excess penalty and matched count."""
    matrix = np.zeros((len(pred_instances), len(gt_instances)))
    for i, pred in enumerate(pred_instances):
        for j, gt in enumerate(gt_instances):
            matrix[i, j] = instance_f1(pred, gt)
    if matrix.shape[0] < len(gt_instances):
        matrix = np.vstack([matrix, np.zeros((len(gt_instances) - matrix.shape[0], len(gt_instances)))])
    rows, cols = linear_sum_assignment(-matrix)
    penalty = len(gt_instances) / max(len(pred_instances), len(gt_instances))
    return float(np.mean(matrix[rows, cols])), float(penalty), int((matrix[rows, cols] > 0).sum())


def optimal_f1(pred_instances, gt_instances):
    """Hungarian-matched instance F1 with the competition's penalty for excess predictions."""
    matched_f1, penalty, _ = match_summary(pred_instances, gt_instances)
    return matched_f1 * penalty


def competition_image_score(pred_instances, gt_instances):
    """Official image score: authentic must match authentic, forged images use optimal_f1."""
    gt_authentic = len(gt_instances) == 0
    pred_authentic = len(pred_instances) == 0
    if gt_authentic or pred_authentic:
        return float(gt_authentic and pred_authentic)
    return optimal_f1(pred_instances, gt_instances)


def image_stats(pred_instances, gt_instances, shape):
    """Per-image statistics from lists of binary instance masks."""
    if pred_instances:
        pred_union = np.any(np.stack(pred_instances), axis=0)
    else:
        pred_union = np.zeros(shape, bool)
    if gt_instances:
        gt_union = np.any(np.stack(gt_instances), axis=0)
    else:
        gt_union = np.zeros(shape, bool)
    tp, fp, fn, tn = confusion(pred_union, gt_union)
    if pred_instances and gt_instances:
        matched_f1, penalty, matched = match_summary(pred_instances, gt_instances)
    else:
        matched_f1, penalty, matched = float("nan"), float("nan"), 0
    return {
        "is_forged": bool(gt_instances),
        "pred_empty": not pred_instances,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pixel_f1": f1_from_counts(tp, fp, fn) if (tp + fp + fn) else 1.0,
        "pixel_iou": iou_from_counts(tp, fp, fn),
        "pixel_mcc": mcc_from_counts(tp, fp, fn, tn),
        "official_score": competition_image_score(pred_instances, gt_instances),
        "n_pred": len(pred_instances), "n_gt": len(gt_instances),
        "matched": matched, "matched_f1": matched_f1, "excess_penalty": penalty,
    }


def summarize(frame, score_column=None):
    """Aggregate per-image statistics into the paper's headline metrics."""
    forged, authentic = frame[frame.is_forged], frame[~frame.is_forged]
    y_true = frame.is_forged.astype(int).to_numpy()
    y_pred = (~frame.pred_empty).astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    summary = {
        "n_images": len(frame), "n_forged": len(forged), "n_authentic": len(authentic),
        "mixed_pixel_f1": float(frame.pixel_f1.mean()),
        "official_score": float(frame.official_score.mean()),
        "forged_pixel_f1": float(forged.pixel_f1.mean()) if len(forged) else np.nan,
        "forged_pixel_iou": float(forged.pixel_iou.mean()) if len(forged) else np.nan,
        "forged_pixel_mcc": float(forged.pixel_mcc.mean()) if len(forged) else np.nan,
        "forged_official_score": float(forged.official_score.mean()) if len(forged) else np.nan,
        "forged_detection_rate": float((~forged.pred_empty).mean()) if len(forged) else np.nan,
        "authentic_tnr": float(authentic.pred_empty.mean()) if len(authentic) else np.nan,
        "image_mcc": mcc_from_counts(tp, fp, fn, tn),
        "image_tp": tp, "image_fp": fp, "image_fn": fn, "image_tn": tn,
        "image_precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "image_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "pixel_tp": int(frame.tp.sum()), "pixel_fp": int(frame.fp.sum()),
        "pixel_fn": int(frame.fn.sum()), "pixel_tn": int(frame.tn.sum()),
    }
    if score_column is not None and len(np.unique(y_true)) == 2:
        scores = frame[score_column].to_numpy()
        summary["image_auroc"] = float(roc_auc_score(y_true, scores))
        summary["image_ap"] = float(average_precision_score(y_true, scores))
        summary["image_ece"] = expected_calibration_error(y_true, scores)
    return summary


def expected_calibration_error(y_true, scores, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        selected = index == b
        if selected.any():
            error += selected.mean() * abs(y_true[selected].mean() - scores[selected].mean())
    return float(error)


def bootstrap_summary(frame, group_column, n_resamples=1000, seed=42, score_column=None, keys=None):
    """95% percentile intervals from resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    groups = {g: part for g, part in frame.groupby(group_column)}
    names = np.array(list(groups))
    draws = []
    for _ in tqdm(range(n_resamples), desc="bootstrap"):
        chosen = rng.choice(names, size=len(names), replace=True)
        draws.append(summarize(pd.concat([groups[g] for g in chosen], ignore_index=True), score_column))
    table = pd.DataFrame(draws)
    keys = keys or list(table.columns)
    return {k: (float(table[k].quantile(0.025)), float(table[k].quantile(0.975))) for k in keys if k.startswith(("mixed", "forged", "authentic", "image", "official"))}

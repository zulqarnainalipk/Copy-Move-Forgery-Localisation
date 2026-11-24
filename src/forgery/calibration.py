"""Calibration of the image-level score used by the gate: reliability bins, ECE, Brier score, Platt and isotonic mapping."""
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def reliability_table(y_true, score, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "images": int(mask.sum()),
                     "mean_score": float(score[mask].mean()) if mask.any() else np.nan,
                     "forged_share": float(y_true[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def ece_from_table(table, total):
    filled = table.dropna()
    return float((filled.images / total * (filled.mean_score - filled.forged_share).abs()).sum())


def brier(y_true, score):
    return float(np.mean((score - y_true) ** 2))


def summarise(y_true, score):
    table = reliability_table(y_true, score)
    return {"ece": ece_from_table(table, len(y_true)), "brier": brier(y_true, score), "auroc": float(roc_auc_score(y_true, score))}, table


def calibrate(val_frame, test_frame, column="mean_probability"):
    """Fit Platt and isotonic maps on validation scores and apply them to test scores."""
    yv, sv = val_frame.is_forged.astype(int).to_numpy(), val_frame[column].to_numpy(float)
    yt, st = test_frame.is_forged.astype(int).to_numpy(), test_frame[column].to_numpy(float)
    platt = LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), yv)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sv, yv)
    versions = {"raw score": st, "Platt": platt.predict_proba(st.reshape(-1, 1))[:, 1], "isotonic": isotonic.predict(st)}
    summary, tables = {}, {}
    for name, calibrated in versions.items():
        summary[name], tables[name] = summarise(yt, calibrated)
    return summary, tables


def run(args):
    """Calibrate the image-level score on validation and write reliability tables and a summary."""
    val = pd.read_csv(args.val_csv)
    test = pd.read_csv(args.test_csv)
    summary, tables = calibrate(val, test)
    for name, table in tables.items():
        table.to_csv(f"{args.output_prefix}_reliability_{name.replace(' ', '_')}.csv", index=False)
    with open(f"{args.output_prefix}_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(pd.DataFrame(summary).T.round(4).to_string())
    return summary

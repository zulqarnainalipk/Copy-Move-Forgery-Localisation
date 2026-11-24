import numpy as np
import pandas as pd
import pytest

from forgery import metrics


def block(shape, rows, cols):
    mask = np.zeros(shape, bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return mask


def test_counts_and_scores():
    pred, gt = block((10, 10), (0, 5), (0, 10)), block((10, 10), (0, 10), (0, 5))
    tp, fp, fn, tn = metrics.confusion(pred, gt)
    assert (tp, fp, fn, tn) == (25, 25, 25, 25)
    assert metrics.f1_from_counts(tp, fp, fn) == pytest.approx(0.5)
    assert metrics.iou_from_counts(tp, fp, fn) == pytest.approx(1 / 3)
    assert metrics.mcc_from_counts(tp, fp, fn, tn) == pytest.approx(0.0)


def test_empty_prediction_on_authentic_scores_one():
    stats = metrics.image_stats([], [], (8, 8))
    assert stats["pixel_f1"] == 1.0 and stats["official_score"] == 1.0 and not stats["is_forged"]


def test_empty_prediction_on_forged_scores_zero():
    stats = metrics.image_stats([], [block((8, 8), (0, 4), (0, 4))], (8, 8))
    assert stats["pixel_f1"] == 0.0 and stats["official_score"] == 0.0 and stats["is_forged"]


def test_false_alarm_on_authentic_scores_zero():
    stats = metrics.image_stats([block((8, 8), (0, 4), (0, 4))], [], (8, 8))
    assert stats["official_score"] == 0.0


def column(length, start, size):
    mask = np.zeros(1000, bool)
    mask[start - 1:start - 1 + size] = True
    return mask


def test_official_docstring_examples():
    shape = 1000
    same = [column(shape, 101, 102)]
    scores = [
        metrics.competition_image_score(same, [column(shape, 101, 103)]),
        metrics.competition_image_score(same, same),
        metrics.competition_image_score(same, same),
    ]
    assert np.mean(scores) == pytest.approx(0.9983739837398374)
    two_instances = [column(shape, 101, 102), column(shape, 300, 100)]
    scores = [
        metrics.competition_image_score(same, two_instances),
        metrics.competition_image_score(same, same),
        metrics.competition_image_score(same, same),
    ]
    assert np.mean(scores) == pytest.approx(0.8333333333333334)


def test_excess_predictions_are_penalised():
    gt = [block((8, 8), (0, 4), (0, 4))]
    pred = [block((8, 8), (0, 4), (0, 4)), block((8, 8), (5, 8), (5, 8))]
    assert metrics.optimal_f1(pred, gt) == pytest.approx(0.5)


def test_summary_and_bootstrap():
    rows = []
    for i in range(40):
        forged = i % 2 == 0
        pred = [block((8, 8), (0, 4), (0, 4))] if (forged and i % 4 == 0) or (not forged and i == 1) else []
        gt = [block((8, 8), (0, 4), (0, 4))] if forged else []
        row = metrics.image_stats(pred, gt, (8, 8))
        row.update(group=i // 2, score=0.9 if pred else 0.1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = metrics.summarize(frame, "score")
    assert summary["forged_detection_rate"] == pytest.approx(0.5)
    assert summary["authentic_tnr"] == pytest.approx(0.95)
    assert 0.5 < summary["image_auroc"] <= 1.0
    intervals = metrics.bootstrap_summary(frame, "group", n_resamples=50, score_column="score")
    assert intervals["forged_detection_rate"][0] <= 0.5 <= intervals["forged_detection_rate"][1]


def test_match_summary_explains_the_competition_score():
    from forgery.metrics import competition_image_score, image_stats, match_summary

    box = np.zeros((20, 20), bool)
    box[2:8, 2:8] = True
    other = np.zeros((20, 20), bool)
    other[12:18, 12:18] = True
    spurious = np.zeros((20, 20), bool)
    spurious[0:2, 14:20] = True
    matched_f1, penalty, matched = match_summary([box, spurious], [box])
    assert matched == 1 and abs(matched_f1 - 1.0) < 1e-9 and abs(penalty - 0.5) < 1e-9
    assert abs(matched_f1 * penalty - competition_image_score([box, spurious], [box])) < 1e-9
    stats = image_stats([box, spurious], [box], (20, 20))
    assert stats["n_pred"] == 2 and stats["n_gt"] == 1 and stats["matched"] == 1 and abs(stats["excess_penalty"] - 0.5) < 1e-9
    merged = image_stats([box | other], [box, other], (20, 20))
    assert merged["n_pred"] == 1 and merged["n_gt"] == 2

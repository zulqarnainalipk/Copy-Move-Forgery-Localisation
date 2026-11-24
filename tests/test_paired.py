import pandas as pd
import pytest

from forgery.paired import paired_difference


def frame(f1, official):
    return pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["forged", "forged", "forged", "authentic"],
                         "group": [0, 0, 1, 2], "is_forged": [True, True, True, False],
                         "pixel_f1": f1, "official_score": official})


def test_forged_f1_difference_uses_forged_images_only():
    a = frame([0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0])
    b = frame([0.2, 0.2, 0.2, 0.0], [0.2, 0.2, 0.2, 0.0])
    mean, low, high = paired_difference(a, b, "forged_pixel_f1", n_resamples=200)
    assert mean == pytest.approx(0.3)
    assert low == pytest.approx(0.3) and high == pytest.approx(0.3)


def test_official_difference_uses_all_images():
    a = frame([0.5] * 4, [1.0, 1.0, 1.0, 1.0])
    b = frame([0.5] * 4, [0.0, 0.0, 0.0, 0.0])
    assert paired_difference(a, b, "official_score", n_resamples=50)[0] == pytest.approx(1.0)


def test_mismatched_images_raise():
    with pytest.raises(ValueError):
        paired_difference(frame([0.5] * 4, [0.5] * 4), frame([0.5] * 4, [0.5] * 4).iloc[:3], n_resamples=10)

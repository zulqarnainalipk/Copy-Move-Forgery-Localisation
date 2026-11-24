import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imagehash")
pytest.importorskip("sklearn")

from forgery.audit import grouped_split


def test_no_group_crosses_splits():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(120), rng.integers(1, 4, 120))
    labels = np.where(rng.random(len(groups)) < 0.5, "forged", "authentic")
    frame = pd.DataFrame({"group": groups, "label": labels})
    frame["split"] = grouped_split(frame)
    assert set(frame.split) == {"train", "val", "test"}
    assert (frame.groupby("group").split.nunique() == 1).all()

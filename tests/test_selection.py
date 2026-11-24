import pandas as pd

from forgery import selection


def table():
    return pd.DataFrame({
        "mixed_pixel_f1": [0.60, 0.70, 0.65, 0.70],
        "official_score": [0.50, 0.40, 0.55, 0.45],
        "forged_pixel_f1": [0.20, 0.50, 0.30, 0.45],
        "authentic_tnr": [0.99, 0.80, 0.96, 0.92],
        "tag": ["a", "b", "c", "d"],
    })


def test_each_objective_picks_its_own_row():
    t = table()
    assert selection.choose(t, "mixed").tag in {"b", "d"}
    assert selection.choose(t, "official").tag == "c"
    assert selection.choose(t, "forged_tnr90").tag == "d"
    assert selection.choose(t, "forged_tnr95").tag == "c"


def test_stable_sort_keeps_first_of_ties():
    assert selection.choose(table(), "mixed", kind="stable").tag == "b"


def test_floor_that_nothing_reaches_falls_back_to_highest_tnr():
    t = table().assign(authentic_tnr=[0.2, 0.5, 0.3, 0.4])
    assert selection.choose(t, "forged_tnr95").tag == "b"


def test_first_objective_has_no_suffix():
    assert selection.suffix("mixed", "mixed") == "" and selection.suffix("official", "mixed") == "_official"

import numpy as np
import pytest

pytest.importorskip("cv2")

from forgery.robustness import CONDITIONS, corrupt


def test_every_corruption_keeps_shape_and_type():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (97, 131, 3), dtype=np.uint8)
    for kind, level in CONDITIONS:
        out = corrupt(image, kind, level, rng)
        assert out.shape == image.shape and out.dtype == np.uint8


def test_corruptions_change_the_image():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for kind, level in CONDITIONS[1:]:
        assert not np.array_equal(corrupt(image, kind, level, rng), image)

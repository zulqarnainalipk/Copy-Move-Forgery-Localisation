import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from forgery.baselines import SiftConfig, detect


def textured_image(seed=0, size=320):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (size, size), dtype=np.uint8)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    stretched = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([stretched] * 3, axis=2)


def test_duplicated_patch_is_found():
    image = textured_image()
    image[200:280, 200:280] = image[30:110, 40:120]
    instances, inliers = detect(image, SiftConfig())
    assert inliers >= 4
    assert instances and instances[0][70:100, 50:100].any()


def test_unique_texture_has_few_inliers():
    _, inliers = detect(textured_image(seed=1), SiftConfig())
    assert inliers < 10

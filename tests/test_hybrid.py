import numpy as np
import pytest

pytest.importorskip("cv2")

from forgery.hybrid import fuse, prune_to_grid


def blob(y, x, size=20, shape=(100, 100)):
    mask = np.zeros(shape, np.uint8)
    mask[y:y + size, x:x + size] = 1
    return mask


MATCHER = [blob(60, 60).astype(bool)]
SEG = blob(10, 10)


def total(instances):
    return sum(int(i.sum()) for i in instances)


def test_modes_respect_gates():
    assert fuse("seg_only", 0, [], SEG, 0.5, 10, 0.2)
    assert not fuse("seg_only", 0, [], SEG, 0.1, 10, 0.2)
    assert fuse("sift_only", 12, MATCHER, SEG, 0.0, 10, 0.2)
    assert not fuse("sift_only", 5, MATCHER, SEG, 0.9, 10, 0.2)


def test_priority_modes_choose_the_expected_source():
    both = fuse("sift_priority", 12, MATCHER, SEG, 0.5, 10, 0.2)
    assert both[0][65, 65] and not both[0][15, 15]
    both = fuse("seg_priority", 12, MATCHER, SEG, 0.5, 10, 0.2)
    assert both[0][15, 15] and not both[0][65, 65]


def test_union_merges_and_confirm_requires_both():
    merged = fuse("union", 12, MATCHER, SEG, 0.5, 10, 0.2)
    assert len(merged) == 1 and merged[0][15, 15] and merged[0][65, 65]
    assert not fuse("confirm", 12, MATCHER, SEG, 0.1, 10, 0.2)
    assert not fuse("confirm", 5, MATCHER, SEG, 0.5, 10, 0.2)
    assert fuse("confirm", 12, MATCHER, SEG, 0.5, 10, 0.2)


def test_refine_keeps_segmentation_near_matches():
    near = blob(55, 55, size=30).astype(bool)
    segmentation = blob(58, 58, size=10)
    refined = fuse("refine", 12, [near], segmentation, 0.5, 10, 0.2)
    assert total(refined) == int(segmentation.sum())
    fallback = fuse("refine", 12, MATCHER, SEG, 0.5, 10, 0.2)
    assert total(fallback) == total(MATCHER)


def test_grid_sizes():
    assert len(prune_to_grid("seg_only", (6, 10), (0.1, 0.2, 0.3))) == 3
    assert len(prune_to_grid("sift_only", (6, 10), (0.1, 0.2, 0.3))) == 2
    assert len(prune_to_grid("union", (6, 10), (0.1, 0.2, 0.3))) == 6

import numpy as np
import pytest

pytest.importorskip("cv2")

from forgery.data import PANEL_LAYOUTS, make_collage


def test_collage_keeps_masks_aligned_with_images():
    for layout in PANEL_LAYOUTS:
        count = layout[0] * layout[1]
        images = [np.full((120, 90, 3), 40 * (i + 1), np.uint8) for i in range(count)]
        masks = [np.zeros((120, 90), np.uint8) for _ in range(count)]
        masks[count - 1][:] = 1
        image, mask = make_collage(images, masks, layout, 96)
        assert image.shape == (96, 96, 3) and mask.shape == (96, 96)
        assert mask.any()
        assert set(np.unique(image[mask == 1])) == {40 * count}

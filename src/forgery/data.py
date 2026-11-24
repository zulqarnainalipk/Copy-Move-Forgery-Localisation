"""Datasets built from the split table written by the audit stage."""
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from forgery.config import SEED, Paths

Image.MAX_IMAGE_PIXELS = None


def load_split_table(path):
    table = pd.read_csv(path, dtype={"id": str})
    missing = table["group"].isna()
    table.loc[missing, "group"] = -1 - np.arange(int(missing.sum()))
    return table


def image_path(row, paths):
    folder = {"authentic": paths.authentic, "forged": paths.forged, "supp": paths.supplemental}[row.label]
    return folder / f"{row.id}.png"


def mask_instances(row, paths):
    """Instance masks of a forged image as a list of boolean arrays; empty for authentic images."""
    if row.label == "authentic":
        return []
    folder = paths.supplemental_masks if row.label == "supp" else paths.masks
    mask = np.load(folder / f"{row.id}.npy")
    if mask.ndim == 2:
        mask = mask[None]
    return [m > 0 for m in mask]


def union_mask(row, paths, shape):
    instances = mask_instances(row, paths)
    if not instances:
        return np.zeros(shape, np.uint8)
    return np.any(np.stack(instances), axis=0).astype(np.uint8)


def strong_augment(image, mask, rng):
    k = int(rng.integers(4))
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    if rng.random() < 0.9:
        alpha, beta = 1 + rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2) * 255
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        sigma = rng.uniform(0.1, 2.0)
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if rng.random() < 0.15:
        image = np.repeat(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    if rng.random() < 0.5:
        quality = int(rng.integers(30, 96))
        ok, encoded = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


PANEL_LAYOUTS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))


def make_collage(images, masks, layout, size):
    """Tile images and their masks into a rows x cols figure resized to size x size."""
    rows, cols = layout
    cell_h, cell_w = size // rows, size // cols
    tiles = [cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA) for image in images]
    tile_masks = [cv2.resize(mask, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST) for mask in masks]
    image = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0)
    mask = np.concatenate([np.concatenate(tile_masks[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas_mask = np.zeros((size, size), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    canvas_mask[:mask.shape[0], :mask.shape[1]] = mask
    return canvas, canvas_mask


class ForgeryDataset(Dataset):
    """augment: none, strong (flips, rotations, photometric, JPEG) or panel (strong plus multi-panel collages)."""

    def __init__(self, table, paths, img_size=518, augment="none", seed=SEED, panel_probability=0.6):
        self.rows = list(table.itertuples(index=False))
        self.paths, self.img_size, self.augment = paths, img_size, augment
        self.seed, self.panel_probability, self._rng = seed, panel_probability, None

    def __len__(self):
        return len(self.rows)

    @property
    def rng(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(self.seed + (info.id + 1 if info else 0))
        return self._rng

    def _load(self, row):
        image = np.asarray(Image.open(image_path(row, self.paths)).convert("RGB"))
        return image, union_mask(row, self.paths, image.shape[:2])

    def __getitem__(self, index):
        row = self.rows[index]
        image, mask = self._load(row)
        if self.augment == "panel" and self.rng.random() < self.panel_probability:
            layout = PANEL_LAYOUTS[int(self.rng.integers(len(PANEL_LAYOUTS)))]
            count = layout[0] * layout[1]
            others = [self._load(self.rows[int(i)]) for i in self.rng.integers(len(self.rows), size=count - 1)]
            slot = int(self.rng.integers(count))
            parts = others[:slot] + [(image, mask)] + others[slot:]
            image, mask = make_collage([p[0] for p in parts], [p[1] for p in parts], layout, self.img_size)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment in ("strong", "panel"):
            image, mask = strong_augment(image, mask, self.rng)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.from_numpy(mask[None].astype(np.float32))


def make_loader(table, paths, batch_size, shuffle, augment="none", img_size=518, workers=2, seed=SEED):
    dataset = ForgeryDataset(table, paths, img_size, augment, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

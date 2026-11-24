"""Probability maps with test-time augmentation, cached per split."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from forgery.config import Paths
from forgery.data import image_path, load_split_table
from forgery.model import ForgerySegmenter

VIEWS = {
    "none": [(lambda x: x, lambda x: x)],
    "flip3": [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [3])),
        (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),
    ],
    "rot4": [(lambda x, k=k: torch.rot90(x, k, [2, 3]), lambda x, k=k: torch.rot90(x, -k, [2, 3])) for k in range(4)],
}


@torch.no_grad()
def probability_map(model, image, device, tta="flip3", img_size=518):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    maps = []
    for forward, backward in VIEWS[tta]:
        maps.append(backward(torch.sigmoid(model(forward(x)))))
    return torch.stack(maps).mean(0)[0, 0].cpu().numpy()


WINDOW_STARTS = (0.0, 0.25, 0.5)


def multiscale_probability_map(model, image, device, tta="none", img_size=518):
    """Global map at twice the model resolution, merged by maximum with the mean of nine half-size windows."""
    canvas = 2 * img_size
    height, width = image.shape[:2]
    merged = cv2.resize(probability_map(model, image, device, tta, img_size), (canvas, canvas), interpolation=cv2.INTER_LINEAR)
    total = np.zeros((canvas, canvas), np.float32)
    count = np.zeros((canvas, canvas), np.float32)
    for top in WINDOW_STARTS:
        for left in WINDOW_STARTS:
            y0, x0 = int(top * height), int(left * width)
            window = image[y0:y0 + height // 2, x0:x0 + width // 2]
            if window.shape[0] < 16 or window.shape[1] < 16:
                continue
            tile = probability_map(model, window, device, tta, img_size)
            cy, cx = int(top * canvas), int(left * canvas)
            total[cy:cy + img_size, cx:cx + img_size] += tile
            count[cy:cy + img_size, cx:cx + img_size] += 1
    return np.maximum(merged, total / np.maximum(count, 1))


def cache_split(model, table, paths, device, tta, img_size, destination, multiscale=False):
    maps = {}
    for row in tqdm(list(table.itertuples(index=False)), desc=f"predict {destination.stem}"):
        image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
        if multiscale:
            maps[row.uid] = multiscale_probability_map(model, image, device, tta, img_size).astype(np.float16)
        else:
            maps[row.uid] = probability_map(model, image, device, tta, img_size).astype(np.float16)
    (np.savez_compressed if multiscale else np.savez)(destination, **maps)
    print(f"[predict] {len(maps)} maps written to {destination}")


def predict(args, paths=None):
    """Cache probability maps of a trained run for the chosen splits."""
    paths = paths or Paths.from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    model = ForgerySegmenter(paths.dinov2_dir, args.head, args.encoder_input, args.img_size)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()
    for split in args.splits_to_predict:
        part = table[table.split == split]
        if args.max_images:
            part = part.head(args.max_images)
        cache_split(model, part, paths, device, args.tta, args.img_size, run_dir / f"probs_{split}.npz", args.multiscale)

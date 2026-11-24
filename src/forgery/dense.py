"""Zero-shot dense baseline: copy-move evidence from self-similarity of frozen DINOv2 patch features."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

from forgery.config import Paths
from forgery.data import image_path, load_split_table
from forgery.model import MEAN, STD

MIN_TOKEN_DISTANCE = 6
SIMILARITY_FLOOR = 0.5


@torch.no_grad()
def similarity_map(encoder, image, device, img_size=518, min_distance=MIN_TOKEN_DISTANCE):
    """Per token: the highest cosine similarity to any token at least min_distance tokens away, rescaled to [0, 1]."""
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    mean, std = torch.tensor(MEAN, device=device).view(1, 3, 1, 1), torch.tensor(STD, device=device).view(1, 3, 1, 1)
    tokens = encoder(pixel_values=(x - mean) / std).last_hidden_state[0, 1:]
    side = int(tokens.shape[0] ** 0.5)
    tokens = F.normalize(tokens, dim=1)
    similarity = tokens @ tokens.T
    grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1).reshape(-1, 2).float().to(device)
    near = torch.cdist(grid, grid) < min_distance
    similarity = similarity.masked_fill(near, -1.0)
    best = similarity.max(dim=1).values.reshape(1, 1, side, side)
    best = ((best - SIMILARITY_FLOOR) / (1 - SIMILARITY_FLOOR)).clamp(0, 1)
    return F.interpolate(best, size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def cache_dense(args, paths=None):
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(paths.dinov2_dir).to(device).eval()
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    for split in ("val", "test", "supp"):
        part = table[table.split == split]
        if args.max_images:
            part = part.head(args.max_images)
        maps = {}
        for row in tqdm(list(part.itertuples(index=False)), desc=f"dense {split}"):
            image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
            maps[row.uid] = similarity_map(encoder, image, device, args.img_size).astype(np.float16)
        np.savez(run_dir / f"probs_{split}.npz", **maps)
        print(f"[dense] {len(maps)} maps written for {split}")

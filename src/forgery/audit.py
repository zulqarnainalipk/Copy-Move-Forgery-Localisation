"""Data audit, near-duplicate grouping and leakage-safe splits."""
import json
from multiprocessing import Pool

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm

from forgery.config import SEED, Paths

Image.MAX_IMAGE_PIXELS = None
HASH_SIZE = 16
HAMMING_THRESHOLD = 20
SIZE_BINS = [0, 0.005, 0.02, 1.0]
SIZE_LABELS = ["<0.5%", "0.5-2%", ">2%"]
N_FOLDS, N_TRAIN_FOLDS, N_VAL_FOLDS = 20, 14, 3


def describe_image(job):
    path, label, mask_dir = job
    stem = path.stem
    image = Image.open(path)
    width, height = image.size
    rgb = image.convert("RGB")
    small = np.asarray(rgb.resize((64, 64)), np.float32)
    digest = imagehash.phash(rgb, hash_size=HASH_SIZE)
    row = {
        "id": stem, "uid": f"{label}_{stem}", "label": label, "H": height, "W": width,
        "mean_intensity": float(small.mean()),
        "saturation": float((small.max(2) - small.min(2)).mean()),
        "hash": "".join("1" if bit else "0" for bit in digest.hash.flatten()),
    }
    if label == "authentic":
        return row
    mask_path = mask_dir / f"{stem}.npy"
    if not mask_path.exists():
        row.update(n_instances=0, area_frac=np.nan, touches_border=np.nan, mask_shape="missing")
        return row
    mask = np.load(mask_path)
    union = (mask.max(0) if mask.ndim == 3 else mask) > 0
    margin_h, margin_w = max(1, int(0.05 * union.shape[0])), max(1, int(0.05 * union.shape[1]))
    band = np.zeros_like(union)
    band[:margin_h] = band[-margin_h:] = True
    band[:, :margin_w] = band[:, -margin_w:] = True
    row.update(
        mask_shape="x".join(map(str, mask.shape)),
        n_instances=int(mask.shape[0]) if mask.ndim == 3 else 1,
        area_frac=float(union.mean()),
        touches_border=bool((union & band).any()),
    )
    return row


def appearance_tag(row):
    if row["saturation"] < 8:
        return "grayscale-dark" if row["mean_intensity"] < 90 else "grayscale-light"
    return "colour"


def collect_metadata(paths, workers=4):
    jobs = [(p, "authentic", paths.masks) for p in sorted(paths.authentic.glob("*.png"))]
    jobs += [(p, "forged", paths.masks) for p in sorted(paths.forged.glob("*.png"))]
    jobs += [(p, "supp", paths.supplemental_masks) for p in sorted(paths.supplemental.glob("*.png"))]
    print(f"[audit] {len(jobs)} images: "
          f"{sum(j[1] == 'authentic' for j in jobs)} authentic, "
          f"{sum(j[1] == 'forged' for j in jobs)} forged, "
          f"{sum(j[1] == 'supp' for j in jobs)} supplemental")
    with Pool(workers) as pool:
        rows = list(tqdm(pool.imap(describe_image, jobs, chunksize=16), total=len(jobs), desc="describe images"))
    frame = pd.DataFrame(rows)
    frame["appearance"] = frame.apply(appearance_tag, axis=1)
    return frame


def group_near_duplicates(frame):
    bits = np.array([[int(c) for c in h] for h in frame["hash"]], dtype=np.uint8)
    parent = list(range(len(frame)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = 0
    for i in tqdm(range(len(frame)), desc="near-duplicate search"):
        distance = (bits[i + 1:] != bits[i]).sum(1)
        for offset in np.nonzero(distance <= HAMMING_THRESHOLD)[0]:
            a, b = find(i), find(i + 1 + offset)
            if a != b:
                parent[a] = b
            pairs += 1
    groups = [find(i) for i in range(len(frame))]
    print(f"[audit] {pairs} near-duplicate pairs (<= {HAMMING_THRESHOLD}/256 bits)")
    return groups


def random_split_validation_ids(frame):
    """Conventional random image-level 80/20 split (sorted file names, seed 42), audited only to quantify leakage."""
    val = set()
    for label in ("authentic", "forged"):
        files = sorted(f"{i}.png" for i in frame.loc[frame.label == label, "id"])
        _, held_out = train_test_split(files, test_size=0.2, random_state=SEED)
        val.update(f"{label}_{name[:-4]}" for name in held_out)
    return val


def grouped_split(frame):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(frame), int)
    for k, (_, held_out) in enumerate(splitter.split(frame, frame.label, frame.group)):
        fold[held_out] = k
    return np.where(fold < N_TRAIN_FOLDS, "train", np.where(fold < N_TRAIN_FOLDS + N_VAL_FOLDS, "val", "test"))


def size_bin(frame):
    binned = pd.cut(frame.area_frac, SIZE_BINS, labels=SIZE_LABELS, include_lowest=True).astype(object)
    return binned.where(frame.label != "authentic", "authentic")


def run(paths=None):
    """Audit the dataset, group near-duplicates and write splits.csv and audit.json."""
    paths = paths or Paths.from_env()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(paths)
    train_part = metadata[metadata.label != "supp"].reset_index(drop=True)
    supp_part = metadata[metadata.label == "supp"].copy()

    train_part["group"] = group_near_duplicates(train_part)
    group_sizes = train_part.groupby("group").size()
    print(f"[audit] {len(group_sizes)} groups, {int((group_sizes > 1).sum())} with more than one image, "
          f"largest {int(group_sizes.max())}")

    random_val = random_split_validation_ids(train_part)
    train_part["random_split"] = np.where(train_part.uid.isin(random_val), "val", "train")
    old_train_groups = set(train_part.loc[train_part.random_split == "train", "group"])
    leaked = train_part[(train_part.random_split == "val") & train_part.group.isin(old_train_groups)]
    print(f"[audit] {len(leaked)} of {len(random_val)} random-split validation images have a "
          f"near-duplicate in the random-split training set")

    train_part["split"] = grouped_split(train_part)
    train_part["size_bin"] = size_bin(train_part)
    supp_part["split"] = "supp"
    supp_part["size_bin"] = size_bin(supp_part)

    combined = pd.concat([train_part, supp_part], ignore_index=True).drop(columns=["hash"])
    combined.to_csv(paths.output_dir / "splits.csv", index=False)

    forged = train_part[train_part.label == "forged"]
    summary = {
        "counts": {"authentic": int((train_part.label == "authentic").sum()),
                   "forged": int((train_part.label == "forged").sum()), "supplemental": len(supp_part)},
        "hamming_threshold_bits": HAMMING_THRESHOLD,
        "groups": {"total": int(len(group_sizes)), "multi_image": int((group_sizes > 1).sum()),
                   "largest": int(group_sizes.max())},
        "random_split_validation_with_near_duplicate_in_train": {"leaked": int(len(leaked)), "total": len(random_val)},
        "grouped_split": train_part.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "forged_area_fraction_quantiles": forged.area_frac.quantile([.05, .25, .5, .75, .95]).to_dict(),
        "forged_instances_per_image": forged.n_instances.value_counts().sort_index().to_dict(),
        "forged_touching_border": float(forged.touches_border.mean()),
        "supplemental_area_fraction_quantiles": supp_part.area_frac.quantile([.05, .25, .5, .75, .95]).to_dict(),
        "supplemental_instances_per_image": supp_part.n_instances.value_counts().sort_index().to_dict(),
        "image_height_quantiles": {"train": train_part.H.quantile([.05, .5, .95]).to_dict(),
                                   "supplemental": supp_part.H.quantile([.05, .5, .95]).to_dict()},
        "appearance_counts": train_part.appearance.value_counts().to_dict(),
        "forged_size_bins_by_split": forged.groupby(["split", "size_bin"]).size().unstack(fill_value=0).to_dict(),
        "mask_shape_examples": forged.mask_shape.head(5).tolist(),
    }
    with open(paths.output_dir / "audit.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return combined

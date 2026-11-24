"""Sensitivity of the near-duplicate grouping to the Hamming threshold, residual leakage across the grouped split,
and the list of near-duplicate pairs."""
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from forgery.audit import collect_metadata
from forgery.config import Paths
from forgery.data import image_path

THRESHOLDS = (8, 12, 16, 20, 24, 28, 32)
RESIDUAL_THRESHOLDS = (20, 24, 28, 32, 40)
PAIR_LIMIT = 32
EXAMPLE_DISTANCES = (0, 4, 8, 12, 16, 20, 24, 28, 32)


def hamming_matrix(hashes):
    """Pairwise Hamming distances between binary perceptual hashes."""
    bits = np.array([[int(c) for c in h] for h in hashes], dtype=np.float32)
    agree = bits @ bits.T + (1 - bits) @ (1 - bits).T
    return (bits.shape[1] - agree).astype(np.int16)


def cluster_labels(distance, threshold):
    """Connected-component group labels at a Hamming threshold."""
    adjacency = csr_matrix((distance <= threshold) & ~np.eye(len(distance), dtype=bool))
    return connected_components(adjacency, directed=False)[1]


def leakage_table(distance, frame):
    """Share of held-out images of the random split with a training neighbour, per threshold."""
    held_out = (frame.random_split == "val").to_numpy()
    rows = []
    for threshold in THRESHOLDS:
        labels = cluster_labels(distance, threshold)
        sizes = np.bincount(labels)
        train_groups = set(labels[~held_out])
        leaked = int(sum(label in train_groups for label in labels[held_out]))
        rows.append({"hamming_threshold": threshold, "clusters": int(len(sizes)), "multi_image_clusters": int((sizes > 1).sum()),
                     "largest_cluster": int(sizes.max()), "images_in_multi_image_clusters": int(sizes[sizes > 1].sum()),
                     "random_split_held_out": int(held_out.sum()), "random_split_leaked": leaked,
                     "random_split_leaked_share": leaked / int(held_out.sum())})
    return pd.DataFrame(rows)


def residual_table(distance, frame):
    """Share of grouped-split validation and test images with a training neighbour, per threshold."""
    rows = []
    train = np.nonzero((frame.split == "train").to_numpy())[0]
    for split in ("val", "test"):
        part = np.nonzero((frame.split == split).to_numpy())[0]
        nearest = distance[np.ix_(part, train)].min(axis=1)
        for threshold in RESIDUAL_THRESHOLDS:
            count = int((nearest <= threshold).sum())
            rows.append({"grouped_split_part": split, "hamming_threshold": threshold, "images": int(len(part)),
                         "with_train_neighbour": count, "share": count / len(part),
                         "median_nearest_distance": float(np.median(nearest))})
    return pd.DataFrame(rows)


def pair_table(distance, frame):
    """Near-duplicate image pairs within PAIR_LIMIT bits, with their labels and splits."""
    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)
    a, b = np.nonzero((distance <= PAIR_LIMIT) & upper)
    group = cluster_labels(distance, 20)
    return pd.DataFrame({
        "uid_a": frame.uid.to_numpy()[a], "uid_b": frame.uid.to_numpy()[b],
        "label_a": frame.label.to_numpy()[a], "label_b": frame.label.to_numpy()[b],
        "hamming_distance": distance[a, b], "same_group_at_20": group[a] == group[b],
        "split_a": frame.split.to_numpy()[a], "split_b": frame.split.to_numpy()[b],
        "random_split_a": frame.random_split.to_numpy()[a], "random_split_b": frame.random_split.to_numpy()[b],
    }).sort_values(["hamming_distance", "uid_a"]).reset_index(drop=True)


def shrink(image, side=300):
    """Resize an image so that its longer side equals side."""
    scale = side / max(image.shape[:2])
    return cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def write_examples(pairs, frame, paths, out_dir):
    """Save example near-duplicate pairs as images."""
    by_uid = frame.set_index("uid")
    chosen = {}
    for target in EXAMPLE_DISTANCES:
        candidates = pairs[(pairs.hamming_distance == target) & (pairs.label_a == pairs.label_b)]
        if not len(candidates):
            candidates = pairs[pairs.hamming_distance == target]
        if len(candidates):
            row = candidates.iloc[len(candidates) // 2]
            chosen[int(target)] = (row.uid_a, row.uid_b)
    info = {}
    for distance, (a, b) in chosen.items():
        for tag, uid in (("a", a), ("b", b)):
            row = next(by_uid.loc[[uid]].reset_index().itertuples(index=False))
            image = np.asarray(Image.open(image_path(row, paths)).convert("RGB"))
            Image.fromarray(shrink(image)).save(out_dir / f"example_d{distance:02d}_{tag}.png")
        info[str(distance)] = {"a": a, "b": b}
    (out_dir / "examples.json").write_text(json.dumps(info, indent=2))
    return info


def run(args, paths=None):
    """Run the grouping sensitivity analysis and write its tables."""
    paths = paths or Paths.from_env()
    out_dir = paths.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(args.splits or paths.output_dir / "splits.csv", dtype={"id": str})
    metadata = collect_metadata(paths, workers=args.workers)
    metadata = metadata[metadata.label != "supp"].reset_index(drop=True)
    frame = metadata[["id", "uid", "label", "hash"]].merge(splits[["uid", "split", random_column(splits)]].rename(
        columns={random_column(splits): "random_split"}), on="uid", how="left")
    print(f"[sensitivity] {len(frame)} images, computing {len(frame)}x{len(frame)} Hamming distances", flush=True)
    distance = hamming_matrix(frame["hash"])
    leakage = leakage_table(distance, frame)
    residual = residual_table(distance, frame)
    pairs = pair_table(distance, frame)
    leakage.to_csv(out_dir / "leakage_by_threshold.csv", index=False)
    residual.to_csv(out_dir / "residual_leakage_grouped_split.csv", index=False)
    pairs.to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    print(leakage.round(4).to_string(index=False))
    print(residual.round(4).to_string(index=False))
    print(f"[sensitivity] {len(pairs)} pairs at distance <= {PAIR_LIMIT}; {int(pairs.same_group_at_20.sum())} within a group at 20")
    write_examples(pairs, metadata, paths, out_dir)
    return leakage, residual, pairs


def random_column(splits):
    """Name of the random-split column in a splits table."""
    return "random_split" if "random_split" in splits.columns else "old_split"

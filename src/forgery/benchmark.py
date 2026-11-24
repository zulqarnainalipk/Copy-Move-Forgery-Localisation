"""Runtime and memory of the segmenter, the keypoint matcher and the fused pipeline."""
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from forgery.baselines import SiftConfig, detect
from forgery.config import Paths
from forgery.data import image_path, load_split_table
from forgery.model import ForgerySegmenter, count_parameters
from forgery.predict import probability_map


def timed(function, repeats):
    """Median and 90th percentile wall time of a function over repeated calls."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)), float(np.percentile(durations, 90))


def benchmark(args, paths=None):
    """Measure inference time and peak memory of each component and write benchmark.json."""
    paths = paths or Paths.from_env()
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    table = load_split_table(args.splits or paths.output_dir / "splits.csv")
    rows = list(table[table.split == "test"].itertuples(index=False))[: args.images]
    images = [np.asarray(Image.open(image_path(row, paths)).convert("RGB")) for row in rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "images": len(images)}
    for encoder_input in ("crop224", "full"):
        model = ForgerySegmenter(paths.dinov2_dir, "conv3", encoder_input).to(device).eval()
        report["parameters"] = count_parameters(model)
        for tta in ("none", "flip3", "rot4"):
            probability_map(model, images[0], device, tta)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            durations = []
            for image in tqdm(images, desc=f"{encoder_input} {tta}"):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                probability_map(model, image, device, tta)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
            report[f"segmenter_{encoder_input}_{tta}"] = {"median_s": float(np.median(durations)),
                                                          "p90_s": float(np.percentile(durations, 90)), "peak_memory_mb": peak}
            print(f"[benchmark] segmenter {encoder_input} {tta}: median {np.median(durations) * 1000:.0f} ms, "
                  f"peak {peak:.0f} MB", flush=True)
        del model
    sift = [timed(lambda image=image: detect(image, SiftConfig()), 1)[0] for image in tqdm(images, desc="sift")]
    report["sift"] = {"median_s": float(np.median(sift)), "p90_s": float(np.percentile(sift, 90))}
    print(f"[benchmark] sift median {np.median(sift) * 1000:.0f} ms, p90 {np.percentile(sift, 90) * 1000:.0f} ms")
    with open(run_dir / "benchmark.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return report

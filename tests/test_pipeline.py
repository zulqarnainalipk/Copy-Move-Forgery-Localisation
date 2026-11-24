import types

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from forgery.config import Paths
from forgery.evaluate import evaluate_run
from forgery.predict import predict
from forgery.train import train


@pytest.fixture()
def workspace(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    for folder in ("train_images/authentic", "train_images/forged", "train_masks", "supplemental_images", "supplemental_masks"):
        (data / folder).mkdir(parents=True)
    rows = []
    for i in range(24):
        label = "forged" if i % 2 else "authentic"
        folder = "forged" if label == "forged" else "authentic"
        image = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(data / "train_images" / folder / f"{i}.png")
        if label == "forged":
            mask = np.zeros((2, 80, 100), np.uint8)
            mask[0, 10:30, 10:30] = 1
            mask[1, 40:60, 60:80] = 1
            np.save(data / "train_masks" / f"{i}.npy", mask)
        split = "train" if i < 16 else ("val" if i < 20 else "test")
        rows.append({"id": str(i), "uid": f"{label}_{i}", "label": label, "H": 80, "W": 100, "group": i, "split": split,
                     "appearance": "colour", "size_bin": "authentic" if label == "authentic" else "0.5-2%",
                     "touches_border": False if label == "forged" else np.nan})
    for i in range(100, 102):
        Image.fromarray(rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)).save(data / "supplemental_images" / f"{i}.png")
        np.save(data / "supplemental_masks" / f"{i}.npy", np.pad(np.ones((1, 20, 20), np.uint8), ((0, 0), (5, 35), (5, 35))))
        rows.append({"id": str(i), "uid": f"supp_{i}", "label": "supp", "H": 60, "W": 60, "group": i, "split": "supp",
                     "appearance": "colour", "size_bin": "0.5-2%", "touches_border": False})
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(rows).to_csv(out / "splits.csv", index=False)
    encoder = tmp_path / "encoder"
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                       intermediate_size=64, image_size=224, patch_size=14)
    transformers.Dinov2Model(config).save_pretrained(encoder)
    return Paths(data, out, encoder)


@pytest.mark.slow
@pytest.mark.parametrize("encoder_input", ["crop224", "full"])
def test_train_predict_evaluate(workspace, encoder_input):
    args = types.SimpleNamespace(
        run_name=f"run_{encoder_input}", splits=None, encoder_input=encoder_input, head="conv3", loss="bce_dice",
        augment="strong", img_size=126, batch_size=4, accumulation=1, epochs1=1, epochs2=1, patience1=1,
        patience2=1, lr_head=1e-3, lr_encoder=1e-4, workers=0, max_images=0, amp=False, seed=0)
    train(args, workspace)
    args.splits_to_predict, args.tta, args.checkpoint, args.multiscale = ["val", "test", "supp"], "flip3", None, False
    predict(args, workspace)
    args.multiscale, args.tta = True, "none"
    predict(args, workspace)
    args.bootstrap, args.cache_dir, args.k_sigma_grid = 5, None, [0.3, 1.0]
    report = evaluate_run(args, workspace)
    assert {"val", "test", "supp"} <= set(report)
    assert 0.0 <= report["test"]["mixed_pixel_f1"] <= 1.0

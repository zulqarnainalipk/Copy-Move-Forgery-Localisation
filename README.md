# Copy-move forgery localisation in biomedical research images

Code and results for a grouped, baseline-controlled study of copy-move forgery localisation. It compares a fine-tuned DINOv2 segmenter, classical and learned keypoint matchers, and a hybrid of the two on the Recod.ai/LUC Scientific Image Forgery Detection data.

## Main findings

Test split (770 images: 418 forged, 352 authentic; grouped so that near-duplicates never cross splits). Forged F1 is the mean pixel F1 over forged images; official is the competition's instance-aware score. Brackets are 95% group-bootstrap intervals. All thresholds and gates were chosen on the validation split only.

| Method | Forged F1 | Official score | Authentic TNR | Image AUROC |
| --- | --- | --- | --- | --- |
| SIFT self-matching | 0.209 [0.183, 0.236] | 0.500 | 0.974 | 0.780 |
| DINOv2, crop 224, no augmentation | 0.176 [0.150, 0.200] | 0.438 | 0.920 | 0.722 |
| DINOv2, full 518 px input, strong augmentation | 0.462 [0.428, 0.498] | 0.517 | 0.906 | 0.876 |
| Hybrid `refine` (segmenter mask near keypoint matches) | 0.499 [0.465, 0.534] | 0.521 | 0.881 | 0.866* |
| Hybrid `union` (segmenter and matcher masks merged) | 0.413 [0.383, 0.443] | 0.601 [0.577, 0.623] | 0.886 | 0.866* |

\* Every hybrid rule shares one fused image-level score (the larger of the segmenter probability and a monotone function of the keypoint inlier count), so the AUROC is the same for all rules and does not rank them.

Sources: `results/c3_sift`, `results/c4_eval_g1`, `results/c4_evalks_g3_full_aug` (k_sigma chosen on validation) and `results/c6_hybrid_g3_ks`; the same rows appear in `results/tables/main_test.md`. The segmenter row is training seed 42; seeds 1 and 2 of the same recipe give test forged F1 0.427 and 0.442 (`results/c4_evalks_g3_seed1`, `results/c4_evalks_g3_seed2`), so expect a few hundredths of spread between seeds.

Full tables, including image-level and pixel-level true and false positives and negatives, ablations, robustness and runtime, are in `results/tables/`. Every run is listed in `docs/experiment_log.md`.

## Layout

| Path | Contents |
| --- | --- |
| `src/forgery/` | Package: audit and splits, metrics, data, models, losses, training, prediction, post-processing, evaluation, baselines, hybrid fusion, ablations, robustness, benchmarking, tables, inference |
| `scripts/` | One command-line entry point per stage |
| `tests/` | Unit and end-to-end tests |
| `results/` | Reports, per-image statistics, logs and tables of each run; `results/README.md` indexes every folder |
| `docs/` | Experiment log, evaluation protocol, reproduction guide, model card |
| `.github/` | Continuous-integration workflow that runs the fast tests on CPU |
| `CITATION.cff` | Citation metadata |

## Setup

```sh
pip install -r requirements.txt
pip install -e .
```

```sh
export FORGERY_DATA_DIR=/path/to/dataset      # train_images/{authentic,forged}, train_masks, supplemental_images, supplemental_masks
export FORGERY_OUTPUT_DIR=outputs
export FORGERY_DINOV2_DIR=/path/to/dinov2-base
```

SuperPoint features need `pip install git+https://github.com/cvg/LightGlue.git`.

## Data layout

The images and masks are not included. `FORGERY_DATA_DIR` must contain:

```text
FORGERY_DATA_DIR/
  train_images/
    authentic/<id>.png
    forged/<id>.png
  train_masks/<id>.npy             # one array per forged image: instances x height x width, values 0 or 1
  supplemental_images/<id>.png
  supplemental_masks/<id>.npy
```

Authentic images have no mask. `scripts/audit.py` reads this folder and writes `splits.csv` to `FORGERY_OUTPUT_DIR`.

## Quickstart

Localise copy-move regions in one image with a trained checkpoint (see Weights):

```sh
python scripts/infer.py figure.png --checkpoint outputs/g3_full_aug --dinov2-dir /path/to/dinov2-base --output-dir predictions
python scripts/infer.py figure.png --checkpoint outputs/g3_full_aug --mode refine --min-inliers 15 --min-probability 0.10
```

`--mode` selects the segmenter alone (`seg_only`, the default) or a fusion rule with the SIFT matcher (`refine`, `union`, `confirm`, `seg_priority`, `sift_priority`, `sift_only`). For each image the script writes `<name>_mask.png` and `<name>_overlay.png`, and `summary.json` lists the decision, instance count, segmenter probability and keypoint inliers. `--encoder-input`, `--head`, `--img-size` and `--tta` must match the training run (defaults: `full`, `conv3`, 518, `flip3`).

## Weights

Trained weights are not included in this repository. Retrain them with the commands in `docs/reproduce.md` (final recipe: `scripts/train.py --run-name g3_full_aug --encoder-input full --augment strong`), starting from the pretrained DINOv2-base encoder given by `FORGERY_DINOV2_DIR`.

## Stages

| Script | Purpose |
| --- | --- |
| `scripts/audit.py` | Near-duplicate grouping, leakage audit and the grouped train, validation and test split (`splits.csv`) |
| `scripts/train.py` | Two-stage training (decoder warmup, then joint fine-tuning) with options for encoder input, head, loss, augmentation and seed |
| `scripts/predict.py` | Probability maps per split with optional flip or rotation averaging and multi-scale tiling |
| `scripts/evaluate.py` | Gate and threshold selection on validation, then metrics with bootstrap intervals on validation, test and supplemental images |
| `scripts/baseline_sift.py`, `scripts/baseline_superpoint.py`, `scripts/baseline_dense.py` | Keypoint and zero-shot dense baselines |
| `scripts/hybrid.py` | Seven fusion rules between segmenter and matcher, selected on validation |
| `scripts/ablate.py` | Post-processing ablations and stratified analysis on cached maps |
| `scripts/robustness.py` | Accuracy under JPEG, blur, noise and rescaling |
| `scripts/benchmark.py` | Inference time and peak memory |
| `scripts/paired.py`, `scripts/sensitivity.py`, `scripts/calibration.py`, `scripts/cases.py` | Paired bootstrap differences, grouping sensitivity and near-duplicate pairs, score calibration, case renderer |
| `scripts/make_tables.py` | Comparison, confusion, ablation, robustness and runtime tables |
| `scripts/infer.py` | Localise forgeries in new images with a trained checkpoint |

`docs/reproduce.md` gives the order and the exact arguments used for each reported run.

## Tests

```sh
pytest tests                    # full suite, including a short end-to-end training run on synthetic data
pytest -m "not slow" tests      # fast tests only, as run in continuous integration
```

## Citation

If you use this code or the results, please cite:

> Zulqarnain Ali and Muhammad Ibrahim Qasmi. Deep segmentation and keypoint matching for copy-move forgery in biomedical images: a grouped, baseline-controlled comparison.

Machine-readable metadata is in `CITATION.cff`.

## Authors and contact

- **Zulqarnain Ali** (corresponding author), Department of Data Science, The Islamia University of Bahawalpur, Pakistan. ORCID [0009-0009-0281-882X](https://orcid.org/0009-0009-0281-882X). Email zulqar445ali@gmail.com.
- **Muhammad Ibrahim Qasmi**, Department of Information Technology, Bahauddin Zakariya University, Multan, Pakistan. ORCID [0009-0006-5392-0873](https://orcid.org/0009-0006-5392-0873).

Repository: https://github.com/zulqarnainalipk/Copy-Move-Forgery-Localisation

## Licence and data

Code is released under the MIT licence. The images and masks are distributed by the competition organisers under their own terms and are not included.

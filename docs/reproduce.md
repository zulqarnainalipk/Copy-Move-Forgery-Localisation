# Reproducing the reported runs

All commands assume the environment variables from the README. Every script accepts `--help`. Each command writes to `FORGERY_OUTPUT_DIR/<run-name>`; the stored copies in `results/` were then filed under the folder names listed in the mapping table at the end of this guide, so a run name and its results folder are not always the same. The commands below were taken from the `[run]` lines of the stored `run.log` files where those exist; the stored runs also passed `--splits outputs/splits.csv`, which is the default location and is omitted here.

## 1. Audit and split

```sh
python scripts/audit.py
```

Writes `splits.csv` and `audit.json` to `FORGERY_OUTPUT_DIR`. Every later stage reads `splits.csv`.

## 2. Baselines

```sh
python scripts/baseline_sift.py --run-name c3_sift
python scripts/baseline_superpoint.py --run-name baseline_superpoint
python scripts/baseline_dense.py --run-name baseline_dense
python scripts/evaluate.py --run-name baseline_dense
```

## 3. Segmenters

Main configuration (DINOv2-base, full 518 px input, strong augmentation, BCE, seed 42):

```sh
python scripts/train.py --run-name g3_full_aug --encoder-input full --augment strong --workers 4
python scripts/predict.py --run-name g3_full_aug --encoder-input full
python scripts/evaluate.py --run-name g3_full_aug --cache-dir outputs/g3_full_aug
python scripts/evaluate.py --run-name g3_full_aug_ks --cache-dir outputs/g3_full_aug --k-sigma-grid 0.3 0.6 1.0 1.5 2.0
```

The first evaluation uses the fixed default k_sigma (0.3); the second chooses k_sigma on validation and gives the headline numbers.

Variants reported in the study differ from this configuration in one setting each:

| Run | Change |
| --- | --- |
| `g1_base` | crop-224 encoder input, no augmentation |
| `g2a_full` | full input, no augmentation |
| `g2d_aug` | crop-224 input, strong augmentation |
| `g2b_dice`, `g2c_focal` | `--loss bce_dice`, `--loss focal` (crop-224, no augmentation) |
| `g2e_linear` | `--head linear` (crop-224, no augmentation) |
| `g6_large` | DINOv2-large, `--batch-size 1 --accumulation 16 --epochs1 6 --epochs2 8` |
| `g7_panel` | `--augment panel` |
| `g3_seed1`, `g3_seed2` | `--seed 1`, `--seed 2` |

Training and prediction commands recorded in the logs:

```sh
python scripts/train.py --run-name g1_base --workers 4
python scripts/predict.py --run-name g1_base
python scripts/train.py --run-name g2b_dice --workers 4 --loss bce_dice
python scripts/predict.py --run-name g2b_dice
python scripts/train.py --run-name g2c_focal --workers 4 --loss focal
python scripts/predict.py --run-name g2c_focal
python scripts/predict.py --run-name g2a_full --encoder-input full
python scripts/predict.py --run-name g2d_aug
```

Evaluation of every segmenter (gates chosen on validation; `_ks` runs also choose k_sigma on validation):

```sh
python scripts/evaluate.py --run-name g1_base --cache-dir outputs/g1_base
python scripts/evaluate.py --run-name g2a_full --cache-dir outputs/g2a_full
python scripts/evaluate.py --run-name g2b_dice --cache-dir outputs/g2b_dice
python scripts/evaluate.py --run-name g2c_focal --cache-dir outputs/g2c_focal
python scripts/evaluate.py --run-name g2d_aug --cache-dir outputs/g2d_aug
python scripts/evaluate.py --run-name g2e_linear --cache-dir outputs/g2e_linear
python scripts/evaluate.py --run-name g6_large --cache-dir outputs/g6_large
python scripts/evaluate.py --run-name g7_panel --cache-dir outputs/g7_panel
python scripts/evaluate.py --run-name g7_panel_ks --cache-dir outputs/g7_panel --k-sigma-grid 0.3 0.6 1.0 1.5 2.0
python scripts/evaluate.py --run-name g3_seed1_ks --cache-dir outputs/g3_seed1 --k-sigma-grid 0.3 0.6 1.0 1.5 2.0
python scripts/evaluate.py --run-name g3_seed2_ks --cache-dir outputs/g3_seed2 --k-sigma-grid 0.3 0.6 1.0 1.5 2.0
```

## 4. Analyses on a trained run

```sh
python scripts/hybrid.py --run-name hybrid_g3_ks --cache-dir outputs/g3_full_aug --k-sigma 1.5 --workers 4
python scripts/hybrid.py --run-name hybrid_g3_full_aug --cache-dir outputs/g3_full_aug --workers 4
python scripts/hybrid.py --run-name hybrid_g2a_full --cache-dir outputs/g2a_full --workers 4
python scripts/hybrid.py --run-name hybrid_g6_large --cache-dir outputs/g6_large --workers 4
python scripts/hybrid.py --run-name hybrid_g7_panel --cache-dir outputs/g7_panel --workers 4
python scripts/ablate.py --run-name ablate_g1_base --cache-dir outputs/g1_base
python scripts/ablate.py --run-name ablate_g2a_full --cache-dir outputs/g2a_full
python scripts/ablate.py --run-name ablate_g3_full_aug --cache-dir outputs/g3_full_aug
python scripts/ablate.py --run-name ablate_g7_panel --cache-dir outputs/g7_panel
python scripts/robustness.py --run-name robust_g3 --method segmenter --checkpoint outputs/g3_full_aug --report-dir outputs/g3_full_aug_ks
python scripts/robustness.py --run-name robust_sift --method sift --min-inliers 15
python scripts/benchmark.py --run-name g5_benchmark --images 60
python scripts/make_tables.py
```

Test-time averaging (flip and rotation averaging on the g2a_full weights). The prediction command for the no-averaging case is below; the prediction command of `g4_tta_rot4` is not recorded in the stored logs. Both were evaluated with:

```sh
python scripts/predict.py --run-name g4_tta_none --encoder-input full --tta none --splits-to-predict val test --checkpoint outputs/g2a_full
python scripts/evaluate.py --run-name g4_tta_none --cache-dir outputs/g4_tta_none
python scripts/evaluate.py --run-name g4_tta_rot4 --cache-dir outputs/g4_tta_rot4
```

Multi-scale tiled inference:

```sh
python scripts/predict.py --run-name ms_g3 --encoder-input full --tta none --multiscale --checkpoint outputs/g3_full_aug
python scripts/evaluate.py --run-name ms_g3
python scripts/predict.py --run-name ms_g7 --encoder-input full --tta none --multiscale --checkpoint outputs/g7_panel
python scripts/evaluate.py --run-name ms_g7
```

## 5. Additional analyses

```sh
python scripts/paired.py
python scripts/sensitivity.py --run-name sensitivity
python scripts/calibration.py --val-csv results/c4_evalks_g3_full_aug/per_image_val.csv --test-csv results/c4_evalks_g3_full_aug/per_image_test.csv --output-prefix results/calibration/final_segmenter
python scripts/evaluate.py --run-name obj_g3_full_aug --cache-dir outputs/g3_full_aug --k-sigma-grid 0.3 0.6 1.0 1.5 2.0 --objectives mixed official forged_tnr90 forged_tnr95 --gate-by appearance
python scripts/evaluate.py --run-name obj_g3_seed1 --cache-dir outputs/g3_seed1 --k-sigma-grid 0.3 0.6 1.0 1.5 2.0 --objectives mixed official forged_tnr90 forged_tnr95 --gate-by appearance
python scripts/evaluate.py --run-name obj_g3_seed2 --cache-dir outputs/g3_seed2 --k-sigma-grid 0.3 0.6 1.0 1.5 2.0 --objectives mixed official forged_tnr90 forged_tnr95 --gate-by appearance
python scripts/baseline_sift.py --run-name obj_sift --objectives mixed official forged_tnr90 forged_tnr95
python scripts/baseline_superpoint.py --run-name obj_superpoint --objectives mixed official forged_tnr90 forged_tnr95
python scripts/hybrid.py --run-name obj_hybrid --cache-dir outputs/g3_full_aug --k-sigma 1.5 --workers 4 --objectives mixed official forged_tnr90 forged_tnr95
python scripts/cases.py --run-name cases --cache-dir outputs/g3_full_aug --cases both_hit:forged_22751 segmenter_only:forged_23930 keypoints_only:forged_55818 both_miss:forged_28953 false_alarm:authentic_35347 supplemental:supp_1712 --pairs 3
python scripts/cases.py --run-name supp_sheets --supp-all --sheet --cache-dir outputs/g3_full_aug
```

## Run names and results folders

Verified against the `[run]` lines of each `results/*/run.log`.

| Run name in the commands | Folder in `results/` |
| --- | --- |
| (audit output) | `c1_audit` |
| `c3_sift` | `c3_sift` |
| `baseline_superpoint`, `baseline_dense` | same name |
| `g1_base`, `g2b_dice`, `g2c_focal`, `g2d_aug`, `g6_large`, `g7_panel` (training) | same name |
| `g2a_full` (training and prediction) | `g2a_full_input` |
| `g1_base` (evaluation) | `c4_eval_g1` |
| `g2a_full` (evaluation) | `c4_eval_g2a_full_input` |
| `g2b_dice`, `g2c_focal`, `g2d_aug`, `g2e_linear`, `g3_full_aug`, `g6_large`, `g7_panel` (evaluation) | `c4_eval_<run name>` |
| `g4_tta_none`, `g4_tta_rot4` | `c4_eval_g4_tta/<run name>` |
| `g3_full_aug_ks` | `c4_evalks_g3_full_aug` |
| `g3_seed1_ks`, `g3_seed2_ks` | `c4_evalks_g3_seed1`, `c4_evalks_g3_seed2` |
| `g7_panel_ks` | `c4_evalks_g7_panel` |
| `ms_g3`, `ms_g7` | `c4_ms_g3`, `c4_ms_g7` |
| `ablate_g1_base`, `ablate_g3_full_aug`, `ablate_g7_panel` | `c5_ablate_g1_base`, `c5_ablate_g3_full_aug`, `c5_ablate_g7_panel` |
| `ablate_g2a_full` | `c5_ablate_g2a_full_input` |
| `hybrid_g3_full_aug` | `c6_hybrid_g3` |
| `hybrid_g3_ks` | `c6_hybrid_g3_ks` |
| `hybrid_g2a_full` | `c6_hybrid_g2a` |
| `hybrid_g6_large`, `hybrid_g7_panel` | `c6_hybrid_g6_large`, `c6_hybrid_g7_panel` |
| `robust_sift` | `c7_robust_sift` |
| `robust_g3` | `robust_g3` |
| `cases` | `c8_cases` |
| `sensitivity`, `supp_sheets` | `c9_checks` |
| `obj_g3_full_aug`, `obj_g3_seed1`, `obj_g3_seed2` | `c10_obj_segmenters` |
| `obj_sift` | `c11_obj_sift` |
| `obj_superpoint` | `c12_obj_superpoint` |
| `obj_hybrid` | `c13_obj_hybrid` |
| `g5_benchmark` | `g5_benchmark` |
| calibration (`--output-prefix`) | `calibration` |
| `make_tables.py`, `paired.py` | `tables` |

## Notes

- Gates (`area_min`, `prob_min`) and the threshold multiplier `k_sigma` are chosen on validation. The test split is scored once per configuration.
- Stage 2 of training unfreezes all encoder parameters (90.06M trainable for DINOv2-base).
- Bootstrap intervals resample near-duplicate groups, not images.
- Expect a few hundredths of forged F1 between seeds (test forged F1 0.462, 0.427 and 0.442 for seeds 42, 1 and 2); see `docs/experiment_log.md`.
- Trained weights and cached probability maps are not included in the repository; the commands above regenerate them.

# Results index

Each folder holds the stored output of one run: usually `report.json` (metrics with 95% group-bootstrap intervals), per-image statistics (`per_image_<split>.csv`), the validation sweep (`sweep_val.csv`) and the console log (`run.log`, whose `[run]` lines give the exact command). Commands are written in full in `docs/reproduce.md`; the run name used in a command is not always the folder name, and the mapping is listed there. Every run is also described in `docs/experiment_log.md`.

## Prefixes

| Prefix | Meaning |
| --- | --- |
| `c1` | Dataset audit: near-duplicate grouping, leakage audit and the grouped split |
| `c3` | SIFT self-matching baseline |
| `c4_eval` | Segmenter evaluation with the fixed default threshold multiplier (k_sigma 0.3); gates chosen on validation |
| `c4_evalks` | Segmenter evaluation with k_sigma chosen on validation from 0.3, 0.6, 1.0, 1.5 and 2.0, together with the gates; this is the reported configuration |
| `c4_ms` | Multi-scale tiled inference (not adopted) |
| `c5_ablate` | Post-processing ablations and stratified analysis on cached probability maps |
| `c6_hybrid` | Fusion of the SIFT matcher with a segmenter, seven rules |
| `c7_robust` | Robustness of the SIFT baseline to JPEG, blur, noise and rescaling |
| `c8` | Qualitative case panels and near-duplicate pairs |
| `c9` | Checks: grouping sensitivity, residual near-duplicates, multi-panel contact sheets |
| `c10` to `c13` | Re-selection under other objectives (competition score, forged F1 under an authentic TNR floor) for segmenters, SIFT, SuperPoint and the hybrid |
| `g` | Training runs (`history.csv` and training logs) and the runtime benchmark (`g5`) |
| `t` | Test-suite logs |
| `calibration` | Calibration of the image-level score |
| `tables` | Tables collected from the folders above |

## Folders

| Folder | Purpose | Used in the paper | Command (run name) |
| --- | --- | --- | --- |
| `c1_audit` | `splits.csv`, `audit.json` | Fig. 1, Methods (data and split) | `audit.py` |
| `c3_sift` | SIFT baseline | Table 1, Figs 4 and 5, Table 2 | `baseline_sift.py` (`c3_sift`) |
| `baseline_superpoint` | SuperPoint baseline | Table 1, Fig. 5 | `baseline_superpoint.py` (`baseline_superpoint`) |
| `baseline_dense` | Zero-shot DINOv2 self-similarity | Table 1 | `baseline_dense.py` and `evaluate.py` (`baseline_dense`) |
| `c4_eval_g1` | Crop-224 baseline segmenter | Table 1, Fig. 2, Table 2 | `evaluate.py` (`g1_base`) |
| `c4_eval_g2a_full_input` | Full 518 px input | Fig. 2 | `evaluate.py` (`g2a_full`) |
| `c4_eval_g2b_dice`, `c4_eval_g2c_focal`, `c4_eval_g2d_aug`, `c4_eval_g2e_linear` | Loss, augmentation and head variants | Fig. 2 | `evaluate.py` |
| `c4_eval_g3_full_aug` | Final recipe, fixed k_sigma | Figs 2 and 5 | `evaluate.py` (`g3_full_aug`) |
| `c4_eval_g6_large`, `c4_eval_g7_panel` | DINOv2-large and panel-collage variants | Figs 2 and 5 | `evaluate.py` |
| `c4_eval_g4_tta` | Flip and rotation averaging (two sub-folders) | Experiment log only | `evaluate.py` (`g4_tta_none`, `g4_tta_rot4`) |
| `c4_evalks_g3_full_aug` | Main segmenter (validation-chosen k_sigma) | Table 1, Figs 2 and 4, Table 2, calibration | `evaluate.py` (`g3_full_aug_ks`) |
| `c4_evalks_g3_seed1`, `c4_evalks_g3_seed2` | Seeds 1 and 2 of the main segmenter | Results text (seed spread) | `evaluate.py` (`g3_seed1_ks`, `g3_seed2_ks`) |
| `c4_evalks_g7_panel` | Panel variant with validation-chosen k_sigma | Experiment log only | `evaluate.py` (`g7_panel_ks`) |
| `c4_ms_g3`, `c4_ms_g7` | Multi-scale tiling | Figs 2 and 5 (`c4_ms_g3`) | `predict.py --multiscale` and `evaluate.py` |
| `c5_ablate_g3_full_aug` | Post-processing ablations of the main segmenter | Fig. 2c | `ablate.py` (`ablate_g3_full_aug`) |
| `c5_ablate_g1_base`, `c5_ablate_g2a_full_input`, `c5_ablate_g7_panel` | Ablations of other segmenters | Experiment log only | `ablate.py` |
| `c6_hybrid_g3_ks` | Main hybrid (refine, union, confirm and others) | Table 1, Figs 4 and 5, Table 2 | `hybrid.py` (`hybrid_g3_ks`) |
| `c6_hybrid_g3`, `c6_hybrid_g2a`, `c6_hybrid_g6_large`, `c6_hybrid_g7_panel` | Hybrids on other maps or with fixed k_sigma | Experiment log only | `hybrid.py` |
| `c7_robust_sift` | SIFT under corruptions | Fig. 4c | `robustness.py --method sift` (`robust_sift`) |
| `robust_g3` | Main segmenter under corruptions | Fig. 4c | `robustness.py --method segmenter` (`robust_g3`) |
| `c8_cases` | Case panels and near-duplicate pairs | Figs 1 and 3 | `cases.py` (`cases`) |
| `c9_checks` | Grouping sensitivity, residual near-duplicates, multi-panel sheets | Supplementary Tables (sensitivity, residual near-duplicates, multi-panel outcomes) and Figures (pairs, sheets) | `sensitivity.py` (`sensitivity`), `cases.py` (`supp_sheets`) |
| `c10_obj_segmenters`, `c11_obj_sift`, `c12_obj_superpoint`, `c13_obj_hybrid` | Other selection objectives | Supplementary Table (four objectives) | `evaluate.py`, `baseline_sift.py`, `baseline_superpoint.py`, `hybrid.py` with `--objectives` |
| `calibration` | Reliability, ECE and Brier score | Supplementary Table (calibration) | `calibration.py` |
| `g1_base`, `g2a_full_input`, `g2b_dice`, `g2c_focal`, `g2d_aug`, `g6_large`, `g7_panel` | Training histories and logs | Methods (training) | `train.py`, `predict.py` |
| `g5_benchmark` | Runtime and peak memory | Results text and Methods | `benchmark.py` (`g5_benchmark`) |
| `t0_tests`, `t1_tests` | Test-suite logs (11 and 19 passed) | None | `pytest` |
| `tables` | Main, confusion, ablation, stratified, robustness, runtime and training tables; paired differences; instance decomposition; multi-panel outcomes; residual leakage effect | Table 2 (`confusion_test`), Supplementary Tables | `make_tables.py`, `paired.py` |

Figure and table numbers follow `paper/main.tex`: Fig. 1 study design, Table 1 main results, Fig. 2 design choices, Table 2 confusion counts, Fig. 3 cases, Fig. 4 image quality and errors, Fig. 5 distribution shift.

## Large files

Model checkpoints (`*.pt`) and cached probability maps (`*.npz`, about 0.4 GB per split) are excluded from version control by `.gitignore`. Local copies, where present, sit in `g1_base/g1_base/`, `g2a_full_input/` and `g2d_aug/`. They are regenerated with `scripts/train.py` and `scripts/predict.py` using the commands in `docs/reproduce.md`.

## Trained weights

The final trained weights are not included in this repository. To obtain them, retrain the final recipe with `scripts/train.py --run-name g3_full_aug --encoder-input full --augment strong` and the pretrained DINOv2-base encoder, as described in `docs/reproduce.md`. Seed-to-seed variation of test forged F1 is a few hundredths (0.462, 0.427 and 0.442 for seeds 42, 1 and 2).

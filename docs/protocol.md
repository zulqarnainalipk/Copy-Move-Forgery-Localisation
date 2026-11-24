# Evaluation protocol

## Data and splits

- 4,078 near-duplicate groups are formed by a 16 by 16 perceptual hash with a Hamming distance of at most 20 of 256 bits. Images are keyed by label and file stem, because identifiers repeat between the authentic and forged folders.
- A stratified group split with 20 folds gives 14 folds for training, 3 for validation and 3 for testing, with no group crossing splits. Test holds 770 images (418 forged, 352 authentic).
- The 48 supplemental multi-panel figures are never used for training or tuning. They measure transfer to a different image type.
- A conventional random image-level 80/20 split is audited only to quantify leakage: 326 of its 1,027 held-out images have a near-duplicate in its training set.

## Selection rules

- Image-level gates (minimum candidate area, minimum mean probability) and the threshold multiplier `k_sigma` are chosen on validation by mixed pixel F1. Hybrid fusion rules choose their inlier and probability thresholds the same way.
- Test and supplemental images are scored once per configuration after selection.
- Ablation rows re-select gates on validation for each variant.

## Metrics

- Pixel F1, IoU and MCC per image, averaged over forged images ("forged") or over all images ("mixed", where an authentic image scores 1 if nothing is predicted).
- Image-level detection: true and false positives and negatives, true-negative rate, detection rate, MCC, AUROC, average precision and expected calibration error.
- Official score: the competition's instance-aware F1 with Hungarian matching and a penalty for excess predicted instances. An authentic image with no prediction scores 1, any authentic and forged mismatch scores 0. It was reimplemented from the published metric definition and unit-tested against the examples in that definition.
- 95% intervals come from a group bootstrap with 1,000 resamples; groups, not images, are resampled.
- Pixel and image counts of true and false positives and negatives are in `results/tables/confusion_*.csv`.

## Compared methods

- SIFT and SuperPoint self-matching with generalised nearest-neighbour matching, affine RANSAC and convex-hull masks.
- Zero-shot self-similarity of frozen DINOv2 patch features.
- DINOv2 segmenters differing in encoder input, augmentation, loss, head, backbone size and seed.
- Seven fusion rules between segmenter and keypoint matcher.

# Model card: copy-move forgery localiser

## Model

DINOv2-base encoder (patch size 14, 518 px input giving a 37 by 37 token grid) with a three-block convolutional decoder and a 1 by 1 output convolution that upsamples to a 518 by 518 probability map. 90.06M parameters, all trained in stage 2. An optional hybrid step fuses the map with SIFT self-matching.

## Intended use

Flagging and localising duplicated regions in single biomedical research images, as a screening aid for a human reviewer. It is not evidence of misconduct on its own.

## Training data

Recod.ai/LUC Scientific Image Forgery Detection training images (authentic and forged, with instance masks), split into 14, 3 and 3 near-duplicate-grouped folds for training, validation and test. 48 supplemental multi-panel figures are used only for testing.

## Evaluation

Test split of 770 images (418 forged, 352 authentic). Final segmenter: forged F1 0.462 (95% CI 0.428 to 0.498), authentic true-negative rate 0.906, image AUROC 0.876, official score 0.517. A second training seed gave forged F1 0.427, so quote a spread of about 0.03 to 0.04 for the recipe. On the supplemental figures the segmenter alone scores forged F1 0.086; the hybrid `union` rule reaches 0.283 forged F1 and 0.243 official score. Robustness to JPEG, blur, noise and rescaling is in `results/tables/robustness.csv`.

## Limitations

- Small regions are missed: forged F1 is 0.018 for regions under 0.5% of the image and 0.596 above 2%.
- Transfer to multi-panel figures is poor. Panel-collage augmentation, a larger backbone and multi-scale tiling did not fix it.
- Keypoint matching is slow on CPU (about 0.75 s per image), the segmenter takes about 0.1 s per image at full resolution on a T4 GPU without test-time averaging.
- Authentic images are flagged at a measurable rate (33 of 352 on test), so every flag needs human review.
- Trained on one source of biomedical images; behaviour on other modalities is untested.

## Dual use

The same detector could be used to tune forgeries until they evade detection. Release the weights and code with this in mind, and do not use scores to accuse individuals without independent review.

"""Paired group-bootstrap differences for the main comparisons of the study."""
import argparse
from pathlib import Path

from forgery.paired import paired_table

SEGMENTER = "c4_evalks_g3_full_aug/per_image_test.csv"
SIFT = "c3_sift/per_image_test.csv"
CROP = "c4_eval_g1/per_image_test.csv"
FULL = "c4_eval_g2a_full_input/per_image_test.csv"
REFINE = "c6_hybrid_g3_ks/per_image_refine_test.csv"
UNION = "c6_hybrid_g3_ks/per_image_union_test.csv"
SEED1 = "c4_evalks_g3_seed1/per_image_test.csv"

COMPARISONS = [
    ("final segmenter minus SIFT", SEGMENTER, SIFT, "forged_pixel_f1"),
    ("final segmenter minus SIFT", SEGMENTER, SIFT, "official_score"),
    ("crop baseline minus SIFT", CROP, SIFT, "forged_pixel_f1"),
    ("crop baseline minus SIFT", CROP, SIFT, "official_score"),
    ("full input minus crop baseline", FULL, CROP, "forged_pixel_f1"),
    ("final segmenter minus full input", SEGMENTER, FULL, "forged_pixel_f1"),
    ("hybrid refine minus final segmenter", REFINE, SEGMENTER, "forged_pixel_f1"),
    ("hybrid refine minus final segmenter", REFINE, SEGMENTER, "official_score"),
    ("hybrid refine minus SIFT", REFINE, SIFT, "forged_pixel_f1"),
    ("hybrid union minus final segmenter", UNION, SEGMENTER, "forged_pixel_f1"),
    ("hybrid union minus final segmenter", UNION, SEGMENTER, "official_score"),
    ("hybrid union minus SIFT", UNION, SIFT, "official_score"),
    ("seed 1 minus final segmenter", SEED1, SEGMENTER, "forged_pixel_f1"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="folder holding the per-run results")
    parser.add_argument("--resamples", type=int, default=1000, help="number of paired group-bootstrap resamples")
    arguments = parser.parse_args()
    results = Path(arguments.results_dir)
    table = paired_table(COMPARISONS, results, arguments.resamples)
    (results / "tables").mkdir(exist_ok=True)
    table.to_csv(results / "tables" / "paired_differences.csv", index=False)
    print(table.round(3).to_string(index=False))

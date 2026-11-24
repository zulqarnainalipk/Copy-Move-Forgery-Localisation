"""Evaluate the SIFT self-matching baseline on validation, test and supplemental images."""
import argparse
import os

from forgery.baselines import evaluate_baseline


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_sift",
                        help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="longer image side in pixels above which images are downscaled before keypoint detection")
    parser.add_argument("--max-images", type=int, default=0, help="debug: limit images per split (0 means all)")
    parser.add_argument("--objectives", nargs="+", default=["mixed"], choices=["mixed", "official", "forged_tnr90", "forged_tnr95"],
                        help="selection objectives on validation; the first one writes the unsuffixed files")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="number of group-bootstrap resamples for the 95%% intervals")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    evaluate_baseline(arguments)

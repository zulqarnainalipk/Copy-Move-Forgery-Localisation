"""Render image, ground truth and method predictions for chosen cases (name:uid)."""
import argparse
import os

from forgery.cases import render_cases, render_pairs


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="cases", help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--cases", nargs="*", default=[], help="name:uid pairs, for example hit:forged_22751")
    parser.add_argument("--cache-dir", help="folder with probs_test.npz and probs_supp.npz")
    parser.add_argument("--k-sigma", type=float, default=1.5, help="threshold multiplier of the segmenter mask")
    parser.add_argument("--area-min", type=int, default=100, help="minimum mask area in pixels (segmenter gate)")
    parser.add_argument("--seg-probability", type=float, default=0.10,
                        help="minimum mean probability of the segmenter mask (segmenter gate)")
    parser.add_argument("--sift-inliers", type=int, default=15, help="minimum RANSAC inliers of the SIFT matcher")
    parser.add_argument("--refine-inliers", type=int, default=15, help="minimum inliers of the refine fusion rule")
    parser.add_argument("--refine-probability", type=float, default=0.10,
                        help="minimum segmenter probability of the refine fusion rule")
    parser.add_argument("--union-inliers", type=int, default=30, help="minimum inliers of the union fusion rule")
    parser.add_argument("--union-probability", type=float, default=0.10,
                        help="minimum segmenter probability of the union fusion rule")
    parser.add_argument("--supp-all", action="store_true", help="render every supplemental figure")
    parser.add_argument("--sheet", action="store_true", help="write one contact sheet per case instead of separate panels")
    parser.add_argument("--pairs", type=int, default=0, help="also write this many real near-duplicate pairs")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    if arguments.pairs:
        render_pairs(arguments)
    if arguments.cases or arguments.supp_all:
        render_cases(arguments)

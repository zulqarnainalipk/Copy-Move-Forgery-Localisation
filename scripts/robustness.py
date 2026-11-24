"""Evaluate a trained segmenter or the keypoint baseline on corrupted test images."""
import argparse
import os

from forgery.robustness import robustness_run


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--method", choices=["segmenter", "sift"], default="segmenter",
                        help="model to evaluate: trained segmenter or SIFT baseline")
    parser.add_argument("--checkpoint", help="checkpoint file or its folder (segmenter)")
    parser.add_argument("--report-dir", help="folder with the report.json of the clean evaluation (segmenter gates)")
    parser.add_argument("--min-inliers", type=int, default=15, help="validation-chosen threshold (sift)")
    parser.add_argument("--encoder-input", choices=["crop224", "full"], default="full",
                        help="crop224: resize 256 and centre-crop 224; full: the whole image at --img-size")
    parser.add_argument("--head", choices=["conv3", "linear"], default="conv3",
                        help="decoder head: three convolution blocks or a single 1x1 convolution")
    parser.add_argument("--img-size", type=int, default=518, help="side length in pixels of the square encoder input")
    parser.add_argument("--tta", choices=["none", "flip3", "rot4"], default="flip3",
                        help="test-time averaging: none, three flips or four rotations")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--max-images", type=int, default=0, help="debug: limit images per split (0 means all)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    robustness_run(arguments)

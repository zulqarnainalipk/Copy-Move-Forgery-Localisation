"""Write zero-shot self-similarity maps of frozen DINOv2 features, ready for scripts/evaluate.py."""
import argparse
import os

from forgery.dense import cache_dense


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_dense",
                        help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--img-size", type=int, default=518, help="side length in pixels of the square encoder input")
    parser.add_argument("--max-images", type=int, default=0, help="debug: limit images per split (0 means all)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    cache_dense(arguments)

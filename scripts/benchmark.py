"""Measure inference time and peak memory of each component on test images."""
import argparse
import os

from forgery.benchmark import benchmark


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--images", type=int, default=60, help="number of test images to time")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    benchmark(arguments)

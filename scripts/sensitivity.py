"""Sensitivity of the near-duplicate grouping to its Hamming threshold, plus the near-duplicate pairs file."""
import argparse
import os

from forgery.sensitivity import run


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sensitivity", help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--workers", type=int, default=4, help="number of worker processes")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    run(arguments)

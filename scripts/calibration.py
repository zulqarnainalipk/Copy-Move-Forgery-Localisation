"""Reliability, ECE and Brier score of the image-level score before and after Platt or isotonic calibration on validation."""
import argparse

from forgery.calibration import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", required=True, help="per-image validation statistics with is_forged and mean_probability")
    parser.add_argument("--test-csv", required=True,
                        help="per-image test statistics with is_forged and mean_probability")
    parser.add_argument("--output-prefix", default="results/calibration",
                        help="path prefix of the reliability tables and summary JSON")
    run(parser.parse_args())

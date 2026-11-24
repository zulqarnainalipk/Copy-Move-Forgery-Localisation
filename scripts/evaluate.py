"""Choose gates on validation and report metrics on validation, test and supplemental images."""
import argparse
import os

from forgery.evaluate import evaluate_run


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--cache-dir", help="folder with probs_<split>.npz (default: the run folder)")
    parser.add_argument("--k-sigma-grid", type=float, nargs="+", help="candidate threshold multipliers, chosen on validation")
    parser.add_argument("--objectives", nargs="+", default=["mixed"], choices=["mixed", "official", "forged_tnr90", "forged_tnr95"],
                        help="selection objectives on validation; the first one writes the unsuffixed files")
    parser.add_argument("--gate-by", choices=["appearance"], help="choose gates separately for each value of this attribute")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="number of group-bootstrap resamples for the 95%% intervals")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    evaluate_run(arguments)

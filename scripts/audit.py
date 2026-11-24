"""Audit the dataset, group near-duplicates and write leakage-safe splits."""
import argparse
import os

from forgery import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="where splits.csv and audit.json go (overrides FORGERY_OUTPUT_DIR)")
    args = parser.parse_args()
    if args.data_dir:
        os.environ["FORGERY_DATA_DIR"] = args.data_dir
    if args.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = args.output_dir
    audit.run()


if __name__ == "__main__":
    main()

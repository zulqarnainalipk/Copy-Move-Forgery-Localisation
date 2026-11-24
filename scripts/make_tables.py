"""Build the comparison, ablation, robustness and runtime tables from the stored results."""
import argparse

from forgery.tables import build_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="folder holding the per-run results")
    build_tables(parser.parse_args().results_dir)

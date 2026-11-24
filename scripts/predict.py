"""Cache probability maps of a trained run for the chosen splits."""
import argparse
import os

from forgery.predict import predict


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--splits-to-predict", nargs="+", default=["val", "test", "supp"],
                        help="splits to write probability maps for")
    parser.add_argument("--encoder-input", choices=["crop224", "full"], default="crop224",
                        help="crop224: resize 256 and centre-crop 224; full: the whole image at --img-size")
    parser.add_argument("--head", choices=["conv3", "linear"], default="conv3",
                        help="decoder head: three convolution blocks or a single 1x1 convolution")
    parser.add_argument("--img-size", type=int, default=518, help="side length in pixels of the square encoder input")
    parser.add_argument("--tta", choices=["none", "flip3", "rot4"], default="flip3",
                        help="test-time averaging: none, three flips or four rotations")
    parser.add_argument("--checkpoint", help="checkpoint file or its folder (default: the run folder)")
    parser.add_argument("--multiscale", action="store_true", help="add nine half-size windows to the global map")
    parser.add_argument("--max-images", type=int, default=0, help="debug: limit images per split (0 means all)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    predict(arguments)

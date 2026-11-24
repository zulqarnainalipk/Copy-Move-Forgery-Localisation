"""Train the segmenter on the grouped split."""
import argparse
import os

from forgery.train import train


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="name of the output folder under FORGERY_OUTPUT_DIR")
    parser.add_argument("--splits", help="path to splits.csv (default: <output-dir>/splits.csv)")
    parser.add_argument("--data-dir", help="dataset root (overrides FORGERY_DATA_DIR)")
    parser.add_argument("--output-dir", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--encoder-input", choices=["crop224", "full"], default="crop224",
                        help="crop224: resize 256 and centre-crop 224; full: the whole image at --img-size")
    parser.add_argument("--head", choices=["conv3", "linear"], default="conv3",
                        help="decoder head: three convolution blocks or a single 1x1 convolution")
    parser.add_argument("--loss", choices=["bce", "bce_dice", "focal", "tversky"], default="bce", help="training loss")
    parser.add_argument("--augment", choices=["none", "strong", "panel"], default="none",
                        help="training augmentation: none, strong photometric and geometric, or strong plus panel collages")
    parser.add_argument("--img-size", type=int, default=518, help="side length in pixels of the square encoder input")
    parser.add_argument("--batch-size", type=int, default=2, help="images per optimiser step before accumulation")
    parser.add_argument("--accumulation", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--epochs1", type=int, default=16, help="maximum epochs of stage 1 (decoder warmup)")
    parser.add_argument("--epochs2", type=int, default=16, help="maximum epochs of stage 2 (joint fine-tuning)")
    parser.add_argument("--patience1", type=int, default=3, help="early-stopping patience of stage 1")
    parser.add_argument("--patience2", type=int, default=5, help="early-stopping patience of stage 2")
    parser.add_argument("--lr-head", type=float, default=1e-5, help="learning rate of the decoder")
    parser.add_argument("--lr-encoder", type=float, default=5e-7, help="learning rate of the encoder in stage 2")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--workers", type=int, default=2, help="number of worker processes")
    parser.add_argument("--max-images", type=int, default=0, help="debug: limit images per split")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed-precision training")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.data_dir:
        os.environ["FORGERY_DATA_DIR"] = arguments.data_dir
    if arguments.output_dir:
        os.environ["FORGERY_OUTPUT_DIR"] = arguments.output_dir
    train(arguments)

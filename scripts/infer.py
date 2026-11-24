"""Localise copy-move regions in images with a trained checkpoint."""
import argparse
import os

from forgery.infer import infer


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="image files or glob patterns")
    parser.add_argument("--checkpoint", required=True, help="checkpoint file or its folder")
    parser.add_argument("--output-dir", default="predictions", help="output root (overrides FORGERY_OUTPUT_DIR)")
    parser.add_argument("--mode", default="seg_only",
                        choices=["seg_only", "sift_only", "sift_priority", "seg_priority", "union", "confirm", "refine"])
    parser.add_argument("--min-inliers", type=int, default=15, help="minimum RANSAC inliers for a keypoint detection")
    parser.add_argument("--min-probability", type=float, default=0.15,
                        help="minimum mean segmenter probability for a segmenter detection")
    parser.add_argument("--encoder-input", choices=["crop224", "full"], default="full",
                        help="crop224: resize 256 and centre-crop 224; full: the whole image at --img-size")
    parser.add_argument("--head", choices=["conv3", "linear"], default="conv3",
                        help="decoder head: three convolution blocks or a single 1x1 convolution")
    parser.add_argument("--img-size", type=int, default=518, help="side length in pixels of the square encoder input")
    parser.add_argument("--tta", choices=["none", "flip3", "rot4"], default="flip3",
                        help="test-time averaging: none, three flips or four rotations")
    parser.add_argument("--dinov2-dir", help="folder with the pretrained DINOv2 weights (overrides FORGERY_DINOV2_DIR)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.dinov2_dir:
        os.environ["FORGERY_DINOV2_DIR"] = arguments.dinov2_dir
    infer(arguments)

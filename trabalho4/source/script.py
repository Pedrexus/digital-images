__author__ = "Pedro Henrique Vaz Valois"

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import skimage as sk
import skimage.color
import skimage.feature

# region: argument parser
parser = argparse.ArgumentParser(
    description="Image Segmentation",
    epilog=f"MO443 - Project 4 - {__author__}",
)

# positional arguments
parser.add_argument(
    "input", help="path to image that the transformation will be applied onto"
)
parser.add_argument("output", help="path to write the transformed image to")

# optional arguments
parser.add_argument(
    "-w",
    "--overwrite",
    action="store_true",
    help="should it overwrite the output image if it already exists",
)

parser.add_argument(
    "-a",
    "--apply",
    choices=["mono", "lbp", "y"],
    default=None,
    help="transformation to be applied to the input image",
)

parser.add_argument(
    "--lbp",
    type=int,
    default=8,
    help="number of circularly symmetric neighbor set points",
)

args = parser.parse_args()

# endregion


def hist(img):
    n_bins = int(img.max() + 1)
    hist, _ = np.histogram(img, density=True, bins=n_bins, range=(0, n_bins))
    return hist


class LBP:
    def __init__(self, img, n_points, radius):
        self.img = sk.feature.local_binary_pattern(img, n_points, radius)


class Image:
    @classmethod
    def process(cls, img):
        if args.apply is None:
            return img

        img = sk.color.rgb2gray(img)
        if args.apply == "mono":
            return img

        radius = 1
        img = sk.feature.local_binary_pattern(img, args.lbp * radius, radius)

        n_bins = int(img.max() + 1)
        plt.hist(img.ravel(), bins=n_bins // 10, range=(0, n_bins))
        plt.show()

        if args.apply == "lbp":
            return img

        return img


def read_image():
    return plt.imread(args.input, format="png")


def save_image(result):
    try:
        plt.imsave(args.output, result, format="png", cmap="gray")
    except AttributeError:
        result.savefig(args.output, format="png")


def may_overwrite_if_output_exists():
    file_exists = os.path.isfile(args.output)
    return (file_exists and args.overwrite) or (not file_exists)


if __name__ == "__main__":
    if may_overwrite_if_output_exists():
        img = read_image()
        result = Image.process(img)
        save_image(result)

        print("Finished processing for", args.input)
    else:
        print(
            "Nothing done. Output file",
            args.output,
            "exists and --overwrite [-w] was",
            args.overwrite,
        )

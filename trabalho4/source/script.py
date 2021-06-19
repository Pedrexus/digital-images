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
    choices=["mono", "lbp", "glcm"],
    default=None,
    help="transformation to be applied to the input image",
)

parser.add_argument(
    "--lbp",
    type=int,
    default=8,
    help="number of circularly symmetric neighbor set points",
)

parser.add_argument(
    "--other",
    help="image path to another image for histogram comparison",
)


parser.add_argument(
    "--hist",
    action="store_true",
    help="save histogram instead of transformed image",
)

args = parser.parse_args()


# endregion


def hist(img):
    n_bins = int(img.max() + 1)
    hist, _ = np.histogram(img, density=True, bins=n_bins, range=(0, n_bins))
    return hist


def plot_hist(img):
    fig = plt.figure()
    ax = fig.add_subplot()

    n_bins = int(img.max() + 1)
    ax.hist(img.ravel(), bins=n_bins // 10, range=(0, n_bins), edgecolor="black", linewidth=1.2)
    return fig



class Image:

    @staticmethod
    def lbp(img):
        radius = 1
        return sk.feature.local_binary_pattern(img, args.lbp * radius, radius)

    distances = {
        "euclidean": lambda a, b: np.linalg.norm(a - b),
        "bhattacharyya": lambda a, b: -np.log(np.sqrt(a * b).sum()),
        "chi-square": lambda a, b: (np.square(a - b) / (a + b + 1)).sum() / 2,
        "correlation": lambda a, b: np.corrcoef(a, b)[0, 1],
    }

    @staticmethod
    def glcm(img):
        return sk.feature.greycomatrix((img * 256).astype(np.uint8), [1], [0])

    props = {
        "2nd-momentum": lambda x: np.square(x/x.max()).sum(),
        "entropy": lambda x: -(x/x.max() * np.log((x + 1)/x.max())).sum(),
        "contrast": lambda x: sk.feature.greycoprops(x, 'contrast')[0, 0],
        "energy": lambda x: sk.feature.greycoprops(x, 'energy')[0, 0],
        "correlation": lambda x: sk.feature.greycoprops(x, 'correlation')[0, 0],
    }

    @classmethod
    def process(cls, img):
        if args.apply is None:
            return img

        if len(img.shape) == 3:
            img = sk.color.rgb2gray(img)

        if args.apply == "mono":
            return img

        if args.apply == "lbp":
            lbp1 = cls.lbp(img)

            if args.other:
                other = sk.color.rgb2gray(read_image(args.other))
                lbp2 = cls.lbp(other)

                hist1 = hist(lbp1)
                hist2 = hist(lbp2)

                print(f"Distances between {args.input} and {args.other} histograms")
                for name, func in cls.distances.items():
                    dist = func(hist1, hist2)
                    print(f"{name}: {dist:.2}")

            if args.hist:
                return plot_hist(lbp1)

            return lbp1

        if args.apply == "glcm":
            glcm1 = cls.glcm(img)

            print(f"Props of image {args.input}")
            for name, func in cls.props.items():
                dist = func(glcm1)
                print(f"{name}: {dist:.2f}")

            return glcm1

        return img


def read_image(path):
    return plt.imread(path, format="png")


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
        img = read_image(args.input)
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
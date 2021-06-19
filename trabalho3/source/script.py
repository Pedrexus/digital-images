__author__ = "Pedro Henrique Vaz Valois"

import os
import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, label
from skimage.morphology import convex_hull_image

# region: argument parser
parser = argparse.ArgumentParser(
    description="Image Segmentation",
    epilog=f"MO443 - Project 3 - {__author__}",
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
    choices=["gray", "edge", "label"],
    default=None,
    help="transformation to be applied to the input image",
)

parser.add_argument(
    "--data",
    action="store_true",
    help="print area, perimeter, centroid, eccentricity and solidity of each blob in the image",
)

parser.add_argument(
    "--hist",
    action="store_true",
    help="print number of small, medium and large blobs",
)

args = parser.parse_args()

# endregion

# constants
WHITE = 1
BLACK = 0


class Blob:

    @staticmethod
    def calc_eccentricity(edge, center):
        x, y = np.where(edge)

        c_xx = np.square(x - center[0]).mean()
        c_yy = np.square(y - center[1]).mean()
        c_xy = ((x - center[0]) * (y - center[1])).mean()

        a = c_xx + c_yy
        b = np.sqrt(a * a - 4 * (c_xx * c_yy - c_xy * c_xy))

        l1 = a + b
        l2 = a - b

        return l2 / l1

    @staticmethod
    def calc_solidity(blob, area):
        hull_area = convex_hull_image(blob).sum()
        return area / hull_area

class BlobImage:
    @staticmethod
    def to_bw(img):
        return np.where(np.mean(img, axis=2) == WHITE, WHITE, BLACK)

    @classmethod
    def to_edges(cls, img):
        # Roberts' gradients
        g = [np.array([[1, 0], [0, -1]]), np.array([[0, -1], [1, 0]])]

        # check if I can use scipy implementation
        edges = np.abs(convolve(img, g[0])) + np.abs(convolve(img, g[1]))

        # set img borders to background
        edges[0, ...] = edges[..., 0] = BLACK

        return np.where(edges >= WHITE, BLACK, WHITE)

    @staticmethod
    def to_labels(img):
        # background must be black for scipy.label
        return label(~np.bool_(img))

    @staticmethod
    def print_blob_data(n, labels, edges):
        print("número de regiões:", n)

        for l in range(1, n + 1):
            blob = labels == l
            blob_edge = edges & blob

            area = blob.sum()
            perimeter = blob_edge.sum()
            c = np.array(np.where(blob_edge)).mean(axis=1).astype(int)
            e = Blob.calc_eccentricity(blob_edge, c)
            s = Blob.calc_solidity(blob, area)

            print(f"região {l}: área: {area} perímetro: {perimeter} centróide: {c} eccentricity: {e:.2f} solidity: {s:.2f}")

    @staticmethod
    def plot_area_hist(n, labels):
        lb, ub = 1500, 3000

        areas = [len(labels[labels == l]) for l in range(1, n + 1)]

        def classify(area):
            if area < lb:
                return "small"
            elif lb <= area < ub:
                return "medium"
            else:
                return "large"

        counts = collections.Counter(classify(a) for a in areas)

        print("number of small regions:", counts["small"])
        print("number of medium regions:", counts["medium"])
        print("number of large regions:", counts["large"])

        bins = [min(*areas, lb), lb, ub, max(*areas, ub)]

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(areas, bins=bins, edgecolor="black", linewidth=1.2)
        plt.xlabel("Área")
        plt.ylabel("Número de Objetos")

        return fig

    @staticmethod
    def plot_image_labeled(img, n, labels, edges):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.imshow(img, cmap="gray")
        for l in range(1, n + 1):
            blob_edge = edges & (labels == l)
            c = np.array(np.where(blob_edge)).mean(axis=1)[::-1]
            offset = np.array([-8, 5])
            ax.annotate(l, c + offset, color="red")

        return fig

    @classmethod
    def process(cls, img):
        bw = cls.to_bw(img)
        if args.apply == "bw":
            return bw

        labels, n = cls.to_labels(bw)
        edges = cls.to_edges(bw)

        if args.data:
            cls.print_blob_data(n, labels, edges)

        if args.hist:
            return cls.plot_area_hist(n, labels)

        if args.apply == "edge":
            return edges

        if args.apply == "label":
            return cls.plot_image_labeled(bw, n, labels, edges)

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
        result = BlobImage.process(img)
        save_image(result)

        print("Finished processing for", args.input)
    else:
        print(
            "Nothing done. Output file",
            args.output,
            "exists and --overwrite [-w] was",
            args.overwrite,
        )

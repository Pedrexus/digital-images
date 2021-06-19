__author__ = "Pedro Henrique Vaz Valois"

import os
import argparse

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# region: argument parser


parser = argparse.ArgumentParser(
    description="Harmonic Analysis on Images",
    epilog=f"MO443 - Project 2 - {__author__}",
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
    "-f",
    "--filter",
    choices=["low", "mid", "rejectmid", "high"],
    help="filter to be applied on the image. Use 'skip' to avoid convolution",
)

parser.add_argument(
    "-fn",
    "--filter_name",
    choices=["ideal", "butterworth"],
    help="filter equation",
)

parser.add_argument(
    "-ff",
    "--filter_factor",
    type=float,
    help="filter factor to be used in the equation",
)


def compression_float_factor(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 1:
        raise argparse.ArgumentTypeError("%r must be larger or equal to 1" % (x,))

    return x


parser.add_argument(
    "-c",
    "--compress",
    type=compression_float_factor,
    help="compression factor.",
)

parser.add_argument(
    "-r",
    "--rotate45",
    action="store_true",
    help="rotate image 45 degrees counterclockwise",
)

parser.add_argument(
    "-s",
    "--space",
    action="store_true",
    help="Saves final result in original space. Otherwise, in fourier space",
)
# rotate, compression, ...

args = parser.parse_args()

# endregion


class FourierImage:

    spatial = 0
    frequency = 1

    def __init__(self, img: np.array):
        self.img = img
        self.data = np.copy(img)
        self.dim = self.spatial

    def to_freq(self):
        x = np.fft.fft2(self.data)
        x = np.fft.fftshift(x)

        self.data = x
        self.dim = self.frequency

        return self

    def to_spatial(self):
        x = np.fft.ifftshift(self.data)
        x = np.fft.ifft2(x)
        x = np.abs(x)

        self.data = x
        self.dim = self.spatial

        return self

    def visualize(self):
        x = self.data
        if self.dim == self.frequency:
            x = np.abs(x) + 1
            x = np.log(x)
        return x

    def assert_freq(self):
        assert (
            self.dim == self.frequency
        ), "Must convert to frequency dimensions beforehand"

    def convolve(self, make_kernel):
        self.assert_freq()
        self.data *= make_kernel(self.data.shape)

        return self

    def compress(self, factor):
        self.assert_freq()

        x = np.abs(self.data)
        self.data[x < factor] = 0

        return self

    def rotate45(self):
        self.data = rotate(self.data, 45, reshape=True)
        return self


def ideal_filter(shape, stop, start=0, reject=False):
    i, j = np.indices(shape)
    i0, j0 = (np.array(shape) - 1) // 2
    arr = np.zeros(shape)

    dist = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    indices = (start <= dist) & (dist <= stop)
    indices = ~indices if reject else indices

    arr[indices] = 1

    return arr


def simple_butterworth_filter(shape, factor):
    i, j = np.indices(shape)
    i0, j0 = (np.array(shape) - 1) // 2

    dist = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return 1 / (1 + (factor / (dist + 0.01)) ** (2 * np.sign(factor)))


def mid_butterworth_filer(shape, width, center, reject=False):
    i, j = np.indices(shape)
    i0, j0 = (np.array(shape) - 1) // 2

    dist = np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    center += 0.01
    d = width * dist / (dist * dist - center * center)
    f = 1 / (1 + d * d)
    return f if reject else 1 - f


class Kernel:
    class LowPass:
        @staticmethod
        def ideal(shape, stop):
            return ideal_filter(shape, stop)

        @staticmethod
        def butterworth(shape, factor):
            return simple_butterworth_filter(shape, -factor)

    class HighPass:
        @staticmethod
        def ideal(shape, start):
            return ideal_filter(shape, start, reject=True)

        @staticmethod
        def butterworth(shape, factor):
            return simple_butterworth_filter(shape, factor)

    class MidPass:
        @staticmethod
        def ideal(shape, start, stop):
            return ideal_filter(shape, stop, start)

        @staticmethod
        def butterworth(shape, factor):
            return mid_butterworth_filer(shape, factor / 10, factor)

    class RejectMidPass:
        @staticmethod
        def ideal(shape, start, stop):
            return ideal_filter(shape, stop, start, reject=True)

        @staticmethod
        def butterworth(shape, factor):
            return mid_butterworth_filer(shape, factor / 2, factor, reject=True)

    @classmethod
    def get_from_args(cls):
        kind = None

        if args.filter == "low":
            kind = cls.LowPass
        elif args.filter == "high":
            kind = cls.HighPass

        if kind:
            if args.filter_name == "ideal":
                return lambda s: kind.ideal(s, args.filter_factor)
            elif args.filter_name == "butterworth":
                return lambda s: kind.butterworth(s, args.filter_factor)

        if args.filter == "mid":
            kind = cls.MidPass
        elif args.filter == "rejectmid":
            kind = cls.RejectMidPass

        if args.filter_name == "ideal":
            return lambda s: kind.ideal(s, args.filter_factor, 10 * args.filter_factor)
        elif args.filter_name == "butterworth":
            return lambda s: kind.butterworth(s, args.filter_factor)

        raise NotImplementedError


def read_image():
    return plt.imread(args.input, format="png")


def save_image(result):
    plt.imsave(args.output, result, format="png", cmap="gray")


def measure_entropy(img):
    assert img.dtype == np.uint8
    _, counts = np.unique(img, return_counts=True)
    n = sum(counts)
    return -sum((n_i / n) * np.log(n_i / n) for n_i in counts)


def may_overwrite_if_output_exists():
    file_exists = os.path.isfile(args.output)
    return (file_exists and args.overwrite) or (not file_exists)


def process_image(image):
    x = FourierImage(img)

    if args.rotate45:
        x.rotate45()

    x.to_freq()

    if args.filter:
        make_kernel = Kernel.get_from_args()
        x.convolve(make_kernel)

    if args.compress:
        x.compress(args.compress)

    if args.space:
        x.to_spatial()

    return x.visualize()


if __name__ == "__main__":
    if may_overwrite_if_output_exists():
        img = read_image()
        result = process_image(img)
        save_image(result)

        print("Finished processing for", args.input)
    else:
        print(
            "Nothing done. Output file",
            args.output,
            "exists and --overwrite [-w] was",
            args.overwrite,
        )

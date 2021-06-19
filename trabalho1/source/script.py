
__author__ = "Pedro Henrique Vaz Valois"

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

# region: argument parser


parser = argparse.ArgumentParser(
    description='Execute different transformations on images',
    epilog=f'MO443 - Project 1 - {__author__}'
)

# positional arguments
parser.add_argument("input", help="path to image that the transformation will be applied onto")
parser.add_argument("output", help="path to write the transformed image to")

# optional arguments
parser.add_argument("-w", "--overwrite", action="store_true", help="should it overwrite the output image if it already exists")
parser.add_argument("--transformation", choices=["vector", "matrix"], help="Transformation to be applied. Ignored for grayscale images.")
parser.add_argument("--kernel",  help="kernel used at kernel convolution. Combine kernels with a + sign. Ignored for RGB images.") # TODO: & for combination
parser.add_argument("--cmap", default="gray", help="Colormap to save grayscale images. Ignored for RGB images.") # TODO: & for combination

args = parser.parse_args()

#endregion

# region solutions

def is_rgb(image):
    return image.shape[-1] == 3

def is_grayscale(image):
    return len(image.shape) == 2 or image.shape[-1] == 1

# 1.1. color

# a)

def color_matrix_transformation(image):
    assert is_rgb(image), "Image is not RGB"

    transformation = np.array([
        [.393, .349, .272],  # R
        [.769, .686, .534],  # G
        [.189, .168, .131],  # B  
    ])

    result = image @ transformation  # apply transformation
    result[result > 1] = 1  # apply threshold

    return result

# b)

def color_vector_transformation(image):
    assert is_rgb(image), "Image is not RGB"
    transformation = np.array([.2989, .5870, .114])
    return img @ transformation

# 1.2 grayscale

# filters

def get_kernels():
    return dict(

    h1 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]),

    h2 = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ]),

    h3 = np.array([
        [-1, -1, -1],
        [-1,  8,  1],
        [-1, -1, -1],
    ]),

    h4 = np.ones((3, 3)) / 9,

    h5 = np.array([
        [-1, -1,  2],
        [-1,  2, -1],
        [ 2, -1, -1],
    ]),

    h6 = np.array([
        [ 2, -1, -1],
        [-1,  2, -1],
        [-1, -1,  2],
    ]),

    h7 = np.array([
        [ 0, 0, 1],
        [ 0, 0, 0],
        [-1, 0, 0],
    ]),

    h8 = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ]),

    h9 = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ]) / 256,

)

def grayscale_kernel_convolution(image, kernel):
    assert is_grayscale(image), f"Image must have no color channels, but had {image.shape[-1]}"

    # padding with zeros
    p = np.array(image.shape) % (2 * np.array(kernel.shape))

    img_pad = np.pad(image, p)

    M, N = kernel.shape
    W, H = image.shape

    i0 = np.repeat(np.arange(M), M)
    i1 = np.repeat(np.arange(W), W)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(N), N)
    j1 = np.tile(np.arange(H), H)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (kernel.ravel() @ img_pad[i, j]).reshape(image.shape)

def multikernel_convolution(image, kernels):
    result = np.zeros(image.shape)
    for k in kernels:
        result += np.square(grayscale_kernel_convolution(image, k))
    return np.sqrt(result)

def make_convolution(image):
    if not args.kernel:
        raise ValueError(f"invalid kernel {args.kernel}")
    
    all_kernels = get_kernels()
    
    if "+" not in args.kernel:
        return grayscale_kernel_convolution(image, all_kernels[args.kernel])
    else:
        kernels = [all_kernels[h.strip()] for h in args.kernel.split("+")]
        return multikernel_convolution(image, kernels)

# endregion

rgb_transformations = {
    "matrix": color_matrix_transformation,
    "vector": color_vector_transformation,
}

def select_and_apply_function(image):
    if is_rgb(image):
        f = rgb_transformations.get(args.transformation)
        if f is None:
            raise ValueError("--transformation was undefined")
        return f(image), f.__name__
    else:
        return make_convolution(image), grayscale_kernel_convolution.__name__

def read_image():
    return plt.imread(args.input, format="png")

def save_image(result):
    plt.imsave(args.output, result, format="png", cmap=args.cmap)

def may_overwrite_if_output_exists():
    file_exists = os.path.isfile(args.output)
    return (file_exists and args.overwrite) or (not file_exists)

if may_overwrite_if_output_exists():
    img = read_image()
    result, fname = select_and_apply_function(img)
    save_image(result)
    print("Finished processing", fname, "for", args.input)
else:
    print("Nothing done. Output file", args.output, "exists and --overwrite [-w] was", args.overwrite)

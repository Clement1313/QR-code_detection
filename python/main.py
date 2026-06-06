import skimage as ski
import numpy as np


def load_image(filepath: str) -> np.ndarray:
    return ski.io.imread(filepath)


def save_image(filepath: str, im: np.ndarray):
    ski.io.imsave(filepath, im)


def histogram_equalization(im: np.ndarray)-> np.ndarray:
    equalized = np.stack(
        [ski.exposure.equalize_hist(im[:, :, c]) for c in range(im.shape[2])], axis=-1
    )
    return (equalized * 255).astype(np.uint8)

def grayscale(im: np.ndarray)-> np.ndarray:
    return (ski.color.rgb2gray(im) * 255).astype(np.uint8)

def bin(im: np.ndarray)-> np.ndarray:
    thresh = ski.filters.threshold_otsu(im)
    return ((im > thresh) * 255).astype(np.uint8)

def preprocess(im: np.ndarray)-> np.ndarray:
    im = load_image("../data/IMG_3888.JPG")
    eqalized = histogram_equalization(im)
    gray = grayscale(eqalized)
    bin_im = bin(gray)
    return bin_im

def main():
    im = load_image("../data/IMG_3888.JPG")
    res = preprocess(im)
    save_image("res.png", res)


main()

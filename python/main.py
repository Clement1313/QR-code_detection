import skimage as ski
import numpy as np


def load_image(filepath: str) -> np.ndarray:
    return ski.io.imread(filepath)


def save_image(filepath: str, im: np.ndarray):
    ski.io.imsave(filepath, im)


def histogram_equalization(im: np.ndarray) -> np.ndarray:
    equalized = np.stack(
        [ski.exposure.equalize_hist(im[:, :, c]) for c in range(im.shape[2])], axis=-1
    )
    return (equalized * 255).astype(np.uint8)


def grayscale(im: np.ndarray) -> np.ndarray:
    return (ski.color.rgb2gray(im) * 255).astype(np.uint8)


def bin(im: np.ndarray) -> np.ndarray:
    thresh = ski.filters.threshold_otsu(im)
    return ((im > thresh) * 255).astype(np.uint8)


def preprocess(im: np.ndarray) -> np.ndarray:
    im = load_image("../data/IMG_3888.JPG")
    eqalized = histogram_equalization(im)
    gray = grayscale(eqalized)
    bin_im = bin(gray)
    return bin_im


def sobel(im: np.ndarray) -> np.ndarray:
    im = ski.filters.sobel(im)
    return (im * 255).astype(np.uint8)


def denoising(im: np.ndarray) -> np.ndarray:
    im = ski.morphology.closing(im)
    im = ski.morphology.opening(im)
    # im = ski.morphology.erosion(im)
    # im = ski.morphology.dilation(im)
    return (im * 255).astype(np.uint8)


def labels(im: np.ndarray) -> np.ndarray:
    return ski.measure.label(im, connectivity=1)


def draw_regions(im_original: np.ndarray, labels: np.ndarray, regions: np.ndarray) -> np.ndarray:
    overlay = ski.color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    overlay = (overlay * 255).astype(np.uint8)
    for region in regions:
        if region.area < 500:
            continue
        minr, minc, maxr, maxc = region.bbox
        rr, cc = ski.draw.rectangle_perimeter(
            (minr, minc), (maxr, maxc), shape=overlay.shape
        )
        overlay[rr, cc] = [255, 0, 0]
        im_original[rr, cc] = [255, 0, 0]
    return overlay, im_original


def draw_squares(im_original, squares):
    img = im_original.copy()

    for minr, minc, maxr, maxc in squares:
        rr, cc = ski.draw.rectangle_perimeter(
            start=(minr, minc),
            end=(maxr, maxc),
            shape=img.shape
        )
        img[rr, cc] = [0, 255, 0]

    return img


def find_squares(im_original: np.ndarray, regions: np.ndarray) -> np.ndarray:
    squares = []
    for region in regions:
        if region.area < 500:
            continue
        minr, minc, maxr, maxc = region.bbox
        if (abs(minc-maxc) - 10) <= abs(minr-maxr) and abs(minr-maxr) <= (abs(minc-maxc) + 10):
            squares.append((minr, minc, maxr, maxc))
    res = draw_squares(im_original, squares)
    print(len(squares))
    return squares, res


def main():
    im = load_image("../data/IMG_3888.JPG")
    res = preprocess(im)

    res = denoising(res)
    lab = labels(res)
    regions = ski.measure.regionprops(lab)
    # overlay, res = draw_regions(im, lab, regions)
    squares, res = find_squares(im, regions)


    save_image("res.png", res)


main()

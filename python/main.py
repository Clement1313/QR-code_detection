from itertools import combinations

import skimage as ski
import numpy as np
from pathlib import Path
from square_detection import (
    square_filter,
    get_triplets,
    get_qr_corners,
    draw_qr,
    get_center,
    filter_contained_triplets,
)


# =============================================================================
#               LOAD AND SAVE IMAGE
# =============================================================================
def load_image(filepath: str) -> np.ndarray:
    return ski.io.imread(filepath)


def save_image(filepath: str, im: np.ndarray):
    ski.io.imsave(filepath, im)


# =============================================================================
#               PREPROCESS
# =============================================================================
def histogram_equalization(im: np.ndarray) -> np.ndarray:
    equalized = np.stack(
        [ski.exposure.equalize_hist(im[:, :, c]) for c in range(im.shape[2])], axis=-1
    )
    return (equalized * 255).astype(np.uint8)


def grayscale(im: np.ndarray) -> np.ndarray:
    return (ski.color.rgb2gray(im) * 255).astype(np.uint8)


def bin(im: np.ndarray) -> np.ndarray:
    block_size = max(35, (min(im.shape[:2]) // 15) | 1)
    thresh = ski.filters.threshold_local(im, block_size, offset=10)
    return ((im > thresh) * 255).astype(np.uint8)


def preprocess(im: np.ndarray) -> np.ndarray:
    if im.shape[2] == 4:
        im = im[:, :, :3]
    eqalized = histogram_equalization(im)
    gray = grayscale(eqalized)
    gray = ski.filters.median(gray, ski.morphology.disk(2))
    bin_im = bin(gray)
    return eqalized, gray, bin_im


def denoising(im: np.ndarray) -> np.ndarray:
    selem = ski.morphology.disk(2)
    im = ski.morphology.closing(im, selem)
    im = ski.morphology.opening(im, selem)
    return im.astype(np.uint8)


# =============================================================================
#               LABEL AND DRAW
# =============================================================================
def labels(im: np.ndarray) -> np.ndarray:
    return ski.measure.label(im, connectivity=1)


def draw_regions(
    im_original: np.ndarray, labels: np.ndarray, regions: np.ndarray
) -> np.ndarray:
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


def draw_squares(im_original: np.ndarray, squares, color=[0, 255, 0]):
    img = im_original.copy()

    for minr, minc, maxr, maxc in squares:
        rr, cc = ski.draw.rectangle_perimeter(
            start=(minr, minc), end=(maxr, maxc), shape=img.shape
        )
        img[rr, cc] = color

    return img


# =============================================================================
#               DEBUG
# =============================================================================
def save_debug(
    image_name: str,
    original: np.ndarray,
    equalized: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    denoise: np.ndarray,
    labels: np.ndarray,
    regions: list,
):

    debug_dir = Path("../debug") / image_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    ski.io.imsave(debug_dir / "1_original.png", original)
    ski.io.imsave(debug_dir / "2_equalized.png", equalized)
    ski.io.imsave(debug_dir / "3_gray.png", gray)
    ski.io.imsave(debug_dir / "4_binary.png", binary)
    ski.io.imsave(debug_dir / "5_denoised.png", denoise)
    overlay = ski.color.label2rgb(labels, bg_label=0)
    overlay = (overlay * 255).astype(np.uint8)
    ski.io.imsave(debug_dir / "6_labels.png", overlay)

    bbox = original.copy()
    element = []
    for region in regions:
        res, coords = square_filter(denoise, binary, region)
        if res:
            element.append(
                {"bbox": region.bbox, "corners": coords, "area": region.area}
            )
            bbox = draw_squares(bbox, [region.bbox])
            minr, minc, maxr, maxc = region.bbox
            """
            for elt in coords:
                global_pt = (elt[0] + minr,elt[1] + minc)
                rr,cc = ski.draw.disk(global_pt,3)
                bbox[rr,cc] = [0, 0, 255]
            """
        else:
            bbox = draw_squares(bbox, [region.bbox], [0, 0, 255])
    triplets = get_triplets(element)
    triplets = filter_contained_triplets(triplets)
    for triplet in triplets:
        fourth = get_qr_corners(triplet)
        bbox = draw_qr(bbox, fourth)
    ski.io.imsave(debug_dir / "7_regions.png", bbox)


# =============================================================================
#               PROCESS FUNCTION OF THE IMAGES
# =============================================================================
def process_directory(image_path: str):
    im = load_image(image_path)
    equalized, gray, binary = preprocess(im)

    ### square detections
    denoise = denoising(binary)
    lab = labels(denoise)
    regions = ski.measure.regionprops(lab)

    element = []
    for region in regions:
        res, coords = square_filter(denoise, binary, region)
        if res:
            element.append(
                {"bbox": region.bbox, "corners": coords, "area": region.area}
            )

    triplets = get_triplets(element)
    triplets = filter_contained_triplets(triplets)
    print(f"{len(triplets)} QR code détectés")
    res = im.copy()
    qr_code = []
    for triplet in triplets:
        fourth = get_qr_corners(triplet)
        qr_code.append(fourth)
        res = draw_qr(res, fourth)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_res.png"
    save_image(str(output_path), res)

    return qr_code


def main():
    data_dir = Path("../data")

    for image_file in data_dir.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img_name = image_file.name
            print(f"\nTraitement de : {img_name}")
            process_directory(str(image_file))


# main()

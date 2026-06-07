import skimage as ski
import numpy as np
from pathlib import Path


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
    if im.shape[2] == 4:
        im = im[:, :, :3]
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


def draw_squares(im_original: np.ndarray, squares):
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
        if region.extent < 0.5:
            continue
        minr, minc, maxr, maxc = region.bbox
        if (abs(minc-maxc) - 10) <= abs(minr-maxr) and abs(minr-maxr) <= (abs(minc-maxc) + 10):
            squares.append((minr, minc, maxr, maxc))
    res = draw_squares(im_original, squares)
    # print(len(squares))
    return squares, res

def verif_square(im: np.ndarray, squares: np.ndarray) -> []:
    res = []
    for minr, minc, maxr, maxc in squares:
        square = im[minr:maxr, minc:maxc]
        e4 = ski.measure.euler_number(square, connectivity=1)
        object_nb_4 = ski.measure.label(square, connectivity=1).max()
        holes_nb_4 = object_nb_4 - e4
        # print(f'e4:{e4}, nb obj:{object_nb_4}, holes:{holes_nb_4}')
        if holes_nb_4 == 1 and object_nb_4 == 1:
            res.append((minr, minc, maxr, maxc))
    return res

def filter_qr_triplets(markers: list, max_distance: float = 500.0) -> list:
    if len(markers) < 3:
        return []
    centers = np.array([
        ((minr + maxr) / 2, (minc + maxc) / 2)
        for minr, minc, maxr, maxc in markers
    ])
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    triplets = []
    n = len(markers)
    for i in range(n):
        for j in range(i+1, n):
            if distance(centers[i], centers[j]) > max_distance:
                continue
            for k in range(j+1, n):
                if distance(centers[i], centers[k]) > max_distance:
                    continue
                if distance(centers[j], centers[k]) > max_distance:
                    continue
                triplets.append((markers[i], markers[j], markers[k]))
    return triplets

def process_directory(image_path: str):
    im = load_image(image_path)
    res = preprocess(im)

    ### square detection
    res = denoising(res)
    lab = labels(res)
    regions = ski.measure.regionprops(lab)
    # overlay, res = draw_regions(im, lab, regions)
    squares, res_draw = find_squares(im, regions)

    ### square verification
    mark = verif_square(res, squares)
    triplets = filter_qr_triplets(mark)
    print(f"{len(triplets)} QR code détectés")
    valid_markers = list({m for triplet in triplets for m in triplet})
    res = draw_squares(im, valid_markers)


    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_res.png"
    save_image(str(output_path), res)
    # save_image("res.png", res_draw)


def main():
    data_dir = Path("../data")

    for image_file in data_dir.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            process_directory(str(image_file))
    # process_directory("../data/Pastedimage.JPG")

main()

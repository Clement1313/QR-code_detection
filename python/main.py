from itertools import combinations

import skimage as ski
import numpy as np
from pathlib import Path
from square_detection import square_filter,get_triplets,get_qr_corners,draw_qr,get_center

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
    block_size = 35
    thresh = ski.filters.threshold_local(im,block_size,offset=10)
    return ((im > thresh) * 255).astype(np.uint8)


def preprocess(im: np.ndarray) -> np.ndarray:
    if im.shape[2] == 4:
        im = im[:, :, :3]
    eqalized = histogram_equalization(im)
    gray = grayscale(eqalized)
    bin_im = bin(gray)
    return eqalized, gray,bin_im


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
        #minr, minc, maxr, maxc = region.bbox
        #if (abs(minc-maxc) - 10) <= abs(minr-maxr) and abs(minr-maxr) <= (abs(minc-maxc) + 10):
        #    squares.append((minr, minc, maxr, maxc))
        squares.append(region.bbox)

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




def save_debug(image_name: str, original: np.ndarray, equalized: np.ndarray, gray: np.ndarray,binary: np.ndarray,
               denoise: np.ndarray,
               labels: np.ndarray,
               regions: list):

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
        res, coords = square_filter(denoise,binary,region)
        if res:
            element.append({ "bbox": region.bbox, "corners": coords, "area": region.area})

    triplets = get_triplets(element)
    for triplet in triplets:
        fourth = get_qr_corners(triplet)
        bbox = draw_qr(bbox,fourth)

    ski.io.imsave(debug_dir / "7_regions.png", bbox)

def process_directory(image_path: str):
    im = load_image(image_path)
    equalized, gray, binary= preprocess(im)

    ### square detection
    denoise = denoising(binary)
    lab = labels(denoise)
    regions = ski.measure.regionprops(lab)

    element = []
    for region in regions:
        res, coords = square_filter(denoise, binary, region)
        if res:
            element.append({"bbox": region.bbox, "corners": coords, "area": region.area})

    triplets = get_triplets(element)
    print(f"{len(triplets)} QR code détectés")
    res = im.copy()
    qr_code = []
    for triplet in triplets:
        fourth = get_qr_corners(triplet)
        qr_code.append(fourth)
        res = draw_qr(res, fourth)
    # valid_markers = list({m for triplet in triplets for m in triplet})


    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_res.png"
    save_image(str(output_path), res)
    # save_image("res.png", res_draw)
    #save_debug(Path(image_path).stem,im,equalized,gray,binary,denoise,lab,regions)
    return qr_code

def precision(tp: int, fp: int) -> float:
    if tp + fp:
        return float(tp) / float(tp + fp)
    return 0

def recall(tp: int, fn: int) -> float:
    if tp + fn:
        return float(tp) / float(tp + fn)
    return 0

def f1(tp: int, fp: int,fn:int) -> float:
    pres = precision(tp, fp)
    recal = recall(tp, fn)
    if pres + recal:
        return float(2* pres * recal) / float(pres + recal)
    return 0




def compute_iou( box1,  box2) :
    aminr, aminc, amaxr, amaxc = box1
    bminr, bminc, bmaxr, bmaxc = box2
    xA = max(aminc, bminc)
    yA = max(aminr, bminr)
    xB = min(amaxc, bmaxc)
    yB = min(amaxr, bmaxr)
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (amaxc-aminc)*(amaxr-aminr)
    areaB = (bmaxc-bminc)*(bmaxr-bminr)

    union = areaA + areaB - inter
    if union == 0:
        return 0

    return inter / union

def evaluate(detections, ground_truth, threshold=0.5):
    matched_gt = set()
    tp = 0
    fp = 0
    for det in detections:
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn

def mean_iou(detections, ground_truth):
    scores = []
    used = set()
    for det in detections:
        best = 0
        best_gt = -1
        for i,gt in enumerate(ground_truth):
            if i in used:
                continue
            iou = compute_iou(det,gt)
            if iou>best:
                best=iou
                best_gt=i
        if best_gt!=-1:
            used.add(best_gt)
            scores.append(best)
    if len(scores)==0:
        return 0
    return np.mean(scores)

def get_result(valid_makers,ground_truth):
    tp,fp,fn = evaluate(valid_makers,ground_truth)
    pres = precision(tp, fp)
    recal = recall(tp, fn)
    f = f1(tp,fp,fn)
    print("Precision :", pres)
    print("Recall :", recal)
    print("F1 :", f)


# Vérité terrain pour l'évaluation (3 marqueurs par QR code attendus)
groun_truh = {
    # 1. Image de l'écran de PC (QR code en bas à droite de l'écran)
    "IMG_3888.JPG": [
        (2910, 2110, 2990, 2190),  # Marqueur Haut-Gauche
        (2915, 2240, 2995, 2320),  # Marqueur Haut-Droite
        (3040, 2115, 3120, 2195)  # Marqueur Bas-Gauche
    ],

    # 2. Image du PC portable de face (Sticker QR sur la table à gauche)
    "IMG_3889.JPG": [
        (2210, 170, 2320, 280),  # Marqueur Haut-Gauche
        (2110, 310, 2220, 420),  # Marqueur Haut-Droite
        (2390, 315, 2500, 425)  # Marqueur Bas-Gauche
    ],

    # 3. Image du PC vue du dessus (Sticker QR plus proche et droit)
    "IMG_3890.JPG": [
        (1910, 380, 2030, 500),  # Marqueur Haut-Gauche
        (1915, 730, 2035, 850),  # Marqueur Haut-Droite
        (2250, 375, 2370, 495)  # Marqueur Bas-Gauche
    ],

    "IMG_3891.JPG": [
        (2040, 790, 2150, 900),  # Marqueur Haut-Gauche (vu incliné)
        (1880, 1070, 1990, 1180),  # Marqueur Haut-Droite
        (2210, 1140, 2320, 1250)  # Marqueur Bas-Gauche
    ],

    "IMG_3892.JPG": [
        (2110, 1720, 2240, 1850),  # Marqueur Haut-Gauche
        (2120, 2220, 2250, 2350),  # Marqueur Haut-Droite
        (2510, 1740, 2640, 1870)  # Marqueur Bas-Gauche
    ],

    "IMG_3895.JPG": [
        (1610, 1110, 1790, 1290),  # Marqueur Haut-Gauche (boîte englobante large due à l'angle)
        (1350, 1700, 1530, 1880),  # Marqueur Haut-Droite
        (2020, 1430, 2200, 1610)  # Marqueur Bas-Gauche
    ],

    "Pastedimage.JPG": [
        (220, 140, 290, 210),  # Marqueur Haut-Gauche
        (90, 245, 160, 315),  # Marqueur Haut-Droite
        (325, 250, 395, 320)  # Marqueur Bas-Gauche
    ]
}



def main():
    data_dir = Path("../data")

    for image_file in data_dir.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img_name = image_file.name
            print(f"\nTraitement de : {img_name}")
            predictions = process_directory(str(image_file))
            """
            if img_name in groun_truh:
                truth = groun_truh[img_name]
                print(f"Évaluation par rapport à la vérité terrain ({len(truth)} attendu(s)) :")
                # 3. On calcule et on affiche la Précision, le Rappel et le F1-Score
                get_result(predictions, truth)

                # Optionnel : afficher aussi le score moyen de superposition (IoU)
                miou = mean_iou(predictions, truth)
                print(f"Mean IoU : {miou:.2f}")
            """
    # process_directory("../data/Pastedimage.JPG")

main()
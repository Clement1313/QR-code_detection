import skimage as ski
import numpy as np
from pathlib import Path
from skimage.transform import resize
from itertools import combinations


def safe_crop(im, box):
    minr, minc, maxr, maxc = box
    minr = max(0, minr)
    minc = max(0, minc)
    maxr = min(im.shape[0], maxr)
    maxc = min(im.shape[1], maxc)
    if maxr <= minr or maxc <= minc:
        return None
    return im[minr:maxr, minc:maxc]

def region_similarity(im, box1, box2):

    r1 = safe_crop(im, box1)
    r2 = safe_crop(im, box2)
    if r1 is None or r2 is None:
        return 0.0
    if r1.size == 0 or r2.size == 0:
        return 0.0
    if np.isnan(r1).any() or np.isnan(r2).any():
        return 0.0

    r1 = resize(r1, (40, 40), preserve_range=True, anti_aliasing=False)
    r2 = resize(r2, (40, 40), preserve_range=True, anti_aliasing=False)
    r1 = r1 > 128
    r2 = r2 > 128
    return np.mean(r1 == r2)

def same_size(box1, box2, tolerance=0.20):
    h1 = box1[2]-box1[0]
    w1 = box1[3]-box1[1]
    h2 = box2[2]-box2[0]
    w2 = box2[3]-box2[1]

    return (abs(h1-h2)/max(h1,h2) < tolerance and abs(w1-w2)/max(w1,w2) < tolerance)

def center(box):
    minr,minc,maxr,maxc = box
    return np.array([(minr+maxr)/2,(minc+maxc)/2])

def angle(a,b,c):
    ba = a-b
    bc = c-b
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    cos = np.clip(cos,-1,1)
    return np.degrees(np.arccos(cos))

def has_right_angle(box1,box2,box3,tol=20):
    p1 = center(box1)
    p2 = center(box2)
    p3 = center(box3)
    angles = [angle(p2,p1,p3), angle(p1,p2,p3),angle(p1,p3,p2)]
    return any(abs(a-90)<tol for a in angles)

def similar_distances(box1,box2,box3,tol=0.3):
    p1 = center(box1)
    p2 = center(box2)
    p3 = center(box3)
    d12 = np.linalg.norm(p1-p2)
    d13 = np.linalg.norm(p1-p3)
    d23 = np.linalg.norm(p2-p3)
    d = sorted([d12,d13,d23])
    return abs(d[0]-d[1])/d[1] < tol

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
    for region in regions:
        if region.area < 500:
            continue
        minr, minc, maxr, maxc = region.bbox
        rr, cc = ski.draw.rectangle_perimeter(
            start=(minr, minc),
            end=(maxr, maxc),
            shape=bbox.shape
        )
        bbox[rr, cc] = [255, 0, 0]

    ski.io.imsave(debug_dir / "7_regions.png", bbox)

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
    t = ski.filters.threshold_otsu(im)
    return ((im > t) * 255).astype(np.uint8)


def preprocess(im: np.ndarray) -> np.ndarray:
    if im.shape[2] == 4:
        im = im[:, :, :3]
    eqalized = histogram_equalization(im)
    gray = grayscale(eqalized)
    bin_im = bin(gray)
    return eqalized,gray,bin_im


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

"""
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
"""


def find_squares(im_original: np.ndarray, regions: np.ndarray):
    squares = []
    for region in regions:
        if region.area < 150 or region.area > 80000:
            continue
        if region.eccentricity > 0.65:
            continue
        if region.solidity < 0.35 or region.solidity > 0.85:
            continue

        minr, minc, maxr, maxc = region.bbox
        squares.append((minr, minc, maxr, maxc))

    res = draw_squares(im_original, squares)
    return squares, res


def verif_square(im_binary: np.ndarray, squares: np.ndarray) -> list:
    res = []
    for minr, minc, maxr, maxc in squares:
        square = im_binary[minr:maxr, minc:maxc]
        if square.size == 0 or square.shape[0] < 8 or square.shape[1] < 8:
            continue
        e4 = ski.measure.euler_number(square, connectivity=1)
        labels_count = ski.measure.label(square, connectivity=1).max()
        holes = labels_count - e4

        if holes >= 1:
            res.append((minr, minc, maxr, maxc))
            continue
        h, w = square.shape
        center_zone = square[h // 3:2 * h // 3, w // 3:2 * w // 3]
        edge_mean = (np.mean(square[0, :]) + np.mean(square[-1, :]) + np.mean(square[:, 0]) + np.mean(
            square[:, -1])) / 4
        center_mean = np.mean(center_zone)

        if abs(center_mean - edge_mean) > 80:
            res.append((minr, minc, maxr, maxc))

    return res


def filter_qr_triplets(im, markers):
    triplets = []
    for m1, m2, m3 in combinations(markers, 3):
        if not same_size(m1, m2, tolerance=0.40) or not same_size(m1, m3, tolerance=0.40):
            continue
        if not similar_distances(m1, m2, m3, tol=0.40):
            continue
        if not has_right_angle(m1, m2, m3, tol=28):
            continue

        p1, p2, p3 = center(m1), center(m2), center(m3)
        dists = sorted([np.linalg.norm(p1 - p2), np.linalg.norm(p1 - p3), np.linalg.norm(p2 - p3)])
        if abs(dists[2] - dists[0] * np.sqrt(2)) / dists[2] > 0.35:
            continue
        triplets.append((m1, m2, m3))

    return triplets

def detect_qr(im):
    _,_,res = preprocess(im)
    res = denoising(res)
    lab = labels(res)
    regions = ski.measure.regionprops(lab)
    squares, _ = find_squares(im.copy(), regions)
    markers = verif_square(res, squares)
    return markers

def remove_duplicates(markers, distance=20):
    result = []
    centers = []
    for box in markers:
        minr, minc, maxr, maxc = box
        c = ((minr + maxr) / 2,
             (minc + maxc) / 2)
        duplicate = False
        for c2 in centers:
            d = np.sqrt((c[0]-c2[0])**2 + (c[1]-c2[1])**2)
            if d < distance:
                duplicate = True
                break
        if not duplicate:
            centers.append(c)
            result.append(box)
    return result

def detect_rotations(im):
    angles = range(0, 360, 15)
    markers = []
    for angle in angles:
        rotated = ski.transform.rotate(im,angle,resize=True,preserve_range=True,mode='constant',cval=0).astype(np.uint8)
        detected = detect_qr(rotated)
        markers.extend(detected)
    markers = remove_duplicates(markers)
    return markers

def process_directory(image_path: str):
    im = load_image(image_path)
    equalized,gray,binary = preprocess(im)

    ### square detection
    denoise = denoising(binary)
    lab = labels(denoise)
    regions = ski.measure.regionprops(lab)
    # overlay, res = draw_regions(im, lab, regions)
    squares, res_draw = find_squares(im, regions)

    ### square verification
    markers = detect_qr(im)
    triplets = filter_qr_triplets(im,markers)
    print(f"{len(triplets)} QR code détectés")
    valid_markers = list({m for triplet in triplets for m in triplet})
    res = draw_squares(im, valid_markers)


    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_res.png"
    save_image(str(output_path), res)
    save_debug(Path(image_path).stem,im,equalized,gray,binary,denoise,lab,regions)
    # save_image("res.png", res_draw)


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


groun_truh = {"IMG_3888.JPG":(803, 1478,918,1593),"IMG83839.JPG":(1050,20,1295,255)}

def main():
    data_dir = Path("../../data")

    for image_file in data_dir.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            process_directory(str(image_file))
    # process_directory("../data/Pastedimage.JPG")

main()
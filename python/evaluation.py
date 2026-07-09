from pathlib import Path
import numpy as np
import skimage as ski

from python.square_detection import should_remove_smaller, filter_contained_triplets
from square_detection import square_filter, get_triplets, get_qr_corners, get_center
from main import load_image, preprocess, denoising, labels,save_debug

def parse_ground_truth(txt_path: Path) -> list[list[tuple]]:
    qr_codes = []
    in_sets = False
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "SETS":
                in_sets = True
                continue
            if in_sets:
                values = list(map(float, line.split()))
                if len(values) == 8:
                    pts = [(values[i + 1], values[i]) for i in range(0, 8, 2)]
                    qr_codes.append(pts)
    return qr_codes


def corners_to_bbox(corners: list[tuple]) -> tuple:
    rows = [p[0] for p in corners]
    cols = [p[1] for p in corners]
    return (min(rows), min(cols), max(rows), max(cols))


def detect_qr(image_path: str) -> list[tuple]:
    im = load_image(image_path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]

    equalized, gray, binary = preprocess(im)
    denoise = denoising(binary)
    lab = labels(denoise)
    regions = ski.measure.regionprops(lab)

    elements = []
    for region in regions:
        res, coords = square_filter(denoise, binary, region)
        if res:
            elements.append({"bbox": region.bbox, "corners": coords, "area": region.area})

    triplets = get_triplets(elements,im.shape)
    triplets = filter_contained_triplets(triplets)
    detections = []
    for triplet in triplets:
        qr_corners = get_qr_corners(triplet)
        rows = [p[0] for p in qr_corners]
        cols = [p[1] for p in qr_corners]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        detections.append(bbox)
    save_debug(Path(image_path).stem, im,equalized,gray,binary,denoise,lab,regions)

    return detections


def compute_iou(box1: tuple, box2: tuple) -> float:
    aminr, aminc, amaxr, amaxc = box1
    bminr, bminc, bmaxr, bmaxc = box2
    xA = max(aminc, bminc)
    yA = max(aminr, bminr)
    xB = min(amaxc, bmaxc)
    yB = min(amaxr, bmaxr)
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (amaxc - aminc) * (amaxr - aminr)
    areaB = (bmaxc - bminc) * (bmaxr - bminr)
    union = areaA + areaB - inter
    if union == 0:
        return 0.0
    return inter / union


def evaluate(detections: list, ground_truth: list, iou_threshold: float = 0.5) -> tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    fp = 0
    for det in detections:
        best_iou = 0.0
        best_gt = -1
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mean_iou(detections: list, ground_truth: list) -> float:
    scores = []
    used = set()
    for det in detections:
        best = 0.0
        best_gt = -1
        for i, gt in enumerate(ground_truth):
            if i in used:
                continue
            iou = compute_iou(det, gt)
            if iou > best:
                best = iou
                best_gt = i
        if best_gt != -1:
            used.add(best_gt)
            scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def find_gt_file(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent / (image_path.stem + ".txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def evaluate_folder(folder_path: str, iou_threshold: float = 0.5, verbose: bool = True):
    folder = Path(folder_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    processed = 0
    skipped = 0

    for image_file in sorted(folder.glob("*")):
        if image_file.suffix.lower() not in image_extensions:
            continue

        gt_file = find_gt_file(image_file)
        if gt_file is None:
            if verbose:
                print(f"  [SKIP] Pas de vérité terrain pour {image_file.name}")
            skipped += 1
            continue

        gt_qr_list = parse_ground_truth(gt_file)
        gt_bboxes = [corners_to_bbox(qr) for qr in gt_qr_list]

        try:
            detections = detect_qr(str(image_file))
        except Exception as e:
            import traceback
            print(f"  [ERREUR] {image_file.name} : {e}")
            traceback.print_exc()
            skipped += 1
            continue

        tp, fp, fn = evaluate(detections, gt_bboxes, iou_threshold)
        miou = mean_iou(detections, gt_bboxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.append(miou)
        processed += 1

        if verbose:
            p = precision(tp, fp)
            r = recall(tp, fn)
            print(
                f"  {image_file.name:40s} | "
                f"det={len(detections)} gt={len(gt_bboxes)} | "
                f"TP={tp} FP={fp} FN={fn} | "
                f"P={p:.2f} R={r:.2f} IoU={miou:.2f}"
            )

    global_p = precision(total_tp, total_fp)
    global_r = recall(total_tp, total_fn)
    global_f1 = f1_score(total_tp, total_fp, total_fn)
    global_iou = float(np.mean(all_ious)) if all_ious else 0.0

    print(f"\n{'='*60}")
    print(f"Dossier : {folder}")
    print(f"Images traitées : {processed}  |  Ignorées : {skipped}")
    print(f"{'='*60}")
    print(f"Précision globale  : {global_p:.4f}")
    print(f"Rappel global      : {global_r:.4f}")
    print(f"F1 global          : {global_f1:.4f}")
    print(f"Mean IoU global    : {global_iou:.4f}")
    print(f"{'='*60}")

    return {
        "precision": global_p,
        "recall": global_r,
        "f1": global_f1,
        "mean_iou": global_iou,
    }


if __name__ == "__main__":
    DATASET_FOLDER = "../data/Dataset/detection/monitor"
    evaluate_folder(DATASET_FOLDER, iou_threshold=0.5, verbose=True)
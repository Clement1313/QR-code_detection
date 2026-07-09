from itertools import combinations

import skimage as ski
import numpy as np
from matplotlib import markers


def get_inner_square(square):
    lab = ski.measure.label(square)
    regions = ski.measure.regionprops(lab)
    region = max(regions,key=lambda r: r.area)
    inner_square = (lab == region.label)
    return inner_square

def get_corner(im:np.ndarray):
    contours =  ski.measure.find_contours(im.astype(float),0.5)
    if not len(contours):
        return np.empty((0,2))
    max_contour = max(contours, key=len)
    poly = ski.measure.approximate_polygon(max_contour,tolerance=6)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly


"""
def get_four_coord(coords):
    if coords.shape[0] == 4:
        return coords
    distance = []
    coords_list = [coord for coord in coords]
    intrus = None
    for (p1,p2) in combinations(coords_list,2):
        dx,dy = p1[0] - p2[0], p1[1] - p2[1]
        dist = (dx**2 + dy**2)**0.5
        distance.append(dist)
    m = np.mean(distance)
    sigma = np.std(distance)
    max_deviation = -1
    for coord in coords_list:
        dists = []
        for other_coord in coords_list:
            if coord[0] == other_coord[0] and coord[1] == other_coord[1]:
                continue
            dx,dy = coord[0] - other_coord[0], coord[1] - other_coord[1]
            dist = (dx**2 + dy**2)**0.5
            dists.append(dist)
        deviation = sum(abs(dist - m) for dist in dists)
        if deviation > max_deviation:
            max_deviation = deviation
            intrus = coord
    coords_list = [coord for coord in coords_list if coord[0] != intrus[0] and coord[1] != intrus[1]]
    return np.array(coords_list)
"""

def get_four_coord(coords):
    if coords.shape[0] < 4:
        return coords

    if coords.shape[0] > 4:
        try:
            hull = ski.morphology.convex_hull_image  # non, on veut juste les points
        except Exception:
            pass

        # Réduction via enveloppe convexe (scipy, plus direct que skimage ici)
    from scipy.spatial import ConvexHull
    if coords.shape[0] > 4:
        try:
            hull = ConvexHull(coords)
            coords = coords[hull.vertices]
        except Exception:
            pass  # points colinéaires ou dégénérés, on garde coords tel quel

    if coords.shape[0] == 4:
        return order_points(coords)

    best_subset = None
    best_score = float('inf')
    for subset in combinations(coords, 4):
        pts = np.array(subset)
        ordered = order_points(pts)
        sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
        diag1 = np.linalg.norm(ordered[0] - ordered[2])
        diag2 = np.linalg.norm(ordered[1] - ordered[3])
        score = np.std(sides) + abs(diag1 - diag2)
        if score < best_score:
            best_score = score
            best_subset = ordered

    return best_subset

def check_longeur(coords:np.ndarray)->bool:
    lengths = []
    for i in range(4):
        p1 = coords[i]
        p2 = coords[(i + 1) % 4]
        lengths.append(np.linalg.norm(p1 - p2))
    lengths = np.array(lengths)
    return lengths.min() * 1.3 > lengths.max()

def check_angle(coords:np.ndarray)->bool:
    angles = []
    coords_list = coords.copy()
    coords_list = coords_list.astype(float)
    for i in range(4):
        p0 = coords_list[i- 1]
        p1 = coords_list[i]
        p2 = coords_list[(i + 1) % 4]

        d1 = p0 - p1
        d2 = p2 - p1

        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        angle = np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0)))
        angles.append(angle)
    return all(65< angle < 115 for angle in angles)

def is_convex_quad(coords):
    n = 4
    cross_signs = []
    for i in range(n):
        p0 = coords[i - 1]
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_signs.append(np.sign(cross))
    return abs(sum(cross_signs)) == n

def check_side_ratio(coords, min_ratio=0.35):
    sides = [np.linalg.norm(coords[i] - coords[(i + 1) % 4]) for i in range(4)]
    return min(sides) / max(sides) >= min_ratio

def check_form(coords:np.ndarray)->bool:
    return coords.shape[0] == 4  and is_convex_quad(coords)  and check_side_ratio(coords)

def order_points(coords):
    coords = coords.astype(float)
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 0] - center[0], coords[:, 1] - center[1])
    return coords[np.argsort(angles)]

def check_region_shape(region, min_solidity=0.7, max_eccentricity=0.6, min_axis_ratio=0.5):
    #if region.solidity < min_solidity:
    #    return False
    if region.eccentricity > max_eccentricity:
        return False
    #if region.axis_minor_length == 0:
    #    return False
    #axis_ratio = region.axis_minor_length / region.axis_major_length
    #if axis_ratio < min_axis_ratio:
    #    return False
    return True

def square_filter(denoise:np.ndarray,binary:np.ndarray, region: object)->(bool,np.ndarray):
    if region.area < 50:
        return False,np.empty((0,2))

    minr, minc, maxr, maxc = region.bbox
    square1 = denoise[minr:maxr, minc:maxc]
    square2 = binary[minr:maxr, minc:maxc]
    # if not ski.measure.euler_number(square1, connectivity=1) == 1:
    #    return False,np.empty((0,2))

    # extraction of the square
    square_im = get_inner_square(square1)
    coords = get_corner(square_im)
    n_coords = coords.shape[0]
    if n_coords < 4:
        return False, coords

    coords = get_four_coord(coords)
    if coords is None:
        return False,np.empty((0,2))
    coords = order_points(coords)
    if not check_region_shape(region):
        return False, np.empty((0, 2))
    return check_form(coords), coords



def check_area(a1, a2, ratio=1.6):
    return max(a1, a2) / min(a1, a2) < ratio

def similar_area(m1, m2, m3):
    return (check_area(m1["area"], m2["area"]) and check_area(m1["area"], m3["area"]) and check_area(m2["area"], m3["area"]))

def angle(a, b, c):
    ba = a - b
    bc = c - b
    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc), -1, 1)))


def check_right_angle(center, tolerance=15):
    A,B,C = center[0], center[1], center[2]
    angles = [angle(B, A, C), angle(A, B, C),angle(A, C, B)]
    return any(abs(a - 90) <= tolerance for a in angles)
def check_distances(center, tolerance=0.20):
    A,B,C = center[0], center[1], center[2]
    d = sorted([np.linalg.norm(A - B), np.linalg.norm(A - C), np.linalg.norm(B - C)])
    sides_ok = abs(d[0] - d[1]) <= tolerance * d[0]
    diag_ok = abs(d[2] - np.sqrt(2) * d[0]) <= tolerance * d[2]
    return sides_ok and diag_ok

def get_center(bbox):
    minr, minc, maxr, maxc = bbox
    return  np.array([(minr + maxr) / 2,(minc + maxc) / 2 ])


def marker_size_from_corners(element):
    corners = element["corners"]
    if corners is None or len(corners) != 4:
        return None
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return np.mean(sides)


def get_triplet_score(centers,elements):
    A,B,C = centers
    angles = [angle(B, A, C), angle(A, B, C), angle(A, C, B)]
    idx = np.argmin(np.abs(np.array(angles) - 90))
    right_angle_dev = abs(angles[idx] - 90) / 90.0

    d = sorted([
        np.linalg.norm(A - B),
        np.linalg.norm(A - C),
        np.linalg.norm(B - C),
    ])
    side1, side2, diag = d

    sizes = [marker_size_from_corners(e) for e in elements]
    side_dev = abs(side1 - side2) / max(side1, side2)
    expected_diag = np.sqrt(2) * (side1 + side2) / 2
    diag_dev = abs(diag - expected_diag) / diag
    if all(s is not None for s in sizes):
        size_dev = (max(sizes) - min(sizes)) / max(sizes)
    else:
        size_dev = 0
    return right_angle_dev + side_dev + diag_dev + size_dev


def estimate_max_distance(elements,image_shape, factor=8):
    assert isinstance(image_shape, tuple), f"image_shape doit être un tuple, reçu {type(image_shape)}"
    if not elements:
        return  min(image_shape[:2]) * 0.5
    sizes = [((e["bbox"][2] - e["bbox"][0]) + (e["bbox"][3] - e["bbox"][1])) / 2 for e in elements]
    median_size = np.median(sizes)
    estimated = median_size * factor
    return  np.clip(estimated, 50, min(image_shape[:2]) * 0.8)




def polygon_area(corners):
    x = corners[:, 1]  # col
    y = corners[:, 0]  # row
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def clip_polygon(subject, clip_poly):
    def inside(p, a, b):
        return (b[1]-a[1])*(p[0]-a[0]) - (b[0]-a[0])*(p[1]-a[1]) >= 0

    def intersect(p1, p2, a, b):
        A1 = b[1]-a[1]
        B1 = a[0]-b[0]
        C1 = A1*a[0] + B1*a[1]
        A2 = p2[1]-p1[1]
        B2 = p1[0]-p2[0]
        C2 = A2*p1[0] + B2*p1[1]
        det = A1*B2 - A2*B1
        if abs(det) < 1e-10:
            return p2
        x = (B2*C1 - B1*C2) / det
        y = (A1*C2 - A2*C1) / det
        return np.array([x, y])

    output = list(subject)
    n_clip = len(clip_poly)
    for i in range(n_clip):
        a, b = clip_poly[i], clip_poly[(i+1) % n_clip]
        if not output:
            break
        input_list = output
        output = []
        n_in = len(input_list)
        for j in range(n_in):
            curr = input_list[j]
            prev = input_list[j-1]
            curr_in = inside(curr, a, b)
            prev_in = inside(prev, a, b)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr, a, b))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr, a, b))
    return np.array(output) if output else np.empty((0, 2))

def should_remove_smaller(outer_corners, inner_corners, size_tolerance=0.9, overlap_threshold=0.8):
    outer_area = polygon_area(outer_corners)
    inner_area = polygon_area(inner_corners)

    if inner_area == 0 or inner_area >= outer_area * size_tolerance:
        return False

    intersection = clip_polygon(inner_corners, outer_corners)
    if len(intersection) < 3:
        return False
    intersection_area = polygon_area(intersection)

    overlap_ratio = intersection_area / inner_area
    return overlap_ratio >= overlap_threshold

def filter_contained_triplets(triplets, size_tolerance=0.9, overlap_threshold=0.8):
    corners_list = [get_qr_corners(t) for t in triplets]

    keep = [True] * len(triplets)
    for i in range(len(triplets)):
        if not keep[i]:
            continue
        for j in range(len(triplets)):
            if i == j or not keep[j]:
                continue
            if should_remove_smaller(corners_list[i], corners_list[j], size_tolerance, overlap_threshold):
                keep[j] = False

    return [t for t, k in zip(triplets, keep) if k]


def get_triplets(elements,image_shape, max_distance = None,angle_tolerance=35):
    candidates = []

    n = len(elements)
    if max_distance is None:
        max_distance = estimate_max_distance(elements,image_shape)
    for i,j,k in combinations(range(n), 3):
        m1, m2, m3 = elements[i], elements[j], elements[k]

        if not similar_area(m1, m2, m3):
            continue
        centers = [ get_center(m1["bbox"]),get_center(m2["bbox"]),get_center(m3["bbox"])]

        d01 = np.linalg.norm(centers[0] - centers[1])
        d02 = np.linalg.norm(centers[0] - centers[2])
        d12 = np.linalg.norm(centers[1] - centers[2])

        #if max(d01, d02, d12) > max_distance:
        #    continue
        """
        if   check_right_angle(centers): # check_distances(centers) and
            triplets.append((m1, m2, m3))
        """
        angles = [angle(centers[1], centers[0], centers[2]),
                  angle(centers[0], centers[1], centers[2]),
                  angle(centers[0], centers[2], centers[1])]
        idx = np.argmin(np.abs(np.array(angles) - 90))
        if abs(angles[idx] - 90) > angle_tolerance:
            continue
        score = get_triplet_score(centers,(m1,m2,m3))
        candidates.append((score, (i, j, k), (m1, m2, m3)))
    candidates.sort(key=lambda c: c[0])

    used = set()
    triplets = []
    for score, idxs, markers in candidates:
        if any(idx in used for idx in idxs):
            continue
        used.update(idxs)
        triplets.append(markers)
    return triplets

def get_qr_corners(triplets, tolerance=15):
    A, B, C = triplets[0],triplets[1],triplets[2]
    centers = [get_center(A["bbox"]),get_center(B["bbox"]),get_center(C["bbox"])]

    angles = [angle(centers[1], centers[0], centers[2]), angle(centers[0], centers[1], centers[2]), angle(centers[0], centers[2], centers[1])]

    idx = np.argmin(np.abs(np.array(angles) - 90))

    if idx == 0:
        TL = A
        TR = B
        BL = C
    elif idx == 1:
        TL = B
        TR = A
        BL = C
    else:
        TL = C
        TR = A
        BL = B

    A = get_center(TL["bbox"])
    B = get_center(TR["bbox"])
    C = get_center(BL["bbox"])

    u = B - A
    u /= np.linalg.norm(u)

    v = C - A
    v /= np.linalg.norm(v)

    def finder_size(marker):
        minr, minc, maxr, maxc = marker["bbox"]
        return ((maxr - minr) + (maxc - minc)) / 2

    size = np.mean([finder_size(TL), finder_size(TR),finder_size(BL) ])
    marge = 1.2
    offset = marge * size / 2
    TL_corner = A - offset * u - offset * v
    TR_corner = B + offset * u - offset * v
    BL_corner = C - offset * u + offset * v

    BR_corner = TR_corner + BL_corner - TL_corner

    return np.array([TL_corner,TR_corner,BR_corner,BL_corner])




def draw_qr(image, qr_corners):
    img = image.copy()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    for i in range(4):
        p1 = qr_corners[i]
        p2 = qr_corners[(i+1)%4]
        rr, cc = ski.draw.line(int(p1[0]), int(p1[1]),int(p2[0]), int(p2[1]))
        mask = (rr >=0 ) & (rr < h) & (cc >= 0) & (cc < w)
        img[rr[mask], cc[mask]] = [255,0,0]
    return img
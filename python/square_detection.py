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
    if coords.shape[0] == 4:
        return coords
    if coords.shape[0] < 4:
        return coords

    best_subset = None
    best_score = float('inf')

    for subset in combinations(coords, 4):
        pts = np.array(subset)
        center = pts.mean(axis=0)

        # tri angulaire autour du centroïde pour avoir les points dans l'ordre du contour
        angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
        ordered = pts[np.argsort(angles)]

        # 4 côtés (consécutifs, en bouclant)
        sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
        # 2 diagonales
        diag1 = np.linalg.norm(ordered[0] - ordered[2])
        diag2 = np.linalg.norm(ordered[1] - ordered[3])

        # un carré a 4 côtés égaux et 2 diagonales égales entre elles
        side_std = np.std(sides)
        diag_diff = abs(diag1 - diag2)

        score = side_std + diag_diff

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

def check_form(coords:np.ndarray)->bool:
    return coords.shape[0] == 4  and check_angle(coords) # and check_longeur(coords)

def order_points(coords):
    coords = coords.astype(float)
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 0] - center[0], coords[:, 1] - center[1])
    return coords[np.argsort(angles)]

def square_filter(denoise:np.ndarray,binary:np.ndarray, region: object)->(bool,np.ndarray):
    if region.area < 50:
        return False,np.empty((0,2))
    minr, minc, maxr, maxc = region.bbox
    square1 = denoise[minr:maxr, minc:maxc]
    square2 = binary[minr:maxr, minc:maxc]
    #if not ski.measure.euler_number(square1, connectivity=1) < 3:
    #    return False,np.empty((0,2))

    # extraction of the square
    square_im = get_inner_square(square1)
    coords = get_corner(square_im)
    n_coords = coords.shape[0]
    if n_coords != 4 and n_coords != 5:
        return False, coords

    coords = get_four_coord(coords)
    coords = order_points(coords)
    return check_form(coords), coords



def check_area(a1, a2, ratio=1.3):
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

def get_triplets(elements):
    triplets = []
    for m1, m2, m3 in combinations(elements, 3):
        if not similar_area(m1, m2, m3):
            continue
        centers = [ get_center(m1["bbox"]),get_center(m2["bbox"]),get_center(m3["bbox"])]
        if   check_right_angle(centers): # check_distances(centers) and
            triplets.append((m1, m2, m3))
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

    # -------------------------------
    # Directions
    # -------------------------------

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
    for i in range(4):
        p1 = qr_corners[i]
        p2 = qr_corners[(i+1)%4]
        rr, cc = ski.draw.line(int(p1[0]), int(p1[1]),int(p2[0]), int(p2[1]))
        img[rr, cc] = [255,0,0]

    return img
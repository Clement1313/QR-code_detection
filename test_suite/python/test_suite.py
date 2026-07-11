import pytest
import numpy as np

from python.evaluation import precision, recall, f1_score, compute_iou
from python.square_detection import (
    check_area,
    similar_area,
    angle,
    get_center,
    check_right_angle,
    check_longeur,
    check_angle,
    filter_contained_triplets,
    marker_size_from_corners,
)


def test_metrics_main():
    """Test des fonctions de métriques"""
    assert precision(8, 2) == 0.8
    assert recall(8, 2) == 0.8
    assert f1_score(8, 2, 2) == pytest.approx(0.8)

    assert precision(0, 0) == 0
    assert recall(0, 0) == 0
    assert f1_score(0, 0, 0) == 0


def test_compute_iou_main():
    """Test du calcul de l'iou"""
    box1 = (0, 0, 10, 10)

    assert compute_iou(box1, box1) == 1.0

    box_disjoint = (20, 20, 30, 30)
    assert compute_iou(box1, box_disjoint) == 0.0

    box_overlap = (5, 5, 15, 15)
    assert pytest.approx(compute_iou(box1, box_overlap), 0.01) == 0.1428


def test_get_center():
    """Test de la méthode get_center (qui retourne le centre)"""
    bbox = (10, 20, 30, 40)  # (minr, minc, maxr, maxc)
    center = get_center(bbox)
    assert np.array_equal(center, np.array([20.0, 30.0]))


def test_check_area_and_similar():
    """Test des tolérances de surface pour les marqueurs"""
    assert check_area(100, 120, ratio=1.3) == True
    assert check_area(100, 150, ratio=1.3) == False

    m1 = {"area": 100}
    m2 = {"area": 110}
    m3 = {"area": 120}
    m_diff = {"area": 200}

    assert similar_area(m1, m2, m3) == True
    assert similar_area(m1, m2, m_diff) == False


def test_angle():
    """Test du calcul de l'angle entre 3 points"""
    a = np.array([0, 1], dtype=float)
    b = np.array([0, 1], dtype=float)
    b = np.array([0, 0], dtype=float)
    c = np.array([1, 0], dtype=float)

    ang = angle(a, b, c)
    assert pytest.approx(ang, 0.1) == 90.0


def test_check_right_angle():
    """Test de la détection d'angle droit pour un triplet de marqueurs"""
    centers = [
        np.array([0, 10], dtype=float),
        np.array([0, 0], dtype=float),
        np.array([10, 0], dtype=float),
    ]
    assert check_right_angle(centers, tolerance=5) == True

    centers_flat = [
        np.array([0, 10], dtype=float),
        np.array([0, 0], dtype=float),
        np.array([0, -10], dtype=float),
    ]
    assert check_right_angle(centers_flat, tolerance=15) == False


def test_check_longeur():
    """Test de la vérification de la longueur des côtés d'un carré"""
    c_square = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    assert check_longeur(c_square) == True

    c_rect = np.array([[0, 0], [0, 50], [10, 50], [10, 0]])
    assert check_longeur(c_rect) == False


def test_check_angle():
    """Test de la vérification des angles internes d'un quadrilatère"""
    c_square = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    assert check_angle(c_square) == True
    c_rhombus = np.array([[0, 0], [10, 5], [20, 0], [10, -5]])
    assert check_angle(c_rhombus) == False


def make_element(bbox, area=None):
    """Construit un élément avec bbox, corners carrés cohérents, et area."""
    minr, minc, maxr, maxc = bbox
    corners = np.array(
        [
            [minr, minc],
            [minr, maxc],
            [maxr, maxc],
            [maxr, minc],
        ],
        dtype=float,
    )
    if area is None:
        area = (maxr - minr) * (maxc - minc)
    return {"bbox": bbox, "corners": corners, "area": area}


def make_triplet(tl_bbox, tr_bbox, bl_bbox):
    return (make_element(tl_bbox), make_element(tr_bbox), make_element(bl_bbox))


class TestFilterContainedTriplets:

    def test_no_triplets_returns_empty(self):
        assert filter_contained_triplets([]) == []

    def test_single_triplet_is_kept(self):
        triplet = make_triplet((0, 0, 20, 20), (0, 180, 20, 200), (180, 0, 200, 20))
        result = filter_contained_triplets([triplet])
        assert len(result) == 1

    def test_smaller_triplet_inside_bigger_is_removed(self):
        big = make_triplet((0, 0, 20, 20), (0, 180, 20, 200), (180, 0, 200, 20))
        small = make_triplet((80, 80, 90, 90), (80, 120, 90, 130), (120, 80, 130, 90))

        result = filter_contained_triplets([big, small])

        assert len(result) == 1
        # on vérifie que c'est bien le grand triplet qui a survécu
        kept_sizes = [marker_size_from_corners(m) for m in result[0]]
        assert np.mean(kept_sizes) == pytest.approx(20.0)

    def test_order_independence(self):
        big = make_triplet((0, 0, 20, 20), (0, 180, 20, 200), (180, 0, 200, 20))
        small = make_triplet((80, 80, 90, 90), (80, 120, 90, 130), (120, 80, 130, 90))

        result_big_first = filter_contained_triplets([big, small])
        result_small_first = filter_contained_triplets([small, big])

        assert len(result_big_first) == 1
        assert len(result_small_first) == 1
        assert marker_size_from_corners(result_big_first[0][0]) == pytest.approx(
            marker_size_from_corners(result_small_first[0][0])
        )

    def test_two_separate_qr_codes_are_both_kept(self):
        qr1 = make_triplet((0, 0, 20, 20), (0, 180, 20, 200), (180, 0, 200, 20))
        qr2 = make_triplet(
            (500, 500, 520, 520), (500, 680, 520, 700), (680, 500, 700, 520)
        )

        result = filter_contained_triplets([qr1, qr2])
        assert len(result) == 2

    def test_multiple_nested_triplets_keeps_only_outer(self):
        """Trois triplets emboîtés (grand > moyen > petit) : seul le plus
        grand doit survivre."""
        big = make_triplet((0, 0, 20, 20), (0, 380, 20, 400), (380, 0, 400, 20))
        medium = make_triplet(
            (100, 100, 115, 115), (100, 250, 115, 265), (250, 100, 265, 115)
        )
        small = make_triplet(
            (150, 150, 158, 158), (150, 200, 158, 208), (200, 150, 208, 158)
        )

        result = filter_contained_triplets([big, medium, small])
        assert len(result) == 1
        assert marker_size_from_corners(result[0][0]) == pytest.approx(20.0)

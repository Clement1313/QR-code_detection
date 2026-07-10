import pytest
import numpy as np

# Import des fonctions depuis tes fichiers
from python.evaluation import precision, recall, f1_score, compute_iou
from python.square_detection import (
    check_area,
    similar_area,
    angle,
    get_center,
    check_right_angle,
    check_longeur,
    check_angle
)

def test_metrics_main():
    """Test des fonctions de métriques"""
    assert precision(8, 2) == 0.8
    assert recall(8, 2) == 0.8
    assert f1_score(8, 2, 2) == 0.8

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
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    ang = angle(a, b, c)
    assert pytest.approx(ang, 0.1) == 90.0


def test_check_right_angle():
    """Test de la détection d'angle droit pour un triplet de marqueurs"""
    centers = [
        np.array([0, 10]),
        np.array([0, 0]),
        np.array([10, 0])
    ]
    assert check_right_angle(centers, tolerance=5) == True

    centers_flat = [np.array([0, 10]), np.array([0, 0]), np.array([0, -10])]
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
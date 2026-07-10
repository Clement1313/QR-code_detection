import pytest
from pathlib import Path
from python.evaluation import evaluate_folder
DATASET_TEST_PATH = "../../data/Dataset/detection/monitor"

# Seuils minimums acceptables
MIN_PRECISION = 0.80
MIN_RECALL = 0.80
MIN_F1_SCORE = 0.80
MIN_MEAN_IOU = 0.70


def test_pipeline_performance():
    """
    Vérification de l'efficacité de la détection sur le dataset monitor
    """
    # Exécution de l'évaluation sur le dossier
    results = evaluate_folder(DATASET_TEST_PATH, iou_threshold=0.5, verbose=False)

    # Vérification que le dossier n'était pas vide
    assert results is not None, "L'évaluation n'a retourné aucun résultat."

    # Récupération des métriques
    precision = results.get("precision", 0)
    recall = results.get("recall", 0)
    f1 = results.get("f1", 0)
    mean_iou = results.get("mean_iou", 0)

    # Vérification des seuils
    assert precision >= MIN_PRECISION, f"Précision trop basse : {precision:.2f} < {MIN_PRECISION}"
    assert recall >= MIN_RECALL, f"Rappel trop bas : {recall:.2f} < {MIN_RECALL}"
    assert f1 >= MIN_F1_SCORE, f"Score F1 trop bas : {f1:.2f} < {MIN_F1_SCORE}"
    assert mean_iou >= MIN_MEAN_IOU, f"Mean IoU trop bas : {mean_iou:.2f} < {MIN_MEAN_IOU}"

    print(f"Test d'intégration réussi ! F1-Score: {f1:.2f}, IoU: {mean_iou:.2f}")
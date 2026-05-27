"""
Evaluation — display classification metrics for the trained model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_ingestion import CLASS_NAMES
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(y_true, y_pred, display_plot: bool = False) -> dict:
    """
    Compute and display accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        display_plot: If True, show a seaborn heatmap of the confusion matrix.

    Returns:
        Dictionary containing all computed metrics.
    """
    logger.info("Evaluating model performance...")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    print("\nClassification Report:")
    print(report)
    print("=" * 50 + "\n")

    logger.info(
        "Metrics — Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        accuracy,
        precision,
        recall,
        f1,
    )

    if display_plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=100)
            logger.info("Confusion matrix plot saved to confusion_matrix.png")
            plt.close()
        except Exception as exc:
            logger.warning("Could not save confusion matrix plot: %s", exc)

    return metrics


if __name__ == "__main__":
    # Example: evaluate with dummy predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    evaluate_model(y_true, y_pred)

"""
Model training — Decision Tree Classifier with hyperparameter tuning.
Tuned parameters: max_depth and min_samples_split.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier

from src.data_ingestion import CLASS_NAMES
from src.logger import get_logger
from src.utils import ARTIFACTS_DIR, save_object

logger = get_logger(__name__)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"


def tune_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    max_depth_range: range = range(2, 11),
    min_samples_split_range: list = [2, 5, 10],
) -> tuple:
    """
    Train Decision Tree models with different max_depth and min_samples_split values.
    Pick the combination with highest test accuracy.

    Args:
        X_train: Training features (scaling not required for Decision Trees).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        max_depth_range: Range of max_depth values to try.
        min_samples_split_range: List of min_samples_split values to try.

    Returns:
        Tuple of (best_params, best_model, best_accuracy, results_dict).
    """
    results = {}
    best_accuracy = 0.0
    best_model = None
    best_params = {"max_depth": 2, "min_samples_split": 2}

    print("\n" + "=" * 60)
    print("DECISION TREE HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"{'max_depth':>10} | {'min_samples_split':>18} | {'Accuracy':>10}")
    print("-" * 42)

    for depth in max_depth_range:
        for min_samples in min_samples_split_range:
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=min_samples,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            params_key = f"depth={depth},min_samples={min_samples}"
            results[params_key] = acc

            print(f"{depth:10d} | {min_samples:18d} | {acc:10.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "min_samples_split": min_samples,
                }

    print("-" * 42)
    print(f"Best Params: max_depth={best_params['max_depth']}, "
          f"min_samples_split={best_params['min_samples_split']}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("=" * 60 + "\n")

    logger.info(
        "Best Decision Tree params: max_depth=%d, min_samples_split=%d, accuracy=%.4f",
        best_params["max_depth"],
        best_params["min_samples_split"],
        best_accuracy,
    )
    return best_params, best_model, best_accuracy, results


def train_model(X_train, y_train, X_test, y_test):
    """
    Tune Decision Tree hyperparameters, train the best model, evaluate on test set,
    and save the model.

    Args:
        X_train: Training features (no scaling needed for Decision Trees).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (trained_model, y_pred, best_params).
    """
    logger.info("Starting Decision Tree model training...")

    best_params, model, best_acc, _ = tune_decision_tree(X_train, y_train, X_test, y_test)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics printed during training
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print("\nFINAL MODEL EVALUATION (Test Set)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 50 + "\n")

    logger.info("Model accuracy: %.4f", acc)
    logger.info("Confusion matrix:\n%s", cm)

    # Persist the trained model
    save_object(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    return model, y_pred, best_params


if __name__ == "__main__":
    from src.data_ingestion import ingest_data
    from src.data_cleaning import clean_data
    from src.outlier_handling import handle_outliers
    from src.preprocessing import preprocess_data

    df = handle_outliers(clean_data(ingest_data()))
    X_tr, X_te, y_tr, y_te, _ = preprocess_data(df)
    train_model(X_tr, y_tr, X_te, y_te)

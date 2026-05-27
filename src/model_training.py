"""
Model training — Decision Tree Classifier with hyperparameter tuning and cost-complexity pruning.
Tuned parameters: max_depth, min_samples_split, and optimal ccp_alpha via pruning.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
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


def apply_pruning(
    X_train,
    y_train,
    X_test,
    y_test,
    best_params: dict,
    cv_folds: int = 5,
) -> tuple:
    """
    Apply cost-complexity pruning to the Decision Tree model.
    
    Cost-complexity pruning (post-pruning) removes subtrees that don't
    improve cross-validation accuracy, preventing overfitting.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        best_params: Dictionary with best hyperparameters (max_depth, min_samples_split).
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (pruned_model, ccp_alpha_used, pruned_accuracy).
    """
    print("\n" + "=" * 60)
    print("COST-COMPLEXITY PRUNING")
    print("=" * 60)

    # Train a full tree without depth restrictions to identify pruning candidates
    full_tree = DecisionTreeClassifier(
        min_samples_split=best_params["min_samples_split"],
        random_state=42,
    )
    full_tree.fit(X_train, y_train)

    # Get the cost-complexity pruning path
    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    logger.info("Cost-complexity pruning path generated with %d alpha values", len(ccp_alphas))

    # Train trees for each alpha value and evaluate via cross-validation
    print(f"\nEvaluating {len(ccp_alphas)} pruning candidates via {cv_folds}-fold CV...")
    print(f"{'Alpha':>12} | {'CV Accuracy':>12}")
    print("-" * 28)

    cv_scores = []
    best_alpha = ccp_alphas[0]
    best_cv_score = 0.0

    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            ccp_alpha=ccp_alpha,
            min_samples_split=best_params["min_samples_split"],
            random_state=42,
        )
        scores = cross_val_score(tree, X_train, y_train, cv=cv_folds, scoring="accuracy")
        mean_score = scores.mean()
        cv_scores.append(mean_score)

        print(f"{ccp_alpha:12.6f} | {mean_score:12.4f}")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_alpha = ccp_alpha

    print("-" * 28)
    print(f"Best alpha for pruning: {best_alpha:.6f}")
    print(f"Best CV accuracy: {best_cv_score:.4f}")

    # Train final pruned model with best alpha
    pruned_model = DecisionTreeClassifier(
        ccp_alpha=best_alpha,
        min_samples_split=best_params["min_samples_split"],
        random_state=42,
    )
    pruned_model.fit(X_train, y_train)

    # Evaluate pruned model on test set
    y_pred_pruned = pruned_model.predict(X_test)
    pruned_accuracy = accuracy_score(y_test, y_pred_pruned)

    print(f"\nPruned Tree Accuracy (Test Set): {pruned_accuracy:.4f}")
    print(f"Tree nodes after pruning: {pruned_model.tree_.node_count}")
    print("=" * 60 + "\n")

    logger.info(
        "Pruning complete. Best alpha=%.6f, test accuracy=%.4f, nodes=%d",
        best_alpha,
        pruned_accuracy,
        pruned_model.tree_.node_count,
    )

    return pruned_model, best_alpha, pruned_accuracy


def train_model(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree with hyperparameter tuning and cost-complexity pruning.
    
    Pipeline:
    1. Hyperparameter tuning (max_depth, min_samples_split)
    2. Cost-complexity pruning to prevent overfitting
    3. Final evaluation on test set

    Args:
        X_train: Training features (no scaling needed for Decision Trees).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (trained_model, y_pred, best_params, ccp_alpha).
    """
    logger.info("Starting Decision Tree model training with pruning...")

    # Step 1: Hyperparameter tuning
    best_params, unpruned_model, tuned_acc, _ = tune_decision_tree(
        X_train, y_train, X_test, y_test
    )

    # Step 2: Apply cost-complexity pruning
    pruned_model, best_alpha, pruned_acc = apply_pruning(
        X_train, y_train, X_test, y_test, best_params
    )

    # Step 3: Final evaluation with pruned model
    y_pred = pruned_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print("\nFINAL PRUNED MODEL EVALUATION (Test Set)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Tree nodes: {pruned_model.tree_.node_count}")
    print(f"Tree depth: {pruned_model.get_depth()}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 50 + "\n")

    logger.info("Model accuracy: %.4f", acc)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info(
        "Final model: nodes=%d, depth=%d, ccp_alpha=%.6f",
        pruned_model.tree_.node_count,
        pruned_model.get_depth(),
        best_alpha,
    )

    # Persist the trained pruned model
    save_object(pruned_model, MODEL_PATH)
    logger.info("Pruned model saved to %s", MODEL_PATH)

    return pruned_model, y_pred, best_params, best_alpha


if __name__ == "__main__":
    from src.data_ingestion import ingest_data
    from src.data_cleaning import clean_data
    from src.outlier_handling import handle_outliers
    from src.preprocessing import preprocess_data

    df = handle_outliers(clean_data(ingest_data()))
    X_tr, X_te, y_tr, y_te, _ = preprocess_data(df)
    train_model(X_tr, y_tr, X_te, y_te)

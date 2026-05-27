"""
Main entry point — runs the full end-to-end ML pipeline for Iris Decision Tree classification.

Pipeline steps:
  1. Data Ingestion
  2. Data Cleaning
  3. Outlier Handling
  4. Preprocessing
  5. Model Training (Decision Tree with hyperparameter tuning)
  6. Evaluation
  7. Sample Prediction
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import ingest_data
from src.data_cleaning import clean_data
from src.outlier_handling import handle_outliers
from src.preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.prediction import predict_flower
from src.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    """Execute all pipeline stages sequentially."""
    print("\n" + "#" * 60)
    print("  IRIS DECISION TREE CLASSIFICATION — END-TO-END PIPELINE")
    print("#" * 60)

    # Step 1: Load Iris dataset
    print("\n>>> STEP 1: Data Ingestion")
    df = ingest_data()

    # Step 2: Clean data (missing values + duplicates)
    print("\n>>> STEP 2: Data Cleaning")
    df = clean_data(df)

    # Step 3: Cap outliers using IQR
    print("\n>>> STEP 3: Outlier Handling")
    df = handle_outliers(df)

    # Step 4: Scale features and split train/test
    print("\n>>> STEP 4: Preprocessing")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    # Step 5: Train Decision Tree with hyperparameter tuning and pruning
    print("\n>>> STEP 5: Model Training (Decision Tree with Pruning)")
    model, y_pred, best_params, ccp_alpha = train_model(X_train, y_train, X_test, y_test)

    # Step 6: Detailed evaluation metrics
    print("\n>>> STEP 6: Evaluation")
    evaluate_model(y_test, y_pred)

    # Step 7: Demo prediction on a sample flower
    print("\n>>> STEP 7: Sample Prediction")
    sample = predict_flower(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    print(f"Sample input -> Predicted species: {sample}")

    print("\n" + "#" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Best Params: max_depth={best_params['max_depth']}, "
          f"min_samples_split={best_params['min_samples_split']}")
    print(f"  Pruning Alpha (ccp_alpha): {ccp_alpha:.6f}")
    print("  Artifacts saved in: artifacts/")
    print("  Run the app: streamlit run app/app.py")
    print("#" * 60 + "\n")

    logger.info(
        "Pipeline finished successfully. Best params: max_depth=%d, min_samples_split=%d, ccp_alpha=%.6f",
        best_params["max_depth"],
        best_params["min_samples_split"],
        ccp_alpha,
    )


if __name__ == "__main__":
    run_pipeline()

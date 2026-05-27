"""
Preprocessing — train-test split with stratification.
Note: Decision Trees do NOT require feature scaling, but we keep StandardScaler
for consistency and in case other models are used in the future.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import get_logger
from src.utils import ARTIFACTS_DIR, save_object

logger = get_logger(__name__)

PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_data.csv"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "encoder.pkl"

# Feature columns used for model training (Iris measurements)
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Encode target labels, split into train/test sets (with stratification),
    optionally scale features, and save processed CSV and artifacts.

    Note: Decision Trees are invariant to feature scaling, so we scale
    for consistency with other ML algorithms and ease of reuse.

    Args:
        df: DataFrame after cleaning and outlier handling.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_columns).
    """
    logger.info("Starting preprocessing...")

    df = df.copy()

    # Save fully processed dataset before splitting
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Processed data saved to %s", PROCESSED_DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    # Label encoder maps numeric target to class index (useful for inverse transform)
    label_encoder = LabelEncoder()
    label_encoder.fit(["Setosa", "Versicolor", "Virginica"])
    save_object(label_encoder, ENCODER_PATH)

    # Train-test split (stratify keeps class proportions balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # StandardScaler: zero mean, unit variance
    # While Decision Trees don't require scaling, we keep it for consistency.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_object(scaler, SCALER_PATH)
    logger.info(
        "Preprocessing complete. Train size: %d, Test size: %d",
        len(X_train_scaled),
        len(X_test_scaled),
    )

    print(f"\nTrain set size: {len(X_train_scaled)}")
    print(f"Test set size:  {len(X_test_scaled)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_COLUMNS


if __name__ == "__main__":
    from src.data_ingestion import ingest_data
    from src.data_cleaning import clean_data
    from src.outlier_handling import handle_outliers

    data = handle_outliers(clean_data(ingest_data()))
    preprocess_data(data)

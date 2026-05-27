"""
Data ingestion — load the built-in Iris dataset and save it as a CSV file.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_PATH = ARTIFACTS_DIR / "iris_raw.csv"

# Human-readable class labels mapped from numeric target (0, 1, 2)
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]


def ingest_data() -> pd.DataFrame:
    """
    Load the Iris dataset from scikit-learn, convert to a DataFrame,
    add target and species columns, save to CSV, and return the DataFrame.

    Returns:
        pandas DataFrame with feature columns, 'target', and 'species'.
    """
    logger.info("Loading Iris dataset from scikit-learn...")

    # load_iris() returns a Bunch with data, target, feature_names, target_names
    iris = load_iris()

    # Build DataFrame from feature matrix and column names
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Target column (numeric: 0, 1, 2)
    df["target"] = iris.target

    # Map numeric target to readable class names
    df["species"] = df["target"].map(
        {i: name for i, name in enumerate(CLASS_NAMES)}
    )

    # Ensure artifacts folder exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Raw dataset saved to %s", RAW_DATA_PATH)

    # Display basic dataset information
    print("\n" + "=" * 50)
    print("DATASET SHAPE:", df.shape)
    print("=" * 50)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df["species"].value_counts())
    print("=" * 50 + "\n")

    logger.info("Data ingestion complete. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    ingest_data()

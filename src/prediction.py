"""
Prediction pipeline — load saved artifacts and predict flower species.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion import CLASS_NAMES
from src.logger import get_logger
from src.preprocessing import FEATURE_COLUMNS
from src.utils import ARTIFACTS_DIR, load_object

logger = get_logger(__name__)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


class IrisPredictor:
    """
    End-to-end predictor that loads the trained KNN model and scaler,
    then returns the predicted flower class for new measurements.
    """

    def __init__(self):
        """Load model and scaler from the artifacts directory."""
        self.model = load_object(MODEL_PATH)
        self.scaler = load_object(SCALER_PATH)
        logger.info("IrisPredictor initialized successfully.")

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> str:
        """
        Predict the iris species from four flower measurements.

        Args:
            sepal_length: Sepal length in cm.
            sepal_width:  Sepal width in cm.
            petal_length: Petal length in cm.
            petal_width:  Petal width in cm.

        Returns:
            Predicted class name: 'Setosa', 'Versicolor', or 'Virginica'.
        """
        # Build a single-row DataFrame with the same column names used during training
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURE_COLUMNS,
        )

        # Scale features using the fitted scaler
        scaled = self.scaler.transform(input_data)

        # Predict numeric class (0, 1, or 2)
        prediction = self.model.predict(scaled)[0]
        species = CLASS_NAMES[int(prediction)]

        logger.info(
            "Prediction - sepal L=%.2f, W=%.2f, petal L=%.2f, W=%.2f -> %s",
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            species,
        )
        return species


def predict_flower(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    """
    Convenience function — create a predictor and return the species name.

    Args:
        sepal_length, sepal_width, petal_length, petal_width: Measurements in cm.

    Returns:
        Predicted species name.
    """
    predictor = IrisPredictor()
    return predictor.predict(sepal_length, sepal_width, petal_length, petal_width)


if __name__ == "__main__":
    # Example prediction (typical Setosa measurements)
    result = predict_flower(5.1, 3.5, 1.4, 0.2)
    print(f"\nPredicted species: {result}")

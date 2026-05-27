"""
Utility functions — reusable helpers for saving and loading Python objects.
"""

from pathlib import Path

import joblib

from src.logger import get_logger

logger = get_logger(__name__)

# Default directory for all serialized artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def save_object(obj, file_path: Path) -> None:
    """
    Serialize a Python object to disk using joblib.

    Args:
        obj: Any picklable Python object (model, scaler, encoder, etc.).
        file_path: Destination path for the .pkl file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info("Saved object to %s", file_path)


def load_object(file_path: Path):
    """
    Deserialize a Python object from disk using joblib.

    Args:
        file_path: Path to the .pkl file.

    Returns:
        The deserialized Python object.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    obj = joblib.load(file_path)
    logger.info("Loaded object from %s", file_path)
    return obj

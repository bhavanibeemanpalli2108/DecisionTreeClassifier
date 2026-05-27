"""
Outlier handling — detect and cap outliers using the IQR (Interquartile Range) method.
"""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def cap_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Cap (floor/ceiling) outliers in numerical columns using the IQR method.

    Formula:
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR

    Values below the lower bound are set to the lower bound.
    Values above the upper bound are set to the upper bound.

    Args:
        df: Input DataFrame.
        columns: List of numerical column names. If None, all numeric columns are used.

    Returns:
        DataFrame with outliers capped.
    """
    df = df.copy()

    if columns is None:
        # Only feature columns (exclude target and species)
        columns = df.select_dtypes(include="number").columns.tolist()
        columns = [c for c in columns if c not in ("target",)]

    outliers_capped = 0

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count values that will be capped
        below = (df[col] < lower_bound).sum()
        above = (df[col] > upper_bound).sum()
        outliers_capped += below + above

        # Cap outliers to bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(
            "Column '%s': bounds [%.4f, %.4f], capped %d outlier(s)",
            col,
            lower_bound,
            upper_bound,
            below + above,
        )

    print(f"\nTotal outlier values capped: {outliers_capped}")
    logger.info("Outlier handling complete. Total values capped: %d", outliers_capped)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry point for the outlier handling step.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with outliers treated.
    """
    logger.info("Starting outlier handling (IQR method)...")
    return cap_outliers_iqr(df)


if __name__ == "__main__":
    from src.data_ingestion import ingest_data
    from src.data_cleaning import clean_data

    df = handle_outliers(clean_data(ingest_data()))
    print(df.describe())

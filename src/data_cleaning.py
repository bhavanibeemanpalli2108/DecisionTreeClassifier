"""
Data cleaning — handle missing values and remove duplicate rows.
"""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numerical columns → median
      - Categorical columns → mode (most frequent value)

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values imputed.
    """
    df = df.copy()
    missing_count = df.isnull().sum().sum()

    if missing_count == 0:
        logger.info("No missing values found in the dataset.")
        return df

    logger.info("Found %d missing value(s). Imputing...", missing_count)

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Numerical → median
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            logger.info("Filled missing values in '%s' with median: %s", col, fill_value)
        else:
            # Categorical → mode
            fill_value = df[col].mode()[0]
            df[col].fillna(fill_value, inplace=True)
            logger.info("Filled missing values in '%s' with mode: %s", col, fill_value)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows and report how many were removed.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without duplicate rows.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    duplicates_removed = before - after

    print(f"\nDuplicates removed: {duplicates_removed}")
    logger.info("Duplicates removed: %d (rows before: %d, after: %d)", duplicates_removed, before, after)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full data cleaning pipeline: missing value handling + deduplication.

    Args:
        df: Raw DataFrame from ingestion.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning...")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    logger.info("Data cleaning complete.")
    return df


if __name__ == "__main__":
    from src.data_ingestion import ingest_data

    raw_df = ingest_data()
    cleaned_df = clean_data(raw_df)
    print(cleaned_df.head())

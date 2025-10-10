"""Feature engineering nodes."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import holidays
import logging

logger = logging.getLogger(__name__)


def add_holiday_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Add day_type column based on date(YYYYMMDD).
    - Weekends (Sat/Sun) or public holidays → 'holiday'
    - Others → 'workday'
    - Failed date parsing (NaT) → 'workday' (conservative approach)

    Pipeline Order: Called within build_features
    Role: Feature engineering - adds day_type only

    Args:
        df: Input DataFrame
        date_column: Date column name

    Returns:
        DataFrame with day_type column added

    Example:
        Input: date='20240101' (New Year)
        Output: day_type='holiday'
    """
    if date_column not in df:
        return df

    # 1) YYYYMMDD → datetime
    s = pd.to_datetime(df[date_column].astype(str), format="%Y%m%d", errors="coerce")

    # 2) Get unique years and create Korean holidays object (weekends not included)
    years = s.dt.year.dropna().astype(int).unique()
    kr = holidays.KR(years=years)

    # 3) Check for off days: weekends or public holidays
    is_weekend = s.dt.dayofweek.isin([5, 6])
    is_public_holiday = s.dt.date.isin(set(kr.keys()))
    is_off = is_weekend | is_public_holiday

    # 4) Add day_type to original df (NaT → workday)
    df.loc[:, "day_type"] = np.where(is_off, "holiday", "workday")

    # Statistics logging
    if 'day_type' in df.columns:
        counts = df['day_type'].value_counts()
        logger.info(f"Day type distribution: {counts.to_dict()}")

    return df


def build_features(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Simplified feature engineering - only holiday features.

    Pipeline Order: 8th step (Feature pipeline, 1st node)
    Role: Feature creation - adds derived features

    Args:
        data: Input DataFrame from validated_data
        params: Feature engineering parameters including date_column

    Returns:
        DataFrame with engineered features (adds 1 column: day_type)

    Example:
        Adds holiday features based on Korean calendar
    """
    logger.info("Building features - holiday features only")

    # Get date column name from parameters
    date_column = params.get('date_column', 'date')

    # Add holiday features
    data = add_holiday_features(data, date_column)

    logger.info(f"Features built. Shape: {data.shape}")

    return data


def select_features(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Select specific features and label columns, then prepare for Dataset conversion.

    Pipeline Order: 9th step (Feature pipeline, final node)
    Role: Feature selection + data cleaning for downstream processing

    Args:
        data: Input DataFrame from build_features
        params: Feature selection parameters containing:
            - features: List of 24 feature column names in desired order
            - label: Label column name (acct_code)

    Returns:
        DataFrame with selected features and label in specified order,
        cleaned of nulls in label column and duplicates

    Example:
        Input: 56 columns, 10000 rows
        Output: 25 columns (24 features + label), ~9900 rows (after cleaning)
    """
    # Get features and label from params (no defaults)
    features = params['features']
    label = params['label']

    # Build ordered column list - maintain order from config
    ordered_columns = []

    # Add feature columns in the order specified
    for feature in features:
        if feature in data.columns:
            if feature not in ordered_columns:
                ordered_columns.append(feature)
        else:
            logger.warning(f"Feature column '{feature}' not found in data")

    # Add label column at the end
    if label in data.columns:
        if label not in ordered_columns:
            ordered_columns.append(label)
    else:
        raise ValueError(f"Label column '{label}' is required but not found in data")

    # Select the columns in specified order
    if not ordered_columns:
        raise ValueError("No columns to select")

    selected_data = data[ordered_columns]
    logger.info(f"Selected {len(ordered_columns)} columns in order")
    logger.info(f"Features: {len([c for c in ordered_columns if c != label])}, Label: {label}")

    # Clean data: remove nulls in label and duplicates (previously in prepare_dataset_inputs)
    initial_rows = len(selected_data)
    base_table = selected_data.dropna(subset=[label]).drop_duplicates().reset_index(drop=True)

    logger.info(f"Data cleaning: {initial_rows} → {len(base_table)} rows")
    logger.info(f"Unique labels: {base_table[label].nunique()}")

    return base_table


def filter_minority_classes(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove label classes whose sample count does not exceed the configured minimum.

    Args:
        data: DataFrame produced by ``select_features``.
        params: Feature selection parameters containing:
            - label: Label column name
            - min_label_count: Classes with <= this many samples are dropped

    Returns:
        DataFrame filtered to majority classes, index reset for downstream Datasets.
    """
    label_col = params.get("label")
    if not label_col:
        raise ValueError("Feature selection parameters must include a 'label' entry.")
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not present in base table.")

    min_count = int(params.get("min_label_count", 0))
    if min_count <= 0:
        logger.info("min_label_count <= 0; skipping minority class filtering.")
        return data

    label_counts = data[label_col].value_counts()
    keep_labels = label_counts[label_counts > min_count].index
    removed_labels = label_counts[label_counts <= min_count]

    filtered = data[data[label_col].isin(keep_labels)].reset_index(drop=True)

    logger.info(
        "Minority class filtering applied: kept %s classes (> %s samples), removed %s rows.",
        len(keep_labels),
        min_count,
        int(removed_labels.sum()),
    )
    if not keep_labels.empty:
        logger.debug("Remaining classes after filtering: %s", list(keep_labels))
    if not removed_labels.empty:
        logger.info(
            "Removed classes (label -> count): %s",
            removed_labels.to_dict(),
        )

    remaining_classes = filtered[label_col].nunique()
    if remaining_classes < 2:
        raise ValueError(
            f"Only {remaining_classes} label class(es) remaining after filtering; "
            "adjust 'min_label_count' to keep at least two classes."
        )

    return filtered

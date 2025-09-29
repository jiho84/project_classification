"""Preprocessing - using pandas directly."""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, clean_params: Dict[str, Any]) -> pd.DataFrame:
    """Clean data by removing duplicates and rows with missing values in specified columns.

    Pipeline Order: Preprocess step 1 (1st node)
    Role: Data quality control - removes invalid records

    Args:
        data: Input DataFrame from standardized_data
        clean_params: Cleaning parameters
            - remove_duplicates: Whether to remove duplicate rows (default: True)
            - dropna_columns: List of columns where nulls should cause row removal

    Returns:
        DataFrame with cleaned data

    Example:
        Input: DataFrame with 1000 rows, some with null party_vat_id
        Output: DataFrame with ~950 rows (nulls in party_vat_id removed)
    """
    initial_rows = len(data)

    # Remove duplicates if specified
    if clean_params.get('remove_duplicates', True):
        data = data.drop_duplicates()
        duplicates_removed = initial_rows - len(data)
        logger.info(f"Removed {duplicates_removed} duplicate rows")

    # Remove rows with nulls in specified columns
    dropna_columns = clean_params.get('dropna_columns', [])
    if dropna_columns:
        # Keep only columns that exist in data
        existing_columns = [col for col in dropna_columns if col in data.columns]
        if existing_columns:
            before_dropna = len(data)
            data = data.dropna(subset=existing_columns)
            nulls_removed = before_dropna - len(data)
            logger.info(f"Removed {nulls_removed} rows with nulls in columns: {existing_columns}")
        else:
            logger.warning(f"None of specified columns {dropna_columns} found in data")

    logger.info(f"Data cleaning complete: {initial_rows} -> {len(data)} rows")
    return data


def normalize_value(
    df: pd.DataFrame,
    code_mappings: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Normalize code values to human-readable text with optimized performance.

    Pipeline Order: Preprocess step 3 (3rd node)
    Role: Data transformation - converts codes to meaningful text

    Args:
        df: DataFrame with code values from cleaned_data
        code_mappings: Optional dictionary containing mappings for each column
                      Format: {'column_name': {code: 'text_value', ...}, ...}
                      Default mappings for: vat_message_code, document_type, party_type_code

    Returns:
        DataFrame with normalized text values (modified in-place for efficiency)

    Example:
        Input: vat_message_code='1', document_type='10'
        Output: vat_message_code='standard', document_type='tax_invoice'
    """
    default_mappings = {
        "vat_message_code": {
            "1": "standard", "2": "simplified", "3": "special", "4": "exempt",
            "5": "nonprofit", "6": "nonvat", "7": "simplified_issuer",
            None: "unregistered", "None": "unregistered", "nan": "unregistered", "": "unregistered",
        },
        "company_vat_message_code": {
            "1": "standard", "2": "simplified", "3": "special", "4": "exempt",
            "5": "nonprofit", "6": "nonvat", "7": "simplified_issuer",
            None: "unregistered", "None": "unregistered", "nan": "unregistered", "": "unregistered",
        },
        "document_type": {
            "10": "tax_invoice", "11": "zero_invoice", "12": "exempt_invoice", "13": "import_invoice",
            "20": "credit_card", "21": "credit_card", "22": "credit_card",
            "30": "cash_receipt", "31": "cash_receipt", "32": "cash_receipt",
        },
        "party_type_code": {
            "11": "general_sale", "12": "general_purchase", "13": "general_both",
            "21": "financial_demand_deposit", "22": "financial_checking", "23": "financial_installment_savings",
            "24": "financial_time_deposit", "25": "financial_other", "26": "financial_foreign_currency",
            "31": "card_sale", "32": "card_purchase",
        },
    }
    mappings = {**default_mappings, **(code_mappings or {})}

    for col, mapping in mappings.items():
        if col not in df.columns:
            continue

        # Process only the column (no full DF copy)
        s_before = df[col].astype(str)  # String view/copy of original column
        s_after = s_before.replace(mapping)  # Replacement result for column

        # Calculate statistics
        diff = s_after != s_before
        mapped_count = int(diff.values.sum())
        total_count = len(s_before)
        try:
            logger.info(
                "[%s] mapped %s/%s (%.2f%%)",
                col,
                mapped_count,
                total_count,
                (mapped_count / total_count * 100) if total_count else 0,
            )
        except (NameError, ZeroDivisionError):
            pass  # Skip if logger not available or division by zero

        # Overwrite in original DF (in-place)
        df[col] = s_after

    return df



def filter_data(data: pd.DataFrame, filter_params: Dict[str, Any]) -> pd.DataFrame:
    """Filter data by excluding columns.

    Pipeline Order: Preprocess step 2 (2nd node)
    Role: Data filtering - removes unwanted columns

    Args:
        data: Input DataFrame from cleaned_data
        filter_params: Filtering parameters
            - exclude_columns: List of columns to exclude

    Returns:
        Filtered DataFrame

    YML Configuration Example:
        preprocess:
          filter:
            exclude_columns: [id, created_at, batch_no]
    """
    initial_cols = len(data.columns)

    # Exclude columns
    exclude_columns = filter_params.get('exclude_columns', [])
    if exclude_columns:
        data = data.drop(columns=[col for col in exclude_columns if col in data.columns])
        logger.info(f"Excluded {initial_cols - len(data.columns)} columns")

    logger.info(f"Filtering complete: {data.shape}")
    return data


def validate_data(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Validate and filter data based on business rules.

    Pipeline Order: Preprocess step 4 (4th node)
    Role: Business rule validation - ensures data quality

    Args:
        data: DataFrame from normalized_data
        params: Validation parameters including min/max amounts

    Returns:
        DataFrame with validated data

    Example:
        Filters out invalid amounts and dates
    """

    # Filter by conditions if provided
    if "min_amount" in params and "amount" in data.columns:
        data = data[data["amount"] >= params["min_amount"]]

    if "max_amount" in params and "amount" in data.columns:
        data = data[data["amount"] <= params["max_amount"]]

    # Remove invalid dates if date column exists
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors='coerce')
        data = data.dropna(subset=["date"])

    return data


def normalize_missing_values(
    data: pd.DataFrame,
    placeholders: Dict[str, Any],
) -> pd.DataFrame:
    """Standardise missing values ahead of text serialisation.

    Args:
        data: DataFrame produced by the validation step.
        placeholders: Mapping of placeholder values per dtype category. Supported keys:
            - ``categorical`` (default ``"__missing__"``)
            - ``numeric`` (default ``0``)
            - ``datetime`` (default reuse categorical placeholder)
            - ``boolean`` (default ``False``)
            - ``strip_whitespace`` (bool, default ``True``)

    Returns:
        DataFrame with consistent placeholder values applied.
    """

    categorical_placeholder = placeholders.get("categorical", "__missing__")
    numeric_placeholder = placeholders.get("numeric", 0)
    datetime_placeholder = placeholders.get("datetime", categorical_placeholder)
    boolean_placeholder = placeholders.get("boolean", False)
    strip_whitespace = placeholders.get("strip_whitespace", True)

    result = data.copy()

    # Object / string columns
    object_columns = result.select_dtypes(include=["object", "string"]).columns
    for col in object_columns:
        before = result[col].isna().sum()
        result[col] = result[col].fillna(categorical_placeholder)
        if strip_whitespace:
            result[col] = result[col].replace(r"^\s*$", categorical_placeholder, regex=True)
        after = (result[col] == categorical_placeholder).sum()
        if before or after:
            logger.debug("Column '%s': replaced %s missing/object-empty entries", col, after)

    # Numeric columns
    numeric_columns = result.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        missing_count = result[col].isna().sum()
        if missing_count:
            result[col] = result[col].fillna(numeric_placeholder)
            logger.debug("Column '%s': filled %s numeric NaNs with %s", col, missing_count, numeric_placeholder)

    # Boolean columns
    bool_columns = result.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        missing_count = result[col].isna().sum()
        if missing_count:
            result[col] = result[col].fillna(boolean_placeholder)
            logger.debug("Column '%s': filled %s boolean NaNs with %s", col, missing_count, boolean_placeholder)

    # Datetime columns
    datetime_columns = list(result.select_dtypes(include=["datetime64[ns]"]).columns)
    datetime_columns += list(result.select_dtypes(include=["datetimetz"]).columns)
    if datetime_columns:
        datetime_placeholder_ts = pd.to_datetime(datetime_placeholder, errors="coerce")
        for col in datetime_columns:
            missing_count = result[col].isna().sum()
            if missing_count:
                result[col] = result[col].fillna(datetime_placeholder_ts)
                logger.debug(
                    "Column '%s': filled %s datetime NaNs with %s",
                    col,
                    missing_count,
                    datetime_placeholder_ts,
                )

    return result

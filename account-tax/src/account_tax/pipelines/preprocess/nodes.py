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
        "vat_message_code_company": {
            "1": "standard", "2": "simplified", "3": "special", "4": "exempt",
            "5": "nonprofit", "6": "nonvat", "7": "simplified_issuer",
            None: "unregistered", "None": "unregistered", "nan": "unregistered", "": "unregistered",
        },
        "document_type": {
            "10": "tax_invoice", "11": "zero_invoice", "12": "exempt_invoice", "13": "import_invoice",
            "20": "credit_card", "21": "credit_card", "22": "credit_card",
            "30": "cash_receipt", "31": "cash_receipt", "32": "cash_receipt",
        },
        "party_type": {
            "11": "general_sale", "12": "general_purchase", "13": "general_both",
            "21": "financial_demand_deposit", "22": "financial_checking", "23": "financial_installment_savings",
            "24": "financial_time_deposit", "25": "financial_other", "26": "financial_foreign_currency",
            "31": "card_sale", "32": "card_purchase",
        },
        "transaction_code": {
            "1": "sale",
            "2": "purchase",
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
    """Filter data by rows and columns based on configuration.

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
    initial_rows = len(data)
    initial_cols = len(data.columns)

    keep_rules = filter_params.get('keep_rows', []) or []
    for rule in keep_rules:
        column = rule.get('column')
        values = rule.get('values')
        if not column or column not in data.columns or not values:
            continue
        before = len(data)
        data = data[data[column].isin(values)]
        logger.info(
            "Kept %s rows for %s in %s (removed %s)",
            len(data),
            column,
            values,
            before - len(data),
        )

    drop_rules = filter_params.get('drop_rows', []) or []
    for rule in drop_rules:
        column = rule.get('column')
        values = rule.get('values')
        if not column or column not in data.columns or not values:
            continue
        before = len(data)
        data = data[~data[column].isin(values)]
        logger.info(
            "Dropped %s rows for %s in %s", before - len(data), column, values
        )

    exclude_columns = filter_params.get('exclude_columns', [])
    if exclude_columns:
        drop_targets = [col for col in exclude_columns if col in data.columns]
        if drop_targets:
            data = data.drop(columns=drop_targets)
            logger.info(
                "Excluded %s columns", initial_cols - len(data.columns)
            )

    logger.info(
        "Filtering complete: rows %s -> %s, cols %s -> %s",
        initial_rows,
        len(data),
        initial_cols,
        len(data.columns),
    )
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

    return data


def normalize_missing_values(
    data: pd.DataFrame,
    placeholders: Dict[str, Any],
) -> pd.DataFrame:
    """Standardise missing values ahead of text serialisation.

    Pipeline Order: Preprocess step 5 (5th node, final step)
    Role: Reserved for future use - currently pass-through

    Note:
        Missing value handling is now done at the ingestion stage using Int64 (nullable integer).
        This function is kept for future enhancements but currently returns data unchanged.

    Args:
        data: DataFrame produced by the validation step.
        placeholders: Mapping of placeholder values per dtype category (not used currently)

    Returns:
        DataFrame unchanged (pass-through)
    """
    return data

"""Nodes for data ingestion pipeline."""

import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Load and perform initial validation of raw accounting data.

    Pipeline Order: 1st step (Ingestion pipeline, 1st node)
    Role: Data loading - reads raw parquet file and validates it's not empty

    Args:
        raw_data: Raw accounting data from data/01_raw/row_data.parquet

    Returns:
        DataFrame with validated raw data (3.4M rows)

    Example:
        Input: row_data.parquet
        Output: DataFrame with 54 columns, 3,483,098 rows
    """
    logger.info(f"Loading raw accounting data with {len(raw_data)} records")

    # Basic validation
    if raw_data.empty:
        raise ValueError("Raw data is empty")

    # Log data shape and columns
    logger.info(f"Data shape: {raw_data.shape}")
    logger.info(f"Columns: {raw_data.columns.tolist()}")

    return raw_data


def collect_missing_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """Capture basic missing-value statistics for raw data.

    Args:
        data: Raw accounting dataframe straight from the source file.

    Returns:
        Dictionary with row/column counts and per-column null metrics.
    """
    total_rows = len(data)
    total_columns = len(data.columns)

    null_counts = data.isna().sum().to_dict()
    null_ratios = (
        data.isna().mean().fillna(0).astype(float).to_dict()
        if total_rows
        else {col: 0.0 for col in data.columns}
    )

    stats = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "null_counts": null_counts,
        "null_ratios": null_ratios,
    }

    logger.info(
        "Collected missing stats: rows=%s, columns=%s",
        total_rows,
        total_columns,
    )

    return stats


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for accounting data.

    Pipeline Order: 2nd step (Ingestion pipeline, 2nd node)
    Role: Schema standardization - converts Korean column names to English

    Args:
        data: DataFrame with raw column names from validated_raw_data

    Returns:
        DataFrame with standardized English column names

    Example:
        Input: '거래처명', '계정과목코드'
        Output: 'party_name', 'acct_code'
    """
    # Define explicit column name mapping based on requirements
    column_mapping = {
        'id': 'id',
        'created_at': 'created_at',
        'Name': 'name',
        '연도': 'year',
        '전표일자': 'date',
        '부가세유형': 'vat_type',
        '차변/대변': 'dr_cr',
        '거래처코드': 'party_code',
        '거래처명': 'party_name',
        '계정과목코드': 'acct_code',
        '적요번호': 'desc_id',
        '적요명': 'desc',
        '계정과목': 'acct_name',
        '회사.회사명': 'company_name',
        '회사.사업자등록번호': 'vat_id_company',
        '매입.품명': 'item',
        '매입.공급가액': 'supply_amount',
        '매입.부가세': 'vat_amount',
        '매입.합계금액': 'total_amount',
        '매입.복수거래여부': 'is_multi',
        '매입.전자여부': 'is_electronic',
        '매입.불공제사유': 'nondeductible_reason',
        '매입.봉사료': 'service_charge',
        '매입.의제류구분': 'deemed_code',
        '거래처유형코드': 'party_type',
        '사업자등록번호': 'vat_id_party',
        '수량': 'quantity',
        '단가': 'unit_price',
        '전표키': 'journal_key',
        '상대.거래처코드': 'counterparty_code',
        '상대.거래처명': 'counterparty_name',
        '상대.계정과목코드': 'counterparty_account_code',
        '상대.계정과목': 'counterparty_account',
        '상대.카드번호': 'card_number',
        '증빙코드': 'document_type',
        '매출매입코드': 'transaction_code',
        '전표키_카운팅': 'journal_key_count',
        'batch_no': 'batch_no',
        '업태': 'business_type',
        '종목': 'business_item',
        '과세유형메시지(코드)': 'vat_message_code',
        '폐업일': 'close_date',
        '과세유형전환일자': 'vat_change_date',
        '세금계산서적용일자': 'tax_invoice_apply_date',
        '변경젼과세유형메시지(코드)': 'vat_message_code_previous',
        '업종코드': 'business_code',
        '주요제품': 'product',
        '기업구분': 'entity_type',
        '회사.업태': 'company_business_type',
        '회사.종목': 'company_business_item',
        '회사.과세유형메시지(코드)': 'vat_message_code_company',
        '회사.업종코드': 'company_business_code',
        '회사.주요제품': 'company_product',
        '회사.기업구분': 'company_entity_type'
    }

    # Apply mapping
    new_columns = []
    for col in data.columns:
        if col in column_mapping:
            new_columns.append(column_mapping[col])
        else:
            # If not in mapping, standardize by lowercase and replace spaces/hyphens
            new_col = col.lower().replace(' ', '_').replace('-', '_')
            logger.warning(f"Column '{col}' not in mapping, using standardized name: '{new_col}'")
            new_columns.append(new_col)

    data.columns = new_columns
    logger.info(f"Standardized columns: {data.columns.tolist()}")

    return data


def transform_dtype(data: pd.DataFrame) -> pd.DataFrame:
    """Transform data types for optimal processing and serialization.

    Pipeline Order: 3rd step (Ingestion pipeline, 3rd node)
    Role: Type normalization - standardize dtypes once at ingestion

    Strategy:
    1. Columns with 'date' in name → Clean to YYYYMMDD format → Int64 (nullable integer)
    2. Columns with 'amount' in name → Round to integer → Int64 (nullable integer)
    3. All other columns → string dtype

    Args:
        data: DataFrame from standardized_data

    Returns:
        DataFrame with standardized data types

    Example:
        Input:  document_type=13.0 (float32), supply_amount=1000.5 (float64), date=20250101 (int)
        Output: document_type="13" (string), supply_amount=1001 (Int64), date=20250101 (Int64)
    """
    DATE_TOKEN = "date"
    AMOUNT_TOKEN = "amount"

    data = data.copy()

    # Identify column groups by pattern
    date_cols = [col for col in data.columns if DATE_TOKEN in col.lower()]
    amount_cols = [col for col in data.columns if AMOUNT_TOKEN in col.lower()]

    # 1. Process date columns → Int64 (YYYYMMDD format)
    for col in date_cols:
        # Convert to string and extract only digits (removes .0 from float strings)
        s = data[col].astype("string")
        cleaned = s.str.extract(r"(\d+)", expand=False).replace("", pd.NA)

        # Validate 8-digit date format
        bad_mask = cleaned.str.len().notna() & (cleaned.str.len() != 8)
        if bad_mask.any():
            invalid_samples = cleaned[bad_mask].unique()[:5].tolist()
            logger.warning(f"{col}: Found {bad_mask.sum()} invalid date tokens (not 8 digits) - samples: {invalid_samples}")
            cleaned = cleaned.mask(bad_mask)

        data[col] = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
        logger.info(f"Converted {col} to Int64 (date format)")

    # 2. Process amount columns → float64 (keep for calculations)
    for col in amount_cols:
        numeric = pd.to_numeric(data[col], errors="coerce")
        data[col] = numeric.astype("float64")
        logger.info(f"Converted {col} to float64 (for calculations)")

    # 3. Process remaining columns → string (dtype-based handling)
    protected = set(date_cols) | set(amount_cols)
    for col in data.columns:
        if col not in protected:
            if pd.api.types.is_float_dtype(data[col].dtype):
                # float → Int64 → string (removes .0 from float representations)
                data[col] = data[col].astype("Int64").astype("string")
                logger.info(f"Converted {col} to string (float → Int64 → string)")
            elif pd.api.types.is_integer_dtype(data[col].dtype):
                # int → string
                data[col] = data[col].astype("string")
                logger.info(f"Converted {col} to string (int → string)")
            else:
                # other types → string (skip if already string)
                if not pd.api.types.is_string_dtype(data[col].dtype):
                    data[col] = data[col].astype("string")
                    logger.info(f"Converted {col} to string (other → string)")

    logger.info(
        f"Type transformation complete: {len(date_cols)} date columns, "
        f"{len(amount_cols)} amount columns, {len(data.columns) - len(protected)} string columns"
    )

    return data


def extract_metadata(data: pd.DataFrame) -> Dict[str, Any]:
    """Extract metadata from the ingested data.

    Pipeline Order: 4th step (Ingestion pipeline, 4th node)
    Role: Metadata collection - captures data statistics for monitoring

    Args:
        data: Ingested DataFrame from standardized_data

    Returns:
        Dictionary containing metadata (row count, columns, dtypes, etc.)

    Example:
        Output: {'total_records': 3483098, 'columns': [...], 'null_counts': {...}}
    """
    metadata = {
        'total_records': len(data),
        'total_columns': len(data.columns),
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.astype(str).to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'null_counts': data.isnull().sum().to_dict(),
        'numeric_columns': data.select_dtypes(include=['number']).columns.tolist(),
        'text_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': data.select_dtypes(include=['datetime64']).columns.tolist()
    }

    # Add accounting-specific metadata
    if 'account' in data.columns:
        metadata['unique_accounts'] = data['account'].nunique()
    if 'category' in data.columns:
        metadata['unique_categories'] = data['category'].nunique()
    if 'amount' in data.columns:
        metadata['total_amount'] = data['amount'].sum()
        metadata['avg_amount'] = data['amount'].mean()

    return metadata

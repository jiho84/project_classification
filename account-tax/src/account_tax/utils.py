"""Minimal utility functions - only custom business logic."""

import pandas as pd
from typing import Dict, Any


# ============================================================================
# ACCOUNTING-SPECIFIC UTILITIES (커스텀 비즈니스 로직만)
# ============================================================================

def categorize_accounts(df: pd.DataFrame, account_mapping: Dict[str, str]) -> pd.DataFrame:
    """회계 계정 분류."""
    df['account_category'] = df['account'].map(account_mapping).fillna('Other')
    return df


def calculate_tax_categories(df: pd.DataFrame) -> pd.DataFrame:
    """세무 카테고리 계산."""
    tax_rules = {
        'income': ['revenue', 'sales', 'income'],
        'expense': ['cost', 'expense', 'overhead'],
        'asset': ['cash', 'receivable', 'inventory', 'equipment'],
        'liability': ['payable', 'loan', 'debt'],
        'equity': ['capital', 'retained', 'stock']
    }

    if 'account' in df.columns:
        df['tax_category'] = 'Other'
        for category, keywords in tax_rules.items():
            for keyword in keywords:
                mask = df['account'].str.lower().str.contains(keyword, na=False)
                df.loc[mask, 'tax_category'] = category

    return df


def aggregate_by_period(df: pd.DataFrame, period_column: str = 'date',
                        aggregation_level: str = 'month') -> pd.DataFrame:
    """기간별 회계 데이터 집계."""
    if period_column not in df.columns:
        return df

    df[period_column] = pd.to_datetime(df[period_column])

    period_map = {
        'month': 'M',
        'quarter': 'Q',
        'year': 'Y'
    }

    if aggregation_level in period_map:
        df['period'] = df[period_column].dt.to_period(period_map[aggregation_level])

    return df
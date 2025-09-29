"""Evaluation pipeline - using sklearn.metrics directly."""

import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """Use sklearn metrics directly."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred)
    }


def calculate_tax_impact(predictions: pd.DataFrame) -> Dict[str, float]:
    """Calculate tax-specific metrics (custom business logic)."""
    if 'amount' not in predictions.columns:
        return {}

    misclassified = predictions[predictions['predicted'] != predictions['actual']]
    return {
        "misclassified_amount": float(misclassified['amount'].sum()),
        "misclassified_percentage": float(
            misclassified['amount'].sum() / predictions['amount'].sum() * 100
        )
    }
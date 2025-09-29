"""Pipeline for model evaluation."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    evaluate_classification_model,
    evaluate_tax_classification,
    generate_evaluation_report,
    calculate_business_metrics
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model evaluation pipeline.

    Returns:
        Pipeline for evaluating account tax models
    """
    return Pipeline(
        [
            node(
                func=evaluate_classification_model,
                inputs=["y_test", "y_pred", "y_proba"],
                outputs="classification_metrics",
                name="evaluate_classification_model"
            ),
            node(
                func=evaluate_tax_classification,
                inputs=["test_predictions", "tax_categories"],
                outputs="tax_metrics",
                name="evaluate_tax_classification"
            ),
            node(
                func=calculate_business_metrics,
                inputs=["test_predictions", "test_actuals"],
                outputs="business_metrics",
                name="calculate_business_metrics"
            ),
            node(
                func=generate_evaluation_report,
                inputs=["classification_metrics", "tax_metrics", "model_info"],
                outputs="evaluation_report",
                name="generate_evaluation_report"
            )
        ]
    )
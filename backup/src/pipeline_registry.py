"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# pipelines 패키지에서 import
from my_project.pipelines.data import create_pipeline as create_data_pipeline
# from train.train_pipeline import create_pipelines as create_train_pipelines
# from inference.inference_pipeline import create_pipelines as create_inference_pipelines


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # 데이터 파이프라인 등록
    data_pipeline = create_data_pipeline()
    
    # TODO: 학습 파이프라인 등록
    # train_pipelines = create_train_pipelines()
    
    # TODO: 추론 파이프라인 등록
    # inference_pipelines = create_inference_pipelines()
    
    # 모든 파이프라인 통합
    all_pipelines = {
        "__default__": data_pipeline,
        "data": data_pipeline,
        # **train_pipelines,
        # **inference_pipelines,
    }
    
    return all_pipelines

#!/usr/bin/env python3
"""
MLOps Pipeline - Kedro Container 방식
데이터와 메타데이터를 완전히 분리하여 처리
MLflow 메트릭 자동 로깅 지원
"""

from kedro.pipeline import Pipeline, node
from .data_nodes import (
    stage_0_column_standardization,
    stage_1_data_validation_filtering,
    stage_2_data_quality_check,
    stage_3_data_cleaning,
    stage_4_feature_engineering,
    stage_5_feature_selection,
    stage_6_data_splitting,
    stage_7_data_scaling,
    stage_8_text_encoding,
    stage_9_final_packaging
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    0-9단계 MLOps 파이프라인 생성
    데이터와 메타데이터가 분리되어 흐름
    MLflow 메트릭 자동 로깅
    
    Returns:
        Pipeline: Kedro 파이프라인 객체
    """
    return Pipeline([
        # Stage 0: 데이터 로드 & 컬럼명 표준화
        node(
            func=stage_0_column_standardization,
            inputs=["raw_data", "parameters"],
            outputs=[
                "stage_0_data", 
                "stage_0_metadata",
                "stage_0_processing_time",
                "stage_0_output_count"
            ],
            name="stage_0_column_standardization",
            tags=["stage_0", "column_roles"]
        ),
        
        # Stage 1: 데이터 검증 및 필터링
        node(
            func=stage_1_data_validation_filtering,
            inputs=["stage_0_data", "stage_0_metadata", "parameters"],
            outputs=[
                "stage_1_data", 
                "stage_1_metadata",
                "stage_1_processing_time",
                "stage_1_output_count"
            ],
            name="stage_1_data_validation_filtering",
            tags=["stage_1", "validation", "filtering"]
        ),
        
        # Stage 2: 데이터 품질 검증
        node(
            func=stage_2_data_quality_check,
            inputs=["stage_1_data", "stage_1_metadata", "parameters"],
            outputs=[
                "stage_2_data", 
                "stage_2_metadata",
                "stage_2_processing_time",
                "stage_2_output_count",
                "data_quality_duplicate_rows",
                "class_imbalance_ratio"
            ],
            name="stage_2_data_quality_check",
            tags=["stage_2", "quality_check"]
        ),
        
        # Stage 3: 데이터 정제 & 라벨 매핑
        node(
            func=stage_3_data_cleaning,
            inputs=["stage_2_data", "stage_2_metadata", "parameters"],
            outputs=[
                "stage_3_data", 
                "stage_3_metadata",
                "stage_3_processing_time",
                "stage_3_output_count"
            ],
            name="stage_3_data_cleaning",
            tags=["stage_3", "cleaning", "label_mapping"]
        ),
        
        # Stage 4: 피처 엔지니어링
        node(
            func=stage_4_feature_engineering,
            inputs=["stage_3_data", "stage_3_metadata", "parameters"],
            outputs=[
                "stage_4_data", 
                "stage_4_metadata",
                "stage_4_processing_time",
                "stage_4_output_count",
                "outlier_removal_ratio"
            ],
            name="stage_4_feature_engineering",
            tags=["stage_4", "feature_engineering"]
        ),
        
        # Stage 5: 피처 선택
        node(
            func=stage_5_feature_selection,
            inputs=["stage_4_data", "stage_4_metadata", "parameters"],
            outputs=[
                "stage_5_data", 
                "stage_5_metadata",
                "stage_5_processing_time",
                "stage_5_output_count"
            ],
            name="stage_5_feature_selection",
            tags=["stage_5", "feature_selection"]
        ),
        
        # Stage 6: 데이터 분할 (중요 분기점!)
        node(
            func=stage_6_data_splitting,
            inputs=["stage_5_data", "stage_5_metadata", "parameters"],
            outputs=[
                "stage_6_data", 
                "stage_6_metadata",
                "stage_6_processing_time",
                "stage_6_output_count"
            ],
            name="stage_6_data_splitting",
            tags=["stage_6", "data_splitting", "critical_point"]
        ),
        
        # Stage 7: 데이터 스케일링
        node(
            func=stage_7_data_scaling,
            inputs=["stage_6_data", "stage_6_metadata", "parameters"],
            outputs=[
                "stage_7_data", 
                "stage_7_metadata",
                "stage_7_processing_time",
                "stage_7_output_count"
            ],
            name="stage_7_data_scaling",
            tags=["stage_7", "scaling"]
        ),
        
        # Stage 8: 텍스트 인코딩
        node(
            func=stage_8_text_encoding,
            inputs=["stage_7_data", "stage_7_metadata", "parameters"],
            outputs=[
                "stage_8_data", 
                "stage_8_metadata",
                "stage_8_processing_time",
                "stage_8_output_count"
            ],
            name="stage_8_text_encoding",
            tags=["stage_8", "text_encoding"]
        ),
        
        # Stage 9: 최종 패키징
        node(
            func=stage_9_final_packaging,
            inputs=["stage_8_data", "stage_8_metadata", "parameters"],
            outputs=[
                "final_data", 
                "final_metadata",
                "stage_9_processing_time"
            ],
            name="stage_9_final_packaging",
            tags=["stage_9", "final_output"]
        )
    ])


def create_0_to_5_pipeline(**kwargs) -> Pipeline:
    """
    0-5단계만 실행하는 파이프라인 (분할 전까지)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(6))
    ])


def create_6_to_9_pipeline(**kwargs) -> Pipeline:
    """
    6-9단계만 실행하는 파이프라인 (분할 후부터)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(6, 10))
    ])


def create_data_quality_pipeline(**kwargs) -> Pipeline:
    """
    데이터 품질 체크 파이프라인 (Stage 0-3)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(4))
    ])


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    """
    피처 엔지니어링 파이프라인 (Stage 3-5)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(3, 6))
    ])


def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    """
    전처리 파이프라인 (Stage 0-6, 분할까지만)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(7))
    ])


def create_model_prep_pipeline(**kwargs) -> Pipeline:
    """
    모델 준비 파이프라인 (Stage 7-9)
    """
    pipeline = create_pipeline(**kwargs)
    return Pipeline([
        node for node in pipeline.nodes 
        if any(f"stage_{i}" in node.tags for i in range(7, 10))
    ])


# 체크포인트가 있는 파이프라인
def create_pipeline_with_checkpoints(**kwargs) -> Pipeline:
    """
    체크포인트를 저장하는 파이프라인
    Stage 3, 6에서 데이터를 파일로 저장
    """
    base_pipeline = create_pipeline(**kwargs)
    
    # 체크포인트 저장 노드 추가
    checkpoint_nodes = [
        # Stage 3 체크포인트
        node(
            func=lambda x: x,
            inputs="stage_3_data",
            outputs="stage_3_checkpoint",
            name="save_stage_3_checkpoint",
            tags=["stage_3", "save_checkpoint"]
        ),
        # Stage 3 라벨 매핑 저장
        node(
            func=lambda metadata: metadata.get("label_mapping", {}),
            inputs="stage_3_metadata",
            outputs="stage_3_label_mapping",
            name="save_stage_3_label_mapping",
            tags=["stage_3", "save_metadata"]
        ),
        
        # Stage 4 스케일링 통계 저장 - 더 이상 사용하지 않음
        # node(
        #     func=lambda metadata: metadata.get("company_scaling_stats", {}),
        #     inputs="stage_4_metadata",
        #     outputs="stage_4_company_scaling_stats",
        #     name="save_stage_4_scaling_stats",
        #     tags=["stage_4", "save_metadata"]
        # ),
        
        # Stage 6 분할 데이터 체크포인트로 저장
        node(
            func=lambda x: x,
            inputs="stage_6_data",
            outputs="stage_6_checkpoint",
            name="save_stage_6_checkpoint",
            tags=["stage_6", "save_checkpoint"]
        ),
    ]
    
    return Pipeline(base_pipeline.nodes + checkpoint_nodes)


# 메타데이터만 추출하는 파이프라인
def create_metadata_only_pipeline(**kwargs) -> Pipeline:
    """
    메타데이터만 추출하여 분석하는 파이프라인
    """
    return Pipeline([
        # 각 스테이지의 메타데이터만 추출
        node(
            func=lambda m0, m1, m2, m3, m4, m5, m6, m7, m8: {
                "stage_0": m0,
                "stage_1": m1,
                "stage_2": m2,
                "stage_3": m3,
                "stage_4": m4,
                "stage_5": m5,
                "stage_6": m6,
                "stage_7": m7,
                "stage_8": m8
            },
            inputs=[
                "stage_0_metadata",
                "stage_1_metadata", 
                "stage_2_metadata",
                "stage_3_metadata",
                "stage_4_metadata",
                "stage_5_metadata",
                "stage_6_metadata",
                "stage_7_metadata",
                "stage_8_metadata"
            ],
            outputs="all_metadata_summary",
            name="extract_all_metadata",
            tags=["metadata", "analysis"]
        )
    ])


# 파이프라인 별칭 (호환성 유지)
default_pipeline = create_pipeline
data_pipeline = create_pipeline
pre_split_pipeline = create_0_to_5_pipeline
post_split_pipeline = create_6_to_9_pipeline
quality_pipeline = create_data_quality_pipeline
feature_pipeline = create_feature_engineering_pipeline
preprocessing_pipeline = create_preprocessing_pipeline
model_prep_pipeline = create_model_prep_pipeline
checkpoint_pipeline = create_pipeline_with_checkpoints
metadata_pipeline = create_metadata_only_pipeline
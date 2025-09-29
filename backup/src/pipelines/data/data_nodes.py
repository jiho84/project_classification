#!/usr/bin/env python3
"""
MLOps Pipeline Nodes - Kedro Container 방식
데이터와 메타데이터를 완전히 분리하여 관리
"""

import pandas as pd
import numpy as np
import json
import os
import time
import gc  # 가비지 컬렉션을 위해 추가
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

# 통합된 유틸리티 함수 import
from .util import (
    # Config utilities
    get_stage_config,
    
    # Data utilities
    validate_required_columns,
    get_column_statistics,
    apply_row_filters,
    calculate_class_distribution,
    
    # Label mapping utilities
    create_label_mapping,
    
    # Feature engineering utilities
    apply_robust_standardization,
    apply_z_score_transformation,
    add_day_type_feature,
    
    # Text encoding utilities
    prepare_split_data,
    perform_tokenizer_analysis,
)
from .util import prepare_metadata_for_json
from .util import optimize_dataframe_dtypes, reduce_metadata_size, get_dataframe_memory_info

logger = logging.getLogger(__name__)


# =============================================================================
# 메모리 관리 유틸리티
# =============================================================================

def cleanup_memory(stage_name: str = "") -> None:
    """
    명시적 가비지 컬렉션 수행 및 메모리 정리
    
    Args:
        stage_name: 스테이지 이름 (로깅용)
    """
    before_gc = gc.collect()
    gc.collect()
    after_gc = gc.collect()
    
    if stage_name:
        logger.debug(f"🧹 {stage_name} 메모리 정리: {before_gc} 객체 해제")
    
    return


# =============================================================================
# Stage 0: 타겟 컬럼 정의
# =============================================================================

def stage_0_column_standardization(
    raw_data: pd.DataFrame, 
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int]:
    """
    Stage 0: 컬럼명 표준화 및 타겟 컬럼 정의
    
    Args:
        raw_data: 원본 데이터프레임
        parameters: 파라미터 설정
        
    Returns:
        df: 표준화된 DataFrame
        metadata: 초기 메타데이터
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 행 수
    """
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("🚉 Stage 0: 컬럼명 표준화 및 타겟 컬럼 정의")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_0")
    stage_config = config["stage_config"]
    
    # 원본 컬럼 저장
    original_columns = list(raw_data.columns)
    df = raw_data
    
    # 컬럼명 표준화
    rename_dict = {}
    if stage_config.get("column_name_standardization", {}).get("enabled", True):
        mapping_file = stage_config.get("column_name_standardization", {}).get("mapping_file")
        if mapping_file and os.path.exists(mapping_file):
            logger.info(f"📂 컬럼 매핑 파일 로드: {mapping_file}")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                columns_mapping = json.load(f)
            
            for col in original_columns:
                if col in columns_mapping:
                    rename_dict[col] = columns_mapping[col]
                    logger.info(f"  📝 {col} → {columns_mapping[col]}")
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
                logger.info(f"✅ {len(rename_dict)}개 컬럼명 표준화 완료")
    
    # 타겟 컬럼 정의
    target_column = stage_config.get("target_column", "")
    if not target_column:
        raise ValueError("타겟 컬럼이 정의되지 않았습니다.")
    
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
    
    logger.info(f"🎯 타겟 컬럼: {target_column}")
    
    # 메모리 최적화
    logger.info("🔧 DataFrame 메모리 최적화 중...")
    df = optimize_dataframe_dtypes(df, categorical_threshold=0.3, exclude_columns=[target_column])
    
    # 메타데이터 초기화 (메모리 효율적으로)
    metadata = {
        "column_mapping": {
            "original_columns": original_columns,  # Stage 9에서 필요
            "original_columns_count": len(original_columns),
            "standardized_columns": list(df.columns),  # 표준화된 컬럼 리스트
            "standardized_columns_count": len(df.columns),
            "rename_dict": rename_dict,
            "columns_renamed": len(rename_dict)
        },
        "target_column": target_column,
        "num_features": len([col for col in df.columns if col != target_column]),
        "pipeline_start_time": datetime.now().isoformat(),
        "memory_info": get_dataframe_memory_info(df)
    }
    
    logger.info(f"✅ Stage 0 완료: {len(df):,} rows, {len(df.columns)} columns")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = len(df)
    
    # 메모리 정리
    cleanup_memory("Stage 0")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 1: 데이터 필터링
# =============================================================================

def stage_1_data_validation_filtering(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int]:
    """
    Stage 1: 스키마/무결성 검증 및 필터링
    
    Args:
        df: DataFrame
        metadata: Stage 0의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        df: 필터링된 DataFrame
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 행 수
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 1: 데이터 검증 및 필터링")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_1")
    stage_config = config["stage_config"]
    validation_config = stage_config.get("validation", {})
    filtering_config = stage_config.get("filtering", {})
    
    input_shape = df.shape
    
    # 검증 수행
    validation_results = {}
    if validation_config.get("enable_validation", True):
        # 필수 컬럼 검증
        required_columns = validation_config.get("validation_rules", {}).get("required_columns", [])
        missing_columns = validate_required_columns(df, required_columns)
        
        if missing_columns:
            logger.warning(f"⚠️ 누락된 필수 컬럼: {missing_columns}")
        
        # 컬럼별 통계 수집
        column_statistics = get_column_statistics(df)
        
        # 특정 컬럼 null 비율 체크
        cust_type_code_null_ratio = None
        if "cust_type_code" in df.columns:
            cust_type_code_null_ratio = column_statistics.get("cust_type_code", {}).get("null_ratio", 0)
            logger.info(f"📊 cust_type_code null 비율: {cust_type_code_null_ratio:.2%}")
        
        validation_results = {
            "missing_required_columns": missing_columns,
            "cust_type_code_null_ratio": cust_type_code_null_ratio
        }
    
    # 필터링 적용
    filtering_stats = {}
    if filtering_config.get("enabled", True):
        df, filtering_stats = apply_row_filters(df, filtering_config)
        logger.info(f"🔽 필터링 후: {len(df):,} rows (제거: {input_shape[0] - len(df):,})")
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,  # 이전 메타데이터 유지
        "column_statistics": column_statistics if validation_config.get("enable_validation", True) else {},
        "validation_results": validation_results,
        "filtering_stats": filtering_stats
    }
    
    logger.info(f"✅ Stage 1 완료: {len(df):,} rows")
    
    # 처리 시간 계산
    processing_time = time.time() - start_time
    output_count = len(df)
    
    # 메모리 정리
    cleanup_memory("Stage 1")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 2: 데이터 검증
# =============================================================================

def stage_2_data_quality_check(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int, int, float]:
    """
    Stage 2: 데이터 정제 및 필터링
    
    Args:
        df: DataFrame
        metadata: Stage 1의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        df: 정제된 DataFrame
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 행 수
        duplicate_rows: 중복 행 수
        class_imbalance_ratio: 클래스 불균형 비율
    """
    start_time = time.time()
    initial_rows = len(df)
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 2: 데이터 정제 및 필터링")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_2")
    stage_config = config["stage_config"]
    
    target_column = metadata["target_column"]
    
    # 1. 컬럼 제외
    exclude_columns = stage_config.get("exclude_columns", [])
    if exclude_columns:
        df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        logger.info(f"🔽 제외된 컬럼: {exclude_columns}")
    
    # 2. 행 필터링
    row_filters = stage_config.get("row_filters", {})
    filter_stats = {}
    for col, filter_config in row_filters.items():
        if col in df.columns:
            before_count = len(df)
            operator = filter_config.get("operator", "equals")
            values = filter_config.get("values", [])
            
            if operator == "between" and len(values) == 2:
                df = df[(df[col] >= values[0]) & (df[col] <= values[1])]
            elif operator == "in":
                df = df[df[col].isin(values)]
            elif operator == "equals" and values:
                df = df[df[col] == values[0]]
            
            after_count = len(df)
            filter_stats[col] = {
                "removed": before_count - after_count,
                "filter_type": operator,
                "values": values
            }
            logger.info(f"🔽 {col} 필터링: {before_count - after_count}개 행 제거")
    
    # 3. NULL 값 처리
    drop_null_columns = stage_config.get("drop_null_columns", [])
    null_stats = {}
    for col in drop_null_columns:
        if col in df.columns:
            before_count = len(df)
            df = df.dropna(subset=[col])
            null_stats[col] = before_count - len(df)
            logger.info(f"🔽 {col} NULL 제거: {null_stats[col]}개 행")
    
    # 4. 중복 제거
    duplicate_count = 0
    dedup_config = stage_config.get("deduplication", {})
    if dedup_config.get("enabled", False):
        subset_columns = dedup_config.get("subset_columns", None)
        keep = dedup_config.get("keep", "first")
        before_count = len(df)
        df = df.drop_duplicates(subset=subset_columns, keep=keep)
        duplicate_count = before_count - len(df)
        logger.info(f"🔽 중복 제거: {duplicate_count}개 행")
    
    # 5. 클래스 필터링
    class_filter_config = stage_config.get("class_filter", {})
    class_filter_stats = {}
    if class_filter_config:
        min_samples = class_filter_config.get("min_samples", 2)
        filter_column = class_filter_config.get("target_column", target_column)
        
        if filter_column and filter_column in df.columns:
            class_counts = df[filter_column].value_counts()
            valid_classes = class_counts[class_counts >= min_samples].index
            before_count = len(df)
            df = df[df[filter_column].isin(valid_classes)]
            
            class_filter_stats = {
                "removed_classes": len(class_counts) - len(valid_classes),
                "removed_rows": before_count - len(df),
                "min_samples": min_samples
            }
            logger.info(f"🔽 클래스 필터링: {class_filter_stats['removed_classes']}개 클래스, "
                       f"{class_filter_stats['removed_rows']}개 행 제거")
    
    # 6. 클래스 분포 계산
    class_distribution = None
    if target_column and target_column in df.columns:
        class_distribution = calculate_class_distribution(df, target_column)
        logger.info(f"📊 클래스 분포: {class_distribution['num_classes']}개 클래스")
    
    # 메타데이터 업데이트
    metadata = {
        **metadata,
        "stage_2_stats": {
            "initial_rows": initial_rows,
            "final_rows": len(df),
            "total_removed": initial_rows - len(df),
            "exclude_columns": exclude_columns,
            "row_filter_stats": filter_stats,
            "null_removal_stats": null_stats,
            "deduplication_stats": {
                "enabled": dedup_config.get("enabled", False),
                "duplicates_removed": duplicate_count
            },
            "class_filter_stats": class_filter_stats
        },
        "class_distribution": class_distribution
    }
    
    logger.info(f"✅ Stage 2 완료: {initial_rows:,} → {len(df):,} 행 (제거: {initial_rows - len(df):,})")
    
    # 처리 시간 및 메트릭
    processing_time = time.time() - start_time
    output_count = len(df)
    duplicate_rows_metric = duplicate_count
    class_imbalance_metric = class_distribution.get('class_imbalance_ratio', 1.0) if class_distribution else 1.0
    
    # 메모리 정리
    cleanup_memory("Stage 2")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count, duplicate_rows_metric, class_imbalance_metric


# =============================================================================
# Stage 3: 데이터 정제
# =============================================================================

def stage_3_data_cleaning(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int]:
    """
    Stage 3: 데이터 정제 및 전처리
    
    Args:
        df: DataFrame
        metadata: Stage 2의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        df: 정제된 DataFrame
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 행 수
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 3: 데이터 정제 및 전처리")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_3")
    stage_config = config["stage_config"]
    
    target_column = metadata["target_column"]
    
    # Stage 3에서는 결측값 처리 제거 - Stage 8에서만 처리
    cleaning_stats = {}
    logger.info("ℹ️ 결측값 처리는 Stage 8에서 수행됩니다.")
    
    # 라벨 매핑 생성 또는 로드
    label_mapping = {}
    if target_column and target_column in df.columns:
        label_mapping_config = stage_config.get("label_mapping", {})
        # parameters.yml에서 existing_path 가져오기
        existing_path = label_mapping_config.get("existing_path", "metadata/mapping/label_mapping.json")
        # 절대 경로로 변환
        project_path = "/home/user/projects/my_project/pj_mlflow_ds3"
        mapping_path = os.path.join(project_path, existing_path)
        
        # create_label_mapping 함수 호출 (auto 모드로)
        label_mapping = create_label_mapping(
            data=df,
            target_column=target_column,
            total_num_classes=label_mapping_config.get("total_num_classes", 300),
            dummy_prefix=label_mapping_config.get("dummy_class_prefix", "DUMMY_"),
            mode="auto",  # auto 모드는 파일 존재 여부에 따라 자동 결정
            existing_path=mapping_path
        )
        
        # 파일 저장 (항상 최신 상태로 유지)
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 라벨 매핑 저장 완료: {mapping_path}")
        
        # 신규 클래스 정보 출력
        if label_mapping.get("new_classes_added"):
            logger.info(f"🆕 추가된 신규 클래스: {label_mapping['new_classes_added']}")
        
        logger.info(f"🏷️ 라벨 매핑 상태: {label_mapping['num_classes']}개 클래스 "
                   f"(실제: {label_mapping['num_base_classes']}, 더미: {label_mapping['num_dummy_classes']})")
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,
        "cleaning_stats": cleaning_stats,
        "label_mapping": label_mapping,  # 중요한 메타데이터 추가
        "label_mapping_summary": {
            "num_classes": label_mapping.get('num_classes', 0),
            "num_base_classes": label_mapping.get('num_base_classes', 0),
            "num_dummy_classes": label_mapping.get('num_dummy_classes', 0)
        }
    }
    
    logger.info(f"✅ Stage 3 완료: {len(df):,} rows")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = len(df)
    
    # 메모리 정리
    cleanup_memory("Stage 3")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 4: 피처 엔지니어링
# =============================================================================

def stage_4_feature_engineering(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int, float]:
    """
    Stage 4: 피처 엔지니어링
    
    Args:
        df: DataFrame
        metadata: Stage 3의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        df: 피처 엔지니어링된 DataFrame
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 행 수
        outlier_removal_ratio: 이상치 제거 비율
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 4: 피처 엔지니어링")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_4")
    stage_config = config["stage_config"]
    
    company_scaling_stats = {}
    feature_engineering_summary = {}
    outlier_stats = {}
    
    # Robust standardization 적용
    if stage_config.get("robust_standardization", {}).get("enable", False):
        robust_config = stage_config["robust_standardization"]
        
        df, company_stats, feature_summary, outlier_stats = apply_robust_standardization(
            data=df,
            config=robust_config
        )
        
        # 회사별 스케일링 통계
        if company_stats:
            company_scaling_stats = {
                "stats": company_stats,
                "global_params": {
                    "outlier_method": robust_config.get("outlier_bounds", {}).get("method", "iqr"),
                    "scaling_range": robust_config.get("scaling_range", [1, 100])
                }
            }
        
        feature_engineering_summary = feature_summary
        
        logger.info(f"🔧 피처 엔지니어링 완료: {feature_summary.get('features_added', 0)}개 피처 추가")
    
    # Z-score 변환 적용
    z_score_stats = {}
    z_score_config = stage_config.get("z_score_transformation", {})
    if z_score_config.get("enabled", False):
        logger.info("📊 Z-score 변환 시작...")
        df, z_score_stats = apply_z_score_transformation(df, z_score_config)
        
        # Z-score 통계 로깅
        for col_name, col_stats in z_score_stats.items():
            if "group_stats" in col_stats:
                logger.info(f"  - {col_name}: {col_stats['total_groups']}개 그룹 처리")
            else:
                logger.info(f"  - {col_name}: 전체 데이터 처리")
            
            if "transformed_mean" in col_stats:
                logger.info(f"    변환 후 평균: {col_stats['transformed_mean']:.2f}, "
                          f"표준편차: {col_stats['transformed_std']:.2f}")
                logger.info(f"    클리핑: 하한 {col_stats['clipped_at_min']}개, "
                          f"상한 {col_stats['clipped_at_max']}개")
    
    # 날짜 관련 피처 생성
    date_features_stats = {}
    date_features_config = stage_config.get("date_features", {})
    if date_features_config.get("enabled", False):
        logger.info("📅 날짜 관련 피처 생성 시작...")
        features = date_features_config.get("features", [])
        
        if "day_type" in features:
            logger.info("  - day_type 피처 추가 중...")
            df = add_day_type_feature(df, date_column='txn_date')
            date_features_stats["day_type"] = "added"
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,
        "feature_engineering_summary": feature_engineering_summary,
        "outlier_stats": outlier_stats,
        "company_scaling_stats": company_scaling_stats,
        "z_score_stats": z_score_stats,
        "date_features_stats": date_features_stats
    }
    
    logger.info(f"✅ Stage 4 완료: {len(df.columns)} features")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = len(df)
    outlier_ratio = outlier_stats.get('outlier_ratio', 0.0)
    
    # 메모리 정리
    cleanup_memory("Stage 4")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count, outlier_ratio


# =============================================================================
# Stage 5: 피처 선택
# =============================================================================

def stage_5_feature_selection(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int]:
    """
    Stage 5: 피처 선택 및 라벨 매핑 업데이트
    
    Args:
        df: DataFrame
        metadata: Stage 4의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        df: 피처 선택된 DataFrame
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 출력 데이터 크기
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 5: 피처 선택")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_5")
    stage_config = config["stage_config"]
    
    target_column = metadata["target_column"]
    
    # 피처 선택 - YAML의 feature_columns 사용
    feature_columns = stage_config.get("feature_columns", [])
    
    if not feature_columns:
        # feature_columns가 지정되지 않은 경우 타겟 컬럼을 제외한 모든 컬럼 사용
        logger.warning("⚠️ feature_columns가 지정되지 않았습니다. 타겟 컬럼을 제외한 모든 컬럼을 사용합니다.")
        feature_columns = [col for col in df.columns if col != target_column]
    else:
        # 지정된 feature_columns 중 실제로 존재하는 컬럼만 필터링
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ 다음 피처들이 데이터에 없습니다: {missing_features}")
        
        feature_columns = available_features
        logger.info(f"📋 YAML에서 지정된 {len(feature_columns)}개 피처 사용")
    
    # 선택된 피처로 데이터 필터링 및 컬럼 순서 정렬
    if feature_columns:
        # feature_columns의 순서대로 컬럼 정렬 + 타겟 컬럼은 마지막에
        selected_columns = feature_columns + ([target_column] if target_column in df.columns else [])
        df = df[selected_columns]
        logger.info(f"✅ {len(feature_columns)}개 피처 선택 및 컬럼 순서 정렬 완료")
        logger.info(f"📊 선택된 피처: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,
        "feature_columns": feature_columns,  # 중요: 선택된 피처 목록
        "feature_selection_info": {
            "method": "yaml_config",
            "num_features_selected": len(feature_columns),
            "feature_names": feature_columns
        }
    }
    
    logger.info(f"✅ Stage 5 완료: {len(feature_columns)} features selected")
    
    # 메트릭 계산
    processing_time = time.time() - start_time
    output_count = len(df)
    
    # 메모리 정리
    cleanup_memory("Stage 5")
    
    return df, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 6: 데이터 분할 (중요 분기점!)
# =============================================================================

def stage_6_data_splitting(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], float, int]:
    """
    Stage 6: 데이터를 train/validation/test로 분할
    이 지점부터 데이터가 3개로 나뉩니다!
    
    Args:
        df: DataFrame
        metadata: Stage 5의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        data: {"train": DataFrame, "val": DataFrame, "test": DataFrame}
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 총 데이터 크기
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 6: 데이터 분할 (중요 분기점!)")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_6")
    stage_config = config["stage_config"]
    splitting_config = stage_config.get("splitting", {})
    
    target_column = metadata["target_column"]
    
    # 분할 비율 - parameters.yml의 실제 구조에 맞게 수정
    train_ratio = splitting_config.get("train_ratio", 0.7)
    val_ratio = splitting_config.get("validation_ratio", 0.15)
    test_ratio = splitting_config.get("test_ratio", 0.15)
    
    # 설정된 비율 로그 출력
    logger.info(f"📊 설정된 분할 비율:")
    logger.info(f"   - Train: {train_ratio:.1%}")
    logger.info(f"   - Validation: {val_ratio:.1%}")
    logger.info(f"   - Test: {test_ratio:.1%}")
    
    # stratify 설정 - 안전한 처리
    def safe_stratify(data, target_col, min_samples_per_class=2):
        """Stratify가 가능한지 확인하고 stratify 컬럼 반환"""
        # splitting_config에서 stratify_column 확인
        stratify_column = splitting_config.get("stratify_column", target_col)
        if not stratify_column or stratify_column not in data.columns:
            return None
        
        # 각 클래스별 샘플 수 확인
        class_counts = data[stratify_column].value_counts()
        min_class_count = class_counts.min()
        
        # splitting_config에서 min_samples_per_class 설정 사용
        min_required = splitting_config.get("min_samples_per_class", min_samples_per_class)
        if min_class_count < min_required:
            logger.warning(f"⚠️ 최소 클래스 샘플 수가 {min_class_count}개로 stratify 불가. 무작위 분할 사용")
            return None
        
        return data[stratify_column]
    
    # 1차 분할: train + val vs test
    stratify_col = safe_stratify(df, target_column)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=splitting_config.get("random_state", 42),
        stratify=stratify_col
    )
    
    # 2차 분할: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    stratify_col = safe_stratify(train_val_df, target_column)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=splitting_config.get("random_state", 42),
        stratify=stratify_col
    )
    
    # 분할 통계
    split_stats = {
        "train": {"rows": len(train_df), "ratio": len(train_df) / len(df)},
        "validation": {"rows": len(val_df), "ratio": len(val_df) / len(df)},
        "test": {"rows": len(test_df), "ratio": len(test_df) / len(df)},
        "total": len(df)
    }
    
    logger.info(f"📊 분할 완료:")
    logger.info(f"   - Train: {split_stats['train']['rows']:,} ({split_stats['train']['ratio']:.1%})")
    logger.info(f"   - Val: {split_stats['validation']['rows']:,} ({split_stats['validation']['ratio']:.1%})")
    logger.info(f"   - Test: {split_stats['test']['rows']:,} ({split_stats['test']['ratio']:.1%})")
    
    # 데이터 구조 변경: {"full": df} → {"train": df, "val": df, "test": df}
    data = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,
        "split_stats": split_stats,
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio
        }
    }
    
    logger.info(f"✅ Stage 6 완료: 데이터가 3개로 분할되었습니다!")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = len(train_df) + len(val_df) + len(test_df)
    
    # 메모리 정리
    cleanup_memory("Stage 6")
    
    return data, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 7: 데이터 스케일링
# =============================================================================

def stage_7_data_scaling(
    data: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], float, int]:
    """
    Stage 7: 데이터 스케일링 (선택적)
    
    Args:
        data: {"train": DataFrame, "val": DataFrame, "test": DataFrame}
        metadata: Stage 6의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        data: {"train": 스케일링된 DataFrame, "val": ..., "test": ...}
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 총 데이터 크기
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 7: 데이터 스케일링")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_7")
    stage_config = config["stage_config"]
    
    # 데이터 타입 확인 (디버깅)
    logger.info(f"Data type: {type(data)}")
    if isinstance(data, dict):
        for key, value in data.items():
            logger.info(f"  {key}: {type(value)}")
    
    # PartitionedDataset 처리 - callable이면 호출하여 데이터 가져오기
    if isinstance(data, dict) and all(callable(v) for v in data.values()):
        logger.info("PartitionedDataset detected - loading actual data...")
        loaded_data = {}
        for key, loader in data.items():
            loaded_data[key] = loader()
            logger.info(f"  Loaded {key}: {type(loaded_data[key])}, shape: {loaded_data[key].shape}")
        data = loaded_data
    
    # 데이터 직접 사용 (복사하지 않음)
    scaled_data = data
    
    # 스케일링 설정
    scaling_enabled = stage_config.get("scaling_enabled", False)
    scaling_info = {"enabled": False, "features_scaled": []}
    
    if scaling_enabled:
        logger.info("📐 스케일링 활성화됨")
        scaling_features = stage_config.get("scaling_features", [])
        
        if scaling_features:
            logger.info(f"   {len(scaling_features)}개 피처에 스케일링 적용")
            # 여기서 실제 스케일링 로직을 구현할 수 있습니다
            # 예: StandardScaler, MinMaxScaler 등
            
            scaling_info = {
                "enabled": True,
                "features_scaled": scaling_features,
                "scaling_method": stage_config.get("scaling_method", "standard")
            }
    else:
        logger.info("⏭️ 스케일링 건너뜀")
    
    # 데이터 컨테이너 (3개 유지)
    data = scaled_data
    
    # 메타데이터 업데이트 (누적)
    metadata = {
        **metadata,
        "scaling_info": scaling_info
    }
    
    logger.info(f"✅ Stage 7 완료")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = len(scaled_data["train"]) + len(scaled_data["val"]) + len(scaled_data["test"])
    
    # 메모리 정리
    cleanup_memory("Stage 7")
    
    return scaled_data, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 8: 모델 입력 준비 (텍스트 인코딩)
# =============================================================================

def stage_8_text_encoding(
    data: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[Dict[str, Any], Dict[str, Any], float, int]:
    """
    Stage 8: 모델 입력을 위한 텍스트 인코딩
    
    Args:
        data: {"train": DataFrame, "val": DataFrame, "test": DataFrame}
        metadata: Stage 7의 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        data: {"train": {text, labels}, "val": {...}, "test": {...}}
        metadata: 업데이트된 메타데이터 (누적)
        processing_time: 처리 시간(초)
        output_count: 총 샘플 수
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 8: 모델 입력 준비 (텍스트 인코딩)")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_8")
    stage_config = config["stage_config"]
    
    target_column = metadata["target_column"]
    feature_columns = metadata["feature_columns"]
    label_mapping = metadata["label_mapping"]
    
    # 텍스트 인코딩 설정
    text_encoding_config = stage_config.get("text_encoding", {})
    
    # 각 분할에 대해 텍스트 인코딩 수행
    logger.info("📝 텍스트 인코딩 시작...")
    
    # PartitionedDataset 처리 - callable이면 호출하여 데이터 가져오기
    if isinstance(data, dict) and all(callable(v) for v in data.values()):
        logger.info("PartitionedDataset detected - loading actual data...")
        loaded_data = {}
        for key, loader in data.items():
            loaded_data[key] = loader()
            logger.info(f"  Loaded {key}: {type(loaded_data[key])}, shape: {loaded_data[key].shape}")
        data = loaded_data
    
    # 실제 데이터프레임에 존재하는 컬럼만 필터링
    if isinstance(data, dict) and len(data) > 0:
        # 첫 번째 분할의 컬럼을 확인 (모든 분할이 동일한 컬럼을 가져야 함)
        sample_df = next(iter(data.values()))
        actual_columns = list(sample_df.columns)
        
        # feature_columns에서 실제 존재하는 컬럼만 선택
        valid_feature_columns = [col for col in feature_columns if col in actual_columns]
        
        logger.info(f"📊 실제 데이터 컬럼 수: {len(actual_columns)}")
        logger.info(f"📊 메타데이터 피처 수: {len(feature_columns)}")
        logger.info(f"📊 사용할 피처 수: {len(valid_feature_columns)}")
        
        # 디버깅: 실제 컬럼과 메타데이터 컬럼 비교
        if len(valid_feature_columns) < len(feature_columns):
            missing_columns = [col for col in feature_columns if col not in actual_columns]
            logger.warning(f"⚠️ 누락된 컬럼: {missing_columns[:10]}...")  # 처음 10개만 표시
            
        # 타겟 컬럼이 없는 경우 처리
        if target_column not in actual_columns:
            logger.error(f"❌ 타겟 컬럼 '{target_column}'이 데이터에 없습니다!")
            logger.info(f"실제 컬럼 목록: {actual_columns}")
            # 타겟 컬럼 찾기 시도
            # 계정과목코드를 타겟으로 사용
            possible_target = [col for col in actual_columns if '계정과목코드' in col or 'acct_code' in col.lower()]
            if possible_target:
                target_column = possible_target[0]
                logger.info(f"🔄 대체 타겟 컬럼 사용: {target_column}")
            else:
                # 마지막 시도 - acct가 포함된 컬럼
                possible_target = [col for col in actual_columns if 'acct' in col.lower() or '계정' in col]
                if possible_target:
                    target_column = possible_target[0]
                    logger.info(f"🔄 대체 타겟 컬럼 사용: {target_column}")
        
        # 사용할 피처 업데이트
        feature_columns = valid_feature_columns
    
    encoded_data = {}
    for split_name, df in data.items():
        encoded = prepare_split_data(
            df, split_name, target_column, feature_columns, 
            label_mapping, text_encoding_config
        )
        
        encoded_data[split_name] = {
            "text": encoded["text"],
            "labels": encoded["labels"],
            "original_labels": encoded["original_labels"]
        }
    
    # 토크나이저 분석 (선택적)
    tokenizer_analysis = {}
    if stage_config.get("tokenizer_analysis", {}).get("enabled", False):
        all_texts = []
        all_labels = []
        
        for split_data in encoded_data.values():
            all_texts.extend(split_data['text'])
            all_labels.extend(split_data['labels'])
        
        tokenizer_analysis = perform_tokenizer_analysis(
            all_texts, all_labels,
            stage_config.get("tokenizer_analysis", {}),
            label_mapping
        )
    
    # 데이터 구조 변경: DataFrame → 인코딩된 데이터
    data = encoded_data
    
    # 메타데이터 업데이트 (누적) - 기존 메타데이터를 모두 보존
    metadata.update({
        "text_encoding_config": text_encoding_config,
        "tokenizer_analysis": tokenizer_analysis,
        "encoding_timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"✅ Stage 8 완료: 텍스트 인코딩 완료")
    
    # 처리 시간 및 데이터 크기
    processing_time = time.time() - start_time
    output_count = sum(len(d["text"]) for d in encoded_data.values())
    
    # 메모리 정리
    cleanup_memory("Stage 8")
    
    return data, prepare_metadata_for_json(metadata), processing_time, output_count


# =============================================================================
# Stage 9: 최종 패키징 및 분석
# =============================================================================

def stage_9_final_packaging(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    parameters: DictConfig
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Stage 9: 최종 데이터셋 패키징 및 전체 파이프라인 분석
    
    Args:
        data: {"train": {...}, "val": {...}, "test": {...}}
        metadata: 누적된 모든 메타데이터
        parameters: 파라미터 설정
        
    Returns:
        final_data: 최종 데이터셋
        final_metadata: 전체 메타데이터
        processing_time: 처리 시간(초)
    """
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("🚉 Stage 9: 최종 패키징 및 분석")
    logger.info("="*60)
    
    config = get_stage_config(parameters, "stage_9")
    stage_config = config["stage_config"]
    
    # 최종 데이터셋 구성 - train/val/test를 최상위로 이동
    final_data = {
        **data,  # train, val, test를 최상위로 풀어냄
        "dataset_info": {
            "creation_timestamp": datetime.now().isoformat(),
            "pipeline_version": stage_config.get("pipeline_version", "1.0.0"),
            "splits": list(data.keys()),
            "sample_counts": {k: len(v["text"]) for k, v in data.items()}
        }
    }
    
    # 파이프라인 실행 시간 계산
    pipeline_start_time = datetime.fromisoformat(metadata["pipeline_start_time"])
    end_time = datetime.now()
    execution_time = (end_time - pipeline_start_time).total_seconds()
    
    # 메타데이터 패키징 설정 가져오기
    packaging_config = stage_config.get("metadata_packaging", {})
    
    # dataset_all.pkl에 포함할 선택적 메타데이터 구성
    selected_metadata = {}
    
    # 라벨 매핑
    if packaging_config.get("include_label_mapping", True):
        selected_metadata["label_mapping"] = metadata.get("label_mapping", {})
        logger.info("✓ 라벨 매핑 포함")
    
    # 피처 칼럼 목록
    if packaging_config.get("include_feature_columns", True):
        selected_metadata["feature_columns"] = metadata.get("feature_columns", [])
        logger.info("✓ 피처 칼럼 목록 포함")
    
    # 칼럼명 매핑
    if packaging_config.get("include_columns_mapping", True):
        selected_metadata["columns_mapping"] = metadata.get("column_mapping", {})
        logger.info("✓ 칼럼명 매핑 포함")
    
    # 회사별 스케일링 통계 - 더 이상 사용하지 않음
    # if packaging_config.get("include_company_scaling_stats", False):
    #     selected_metadata["company_scaling_stats"] = metadata.get("company_scaling_stats", {})
    #     logger.info("✓ 회사별 스케일링 통계 포함")
    
    # 텍스트 인코딩 설정
    if packaging_config.get("include_text_encoding_config", True):
        selected_metadata["text_encoding_config"] = metadata.get("text_encoding_config", {})
        logger.info("✓ 텍스트 인코딩 설정 포함")
    
    # 타겟 칼럼명
    if packaging_config.get("include_target_column", True):
        selected_metadata["target_column"] = metadata.get("target_column", "")
        logger.info("✓ 타겟 칼럼명 포함")
    
    # 데이터 분할 정보
    if packaging_config.get("include_split_info", True):
        selected_metadata["split_info"] = metadata.get("split_stats", {})
        logger.info("✓ 데이터 분할 정보 포함")
    
    # Z-score 변환 파라미터
    if packaging_config.get("include_z_score_params", True):
        selected_metadata["z_score_params"] = metadata.get("z_score_stats", {})
        logger.info("✓ Z-score 변환 파라미터 포함")
    
    # dataset_all.pkl에 포함될 최종 데이터 구조
    final_data["metadata"] = selected_metadata
    
    logger.info(f"📦 선택된 메타데이터 항목 수: {len(selected_metadata)}")
    
    # 전체 메타데이터를 평탄하고 중복없는 구조로 재구성
    final_metadata = {
        # === 파이프라인 정보 ===
        "pipeline_version": stage_config.get("pipeline_version", "1.0.0"),
        "pipeline_start_time": metadata["pipeline_start_time"],
        "pipeline_end_time": datetime.now().isoformat(),
        "pipeline_execution_seconds": execution_time,
        "pipeline_total_stages": 10,
        
        # === 데이터 형상 정보 ===
        "data_original_shape": metadata.get("original_shape", {}),
        "data_final_shape": {
            "train": len(data.get("train", {}).get("text", [])),
            "validation": len(data.get("validation", {}).get("text", [])),
            "test": len(data.get("test", {}).get("text", []))
        },
        
        # === 컬럼 정보 (중복 제거) ===
        "columns_original": metadata.get("column_mapping", {}).get("original_columns", []),
        "columns_standardized": metadata.get("column_mapping", {}).get("standardized_columns", []),
        "columns_mapping": metadata.get("column_mapping", {}).get("rename_dict", {}),
        "columns_feature": metadata.get("feature_columns", []),
        "columns_target": metadata.get("target_column", ""),
        
        # === 라벨 정보 (중복 제거) ===
        "label_mapping": metadata.get("label_mapping", {}),
        "label_num_classes": metadata.get("label_mapping", {}).get("num_classes", 0),
        "label_num_base_classes": metadata.get("label_mapping", {}).get("num_base_classes", 0),
        "label_num_dummy_classes": metadata.get("label_mapping", {}).get("num_dummy_classes", 0),
        
        # === 데이터 분할 정보 ===
        "split_train_ratio": metadata.get("split_stats", {}).get("train_ratio", 0.0),
        "split_val_ratio": metadata.get("split_stats", {}).get("val_ratio", 0.0),
        "split_test_ratio": metadata.get("split_stats", {}).get("test_ratio", 0.0),
        "split_train_count": metadata.get("split_stats", {}).get("train_count", 0),
        "split_val_count": metadata.get("split_stats", {}).get("val_count", 0),
        "split_test_count": metadata.get("split_stats", {}).get("test_count", 0),
        
        # === 데이터 품질 정보 ===
        "quality_duplicate_rows": metadata.get("data_quality", {}).get("duplicate_rows", 0),
        "quality_issues": metadata.get("data_quality", {}).get("quality_issues", []),
        "quality_class_distribution": metadata.get("class_distribution", {}),
        
        # === 필터링 통계 ===
        "filter_stage2_removed_rows": metadata.get("stage_2_stats", {}).get("total_removed", 0),
        "filter_stage2_final_rows": metadata.get("stage_2_stats", {}).get("final_rows", 0),
        
        # === 피처 엔지니어링 정보 ===
        "feature_engineering_added": metadata.get("feature_engineering_summary", {}).get("features_added", 0),
        "feature_engineering_names": metadata.get("feature_engineering_summary", {}).get("new_features", []),
        "feature_outlier_ratio": metadata.get("outlier_stats", {}).get("outlier_ratio", 0.0),
        
        # === 텍스트 인코딩 정보 ===
        "encoding_method": metadata.get("text_encoding_config", {}).get("method", ""),
        "encoding_separator": metadata.get("text_encoding_config", {}).get("separator", ""),
        "encoding_missing_value": metadata.get("text_encoding_config", {}).get("missing_value", ""),
        "encoding_max_length": metadata.get("text_encoding_config", {}).get("max_length", 0),
        
        # === 스케일링 정보 - 더 이상 사용하지 않음 ===
        # "scaling_stats_available": False,
        # "scaling_num_companies": 0,
        
        # === 단계별 처리 시간 ===
        "stage_processing_times": {
            f"stage_{i}": metadata.get(f"stage_{i}_processing_time", 0.0) 
            for i in range(10)
        },
        
        # === 단계별 출력 행 수 ===
        "stage_output_counts": {
            f"stage_{i}": metadata.get(f"stage_{i}_output_count", 0) 
            for i in range(10)
        }
    }
    
    logger.info(f"📦 최종 데이터셋 패키징 완료")
    logger.info(f"⏱️ 전체 파이프라인 실행 시간: {execution_time:.2f}초")
    logger.info(f"✅ Stage 9 완료: 파이프라인 완료!")
    
    # 처리 시간
    processing_time = time.time() - start_time
    
    # 메모리 정리
    cleanup_memory("Stage 9")
    
    return final_data, prepare_metadata_for_json(final_metadata), processing_time
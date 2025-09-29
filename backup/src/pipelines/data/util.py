#!/usr/bin/env python3
"""
데이터 파이프라인 유틸리티 함수 통합
모든 스테이지에서 사용하는 유틸리티 함수들을 한 곳에 모음
"""

import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
from datetime import datetime
import datetime as dt
from omegaconf import DictConfig
import holidays

logger = logging.getLogger(__name__)


# =============================================================================
# Config 유틸리티
# =============================================================================

def get_stage_config(parameters: DictConfig, stage_name: str) -> Dict[str, Any]:
    """
    스테이지별 설정 추출
    
    Args:
        parameters: 전체 파라미터
        stage_name: 스테이지 이름 (예: "stage_0")
        
    Returns:
        스테이지 설정 딕셔너리
    """
    stage_config = parameters.get(stage_name, {})
    
    return {
        "stage_config": stage_config,
        "mlflow_enabled": parameters.get("mlflow", {}).get("enabled", True),
        "local_base_path": parameters.get("paths", {}).get("local_base", "data/pipeline_metadata"),
        "artifact_paths": {
            "data": parameters.get("paths", {}).get("mlflow_artifacts", {}).get("data", "data"),
            "metadata": parameters.get("paths", {}).get("mlflow_artifacts", {}).get("metadata", "metadata")
        },
        "save_options": stage_config.get("save_options", {})
    }


# =============================================================================
# Data 유틸리티
# =============================================================================

def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """
    필수 컬럼 존재 여부 검증
    
    Args:
        df: 검증할 데이터프레임
        required_columns: 필수 컬럼 리스트
        
    Returns:
        누락된 컬럼 리스트
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns


def get_column_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    컬럼별 통계 정보 추출
    
    Args:
        df: 데이터프레임
        
    Returns:
        컬럼별 통계 딕셔너리
    """
    stats = {}
    
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "null_ratio": float(df[col].isnull().mean()),
            "unique_count": int(df[col].nunique())
        }
        
        # 숫자형 컬럼에 대한 추가 통계
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                col_stats.update({
                    "mean": float(non_null_values.mean()),
                    "std": float(non_null_values.std()),
                    "min": float(non_null_values.min()),
                    "max": float(non_null_values.max()),
                    "median": float(non_null_values.median()),
                    "q1": float(non_null_values.quantile(0.25)),
                    "q3": float(non_null_values.quantile(0.75))
                })
        
        stats[col] = col_stats
    
    return stats


def apply_row_filters(df: pd.DataFrame, filter_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    행 필터링 적용
    
    Args:
        df: 입력 데이터프레임
        filter_config: 필터링 설정
        
    Returns:
        (필터링된 데이터프레임, 필터링 통계)
    """
    initial_rows = len(df)
    filtered_df = df.copy()
    filter_stats = {
        "initial_rows": initial_rows,
        "filters_applied": []
    }
    
    # 행 필터 적용
    row_filters = filter_config.get("row_filters", [])
    for filter_rule in row_filters:
        column = filter_rule.get("column")
        operator = filter_rule.get("operator", "==")
        value = filter_rule.get("value")
        
        if column in filtered_df.columns:
            before_rows = len(filtered_df)
            
            if operator == "==":
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == "!=":
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == ">":
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == "<":
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == ">=":
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == "<=":
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == "isin":
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif operator == "not_null":
                filtered_df = filtered_df[filtered_df[column].notna()]
            elif operator == "is_null":
                filtered_df = filtered_df[filtered_df[column].isna()]
            
            removed_rows = before_rows - len(filtered_df)
            filter_stats["filters_applied"].append({
                "column": column,
                "operator": operator,
                "value": value,
                "rows_removed": removed_rows
            })
            
            logger.info(f"필터 적용: {column} {operator} {value} → {removed_rows}개 행 제거")
    
    filter_stats["final_rows"] = len(filtered_df)
    filter_stats["total_removed"] = initial_rows - len(filtered_df)
    filter_stats["removal_ratio"] = (initial_rows - len(filtered_df)) / initial_rows if initial_rows > 0 else 0
    
    return filtered_df, filter_stats


def calculate_class_distribution(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    클래스 분포 계산
    
    Args:
        df: 데이터프레임
        target_column: 타겟 컬럼명
        
    Returns:
        클래스 분포 정보
    """
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}
    
    value_counts = df[target_column].value_counts()
    total_samples = len(df)
    
    distribution = {
        "num_classes": len(value_counts),
        "total_samples": total_samples,
        "class_counts": value_counts.to_dict(),
        "class_ratios": (value_counts / total_samples).to_dict(),
        "min_class_samples": int(value_counts.min()),
        "max_class_samples": int(value_counts.max()),
        "class_imbalance_ratio": float(value_counts.max() / value_counts.min()) if value_counts.min() > 0 else float('inf')
    }
    
    # 상위 10개 클래스
    distribution["top_10_classes"] = value_counts.head(10).to_dict()
    
    # 희소 클래스 (전체의 1% 미만)
    rare_threshold = total_samples * 0.01
    rare_classes = value_counts[value_counts < rare_threshold]
    distribution["rare_classes"] = {
        "count": len(rare_classes),
        "samples": int(rare_classes.sum()),
        "ratio": float(rare_classes.sum() / total_samples) if total_samples > 0 else 0
    }
    
    return distribution


# =============================================================================
# Label Mapping 유틸리티
# =============================================================================

def create_label_mapping(
    data: pd.DataFrame,
    target_column: str,
    total_num_classes: int = 300,
    dummy_prefix: str = "DUMMY_",
    mode: str = "auto",
    existing_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    라벨 매핑 생성 (더미 클래스 포함)
    
    Args:
        data: 입력 데이터
        target_column: 타겟 컬럼명
        total_num_classes: 총 클래스 수
        dummy_prefix: 더미 클래스 접두사
        mode: 생성 모드 ("auto", "create_new", "update")
        existing_path: 기존 매핑 파일 경로
        
    Returns:
        라벨 매핑 딕셔너리
    """
    if target_column not in data.columns:
        logger.warning(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        return {
            "text2id": {},
            "id2text": {},
            "vocabulary": [],
            "num_classes": 0,
            "num_base_classes": 0,
            "num_dummy_classes": 0,
            "target_column": target_column,
            "creation_timestamp": datetime.now().isoformat(),
            "mode": mode
        }
    
    # 현재 데이터의 고유 클래스
    unique_labels = sorted([str(label) for label in data[target_column].unique()])
    current_class_count = len(unique_labels)
    logger.info(f"📊 현재 데이터에서 {current_class_count}개의 고유 클래스 발견")
    
    # mode가 auto인 경우 existing_path 존재 여부로 결정
    if mode == "auto":
        mode = "update" if existing_path and os.path.exists(existing_path) else "create_new"
    
    if mode == "update" and existing_path and os.path.exists(existing_path):
        # 기존 매핑 로드 및 업데이트
        logger.info(f"🔄 기존 매핑 파일 로드 및 업데이트: {existing_path}")
        
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing_mapping = json.load(f)
        
        # 기존 실제 클래스 추출 (ID 순서대로)
        existing_real_labels = []
        for label, id_val in sorted(existing_mapping['text2id'].items(), key=lambda x: x[1]):
            if not label.startswith(dummy_prefix):
                existing_real_labels.append(label)
        
        # 신규 클래스 찾기
        new_labels = [label for label in unique_labels if label not in existing_real_labels]
        if new_labels:
            logger.info(f"🆕 신규 클래스 발견: {new_labels}")
        
        # 전체 실제 클래스 = 기존 + 신규 (순서 유지)
        # 중요: 기존 클래스의 ID는 절대 변경하지 않음
        all_real_labels = existing_real_labels + sorted(new_labels)
        base_class_count = len(all_real_labels)
        
        # 더미 클래스 재생성
        num_dummy = max(0, total_num_classes - base_class_count)
        dummy_classes = [f"{dummy_prefix}{i:03d}" for i in range(num_dummy)]
        
        # 전체 라벨 리스트
        all_labels = all_real_labels + dummy_classes
        
        # 라벨 매핑 재생성
        label_mapping = {
            "text2id": {str(label): int(idx) for idx, label in enumerate(all_labels)},
            "id2text": {int(idx): str(label) for idx, label in enumerate(all_labels)},
            "vocabulary": all_labels,
            "num_classes": int(len(all_labels)),
            "num_base_classes": int(base_class_count),
            "num_dummy_classes": int(num_dummy),
            "target_column": target_column,
            "creation_timestamp": existing_mapping.get("creation_timestamp", datetime.now().isoformat()),
            "update_timestamp": datetime.now().isoformat(),
            "mode": "updated",
            "new_classes_added": new_labels,
            "existing_classes_count": len(existing_real_labels)
        }
        
        logger.info(f"✅ 라벨 매핑 업데이트 완료: 총 {base_class_count}개 실제 클래스 ({len(new_labels)}개 신규)")
        logger.info(f"📋 기존 클래스 ID 유지, 신규 클래스만 추가")
        
    else:
        # 새로 생성
        logger.info("🆕 새로운 라벨 매핑 생성")
        base_class_count = current_class_count
        
        # 더미 클래스 생성
        num_dummy = max(0, total_num_classes - base_class_count)
        dummy_classes = [f"{dummy_prefix}{i:03d}" for i in range(num_dummy)]
        all_labels = unique_labels + dummy_classes
        
        # 라벨 매핑 딕셔너리 생성
        label_mapping = {
            "text2id": {str(label): int(idx) for idx, label in enumerate(all_labels)},
            "id2text": {int(idx): str(label) for idx, label in enumerate(all_labels)},
            "vocabulary": all_labels,
            "num_classes": int(len(all_labels)),
            "num_base_classes": int(base_class_count),
            "num_dummy_classes": int(num_dummy),
            "target_column": target_column,
            "creation_timestamp": datetime.now().isoformat(),
            "mode": "created"
        }
    
    return label_mapping


# =============================================================================
# Feature Engineering 유틸리티
# =============================================================================

def apply_robust_standardization(
    data: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Robust standardization 적용
    
    Args:
        data: 입력 데이터
        config: robust_standardization 설정
        
    Returns:
        (처리된 데이터, 회사별 통계, 피처 엔지니어링 요약, 이상치 통계)
    """
    engineered_data = data.copy()
    company_stats = {}
    outlier_stats = {}
    feature_engineering_summary = {
        "features_added": 0,
        "new_features": [],
        "total_features": len(engineered_data.columns)
    }
    
    if config.get("enable", False):
        group_col = config.get("group_column", "co_vat_id")
        target_col = config.get("target_column", "pur_total")
        
        if group_col in engineered_data.columns and target_col in engineered_data.columns:
            # 피처 엔지니어링 실행
            new_feature, company_stats = optimized_robust_standardization_with_minmax(
                df=engineered_data,
                group_col=group_col,
                target_col=target_col,
                outlier_method=config.get("outlier_bounds", {}).get("method", "iqr"),
                iqr_multiplier=config.get("outlier_bounds", {}).get("iqr_multiplier", 1.5),
                z_score_threshold=config.get("outlier_bounds", {}).get("z_score_threshold", 3.0),
                percentile_range=config.get("outlier_bounds", {}).get("percentile_range", [5, 95]),
                min_group_size=config.get("min_group_size", 3),
                min_normal_size=config.get("min_normal_size", 2),
                scaling_range=config.get("scaling_range", [1, 100])
            )
            
            # 새 칼럼 추가
            output_suffix = config.get("output_column_suffix", "_scaled")
            new_column_name = f"{target_col}{output_suffix}"
            engineered_data[new_column_name] = new_feature
            
            # 통계 업데이트
            feature_engineering_summary["features_added"] = 1
            feature_engineering_summary["new_features"] = [new_column_name]
            feature_engineering_summary["total_features"] = len(engineered_data.columns)
            
            # 이상치 통계
            total_outliers = sum(stat.get("outlier_removed", 0) for stat in company_stats.values())
            total_samples = sum(stat.get("sample_count", 0) for stat in company_stats.values())
            outlier_stats = {
                "total_outliers_removed": total_outliers,
                "total_samples_processed": total_samples,
                "outlier_ratio": total_outliers / total_samples if total_samples > 0 else 0,
                "companies_processed": len(company_stats)
            }
    
    return engineered_data, company_stats, feature_engineering_summary, outlier_stats


def optimized_robust_standardization_with_minmax(
    df: pd.DataFrame, 
    group_col: str, 
    target_col: str,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
    percentile_range: List[float] = [5, 95],
    min_group_size: int = 3,
    min_normal_size: int = 2,
    scaling_range: List[float] = [1, 100]
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    최적화된 그룹별 이상치 강건 표준화 + MinMax 변환
    """
    logger.info(f"🔧 피처 엔지니어링 시작: {group_col} → {target_col}")
    start_time = time.time()
    
    # 결과 시리즈 초기화
    result = pd.Series(index=df.index, dtype='float64')
    
    # 회사별 통계 저장용 딕셔너리
    company_stats = {}
    
    # 그룹별 처리
    processed_groups = 0
    processed_rows = 0
    
    for group_name, group_data in df.groupby(group_col):
        group_size = len(group_data)
        
        if group_size < min_group_size:
            continue
            
        series = group_data[target_col]
        
        # 절대값 변환 (이상치 탐지용)
        series_abs = series.abs()
        
        # 이상치 탐지
        if outlier_method == "iqr":
            Q1 = series_abs.quantile(0.25)
            Q3 = series_abs.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                result.loc[group_data.index] = (scaling_range[0] + scaling_range[1]) / 2
                processed_groups += 1
                processed_rows += group_size
                continue
            
            normal_mask = (series_abs >= Q1 - iqr_multiplier * IQR) & (series_abs <= Q3 + iqr_multiplier * IQR)
            
        elif outlier_method == "z_score":
            mean_val = series_abs.mean()
            std_val = series_abs.std()
            
            if std_val == 0:
                result.loc[group_data.index] = (scaling_range[0] + scaling_range[1]) / 2
                processed_groups += 1
                processed_rows += group_size
                continue
                
            z_scores = np.abs((series_abs - mean_val) / std_val)
            normal_mask = z_scores <= z_score_threshold
            
        elif outlier_method == "percentile":
            lower_bound = series_abs.quantile(percentile_range[0] / 100)
            upper_bound = series_abs.quantile(percentile_range[1] / 100)
            normal_mask = (series_abs >= lower_bound) & (series_abs <= upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        normal_values = series_abs[normal_mask]
        
        if len(normal_values) < min_normal_size:
            continue
            
        # 정상 데이터로 표준화 파라미터 학습
        mean_val = normal_values.mean()
        std_val = normal_values.std()
        
        if std_val == 0:
            result.loc[group_data.index] = (scaling_range[0] + scaling_range[1]) / 2
            processed_groups += 1
            processed_rows += group_size
            continue
        
        # 회사별 통계 저장
        company_stats[str(group_name)] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'sample_count': int(len(normal_values)),
            'outlier_method': outlier_method,
            'outlier_removed': int(group_size - len(normal_values))
        }
        
        # 표준화 적용 (원본 데이터의 절대값에 대해)
        standardized = (series_abs - mean_val) / std_val
        result.loc[group_data.index] = standardized.astype(np.float64)
        
        processed_groups += 1
        processed_rows += group_size
    
    # 선형변환 적용 (1-100)
    valid_mask = result.notna()
    if valid_mask.sum() > 0:
        valid_z_scores = result[valid_mask]
        
        # Z-score를 1-100 범위로 연속적으로 매핑
        # 실제 데이터 특성에 맞춘 범위: z-score -1 → 1, z-score 5 → 100
        # 절대값 변환 후 z-score 분포에 적합한 범위
        linear_scaled = 1 + (valid_z_scores - (-1)) * (100 - 1) / (5 - (-1))
        
        # 1-100 범위로 클리핑 (범위를 초과하는 값들은 1 또는 100으로 처리)
        linear_scaled = np.clip(linear_scaled, scaling_range[0], scaling_range[1])
        
        # 정수로 반올림하여 소수점 제거
        result.loc[valid_mask] = np.round(linear_scaled).astype(int)
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"✅ 피처 엔지니어링 완료: {processed_groups} 그룹, {processed_rows} 행 ({elapsed_time:.2f}초)")
    
    return result, company_stats


# =============================================================================
# Memory Optimization 유틸리티
# =============================================================================

def optimize_dataframe_dtypes(df: pd.DataFrame, 
                            categorical_threshold: float = 0.5,
                            exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    DataFrame의 데이터 타입을 최적화하여 메모리 사용량 감소
    
    Args:
        df: 최적화할 DataFrame
        categorical_threshold: 카테고리로 변환할 unique ratio 임계값
        exclude_columns: 최적화에서 제외할 컬럼 리스트
        
    Returns:
        최적화된 DataFrame
    """
    exclude_columns = exclude_columns or []
    
    # 초기 메모리 사용량
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"📊 최적화 전 메모리: {initial_memory:.2f} MB")
    
    # 1. 정수형 다운캐스팅
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        if col not in exclude_columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # int8로 변환 가능한지 확인
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
            # int16으로 변환 가능한지 확인
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
            # int32로 변환 가능한지 확인
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
    
    # 2. 실수형 다운캐스팅
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        if col not in exclude_columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 3. 문자열을 카테고리로 변환
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        if col not in exclude_columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            unique_ratio = unique_count / total_count
            
            # unique 비율이 임계값 이하면 카테고리로 변환
            if unique_ratio < categorical_threshold:
                df[col] = df[col].astype('category')
                logger.debug(f"  {col}: object → category (unique: {unique_count:,})")
    
    # 최종 메모리 사용량
    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction_pct = (initial_memory - final_memory) / initial_memory * 100
    
    logger.info(f"📊 최적화 후 메모리: {final_memory:.2f} MB (감소율: {reduction_pct:.1f}%)")
    
    return df


def reduce_metadata_size(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    메타데이터 크기 줄이기
    
    Args:
        metadata: 원본 메타데이터
        
    Returns:
        축소된 메타데이터
    """
    reduced_metadata = {}
    
    for key, value in metadata.items():
        # 큰 리스트는 요약 정보만 저장
        if isinstance(value, list) and len(value) > 100:
            reduced_metadata[key] = {
                "type": "list",
                "length": len(value),
                "sample": value[:5] if len(value) > 5 else value
            }
        # DataFrame은 shape 정보만 저장
        elif isinstance(value, pd.DataFrame):
            reduced_metadata[key] = {
                "type": "DataFrame",
                "shape": value.shape,
                "columns": list(value.columns)[:10]  # 처음 10개 컬럼만
            }
        # 딕셔너리는 재귀적으로 처리
        elif isinstance(value, dict):
            reduced_metadata[key] = reduce_metadata_size(value)
        else:
            reduced_metadata[key] = value
    
    return reduced_metadata


def get_dataframe_memory_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    DataFrame의 메모리 사용 정보 반환
    
    Args:
        df: 분석할 DataFrame
        
    Returns:
        메모리 정보 딕셔너리
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024 / 1024
    
    # 컬럼별 메모리 사용량 (상위 10개)
    column_memory = memory_usage.drop('Index').sort_values(ascending=False).head(10)
    column_memory_mb = column_memory / 1024 / 1024
    
    # 데이터 타입별 메모리 사용량
    dtype_memory = {}
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        dtype_memory[str(dtype)] = df[cols].memory_usage(deep=True).sum() / 1024 / 1024
    
    return {
        "total_memory_mb": total_memory,
        "shape": df.shape,
        "top_memory_columns": column_memory_mb.to_dict(),
        "memory_by_dtype": dtype_memory,
        "dtype_counts": df.dtypes.value_counts().to_dict()
    }


# =============================================================================
# Text Encoding 유틸리티
# =============================================================================

def prepare_split_data(
    df: pd.DataFrame, 
    split_name: str,
    target_column: str,
    feature_columns: List[str],
    label_mapping: Dict[str, Any],
    text_encoding_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    분할 데이터 준비 및 텍스트 인코딩
    
    Args:
        df: 입력 데이터프레임
        split_name: 분할 이름 (train/validation/test)
        target_column: 타겟 컬럼명
        feature_columns: 피처 컬럼 리스트
        label_mapping: 라벨 매핑 정보
        text_encoding_config: 텍스트 인코딩 설정
        
    Returns:
        준비된 데이터 딕셔너리
    """
    if len(df) == 0:
        return {'text': [], 'labels': [], 'original_labels': []}
    
    # 텍스트 인코딩 설정 추출
    encoding_method = text_encoding_config.get("encoding_method", "sequential")
    
    if encoding_method == "sequential":
        sequential_config = text_encoding_config.get("sequential", {})
        separator = sequential_config.get("separator", "^ ")
        column_value_separator = sequential_config.get("column_value_separator", ":")
        include_column_names = sequential_config.get("include_column_names", True)
        null_replacement = sequential_config.get("null_replacement", "None")
        # 결측값 패턴들 가져오기
        missing_patterns = text_encoding_config.get("missing_value_patterns", [
            "N/A", "NULL", "nan", "NaN", "null", "None", "none", 
            "#N/A", "#DIV/0!", "<NA>", "NaT", ""
        ])
    else:
        # 기본값
        separator = "^ "
        column_value_separator = ":"
        include_column_names = True
        null_replacement = "None"
        missing_patterns = [
            "N/A", "NULL", "nan", "NaN", "null", "None", "none", 
            "#N/A", "#DIV/0!", "<NA>", "NaT", ""
        ]
    
    # 피처 컬럼 결정
    if feature_columns:
        # feature_columns에서 실제 존재하는 컬럼만 사용
        feat_cols = [col for col in feature_columns if col in df.columns]
        if not feat_cols:
            # feature_columns가 모두 매칭되지 않으면 타겟 제외한 모든 컬럼 사용
            logger.warning(f"⚠️ feature_columns가 데이터와 매칭되지 않음. 모든 컬럼 사용.")
            feat_cols = [col for col in df.columns if col != target_column]
    else:
        feat_cols = [col for col in df.columns if col != target_column]
    
    # 텍스트 변환 (벡터화된 방식)
    texts = create_text_representation_vectorized(
        df, feat_cols, separator, include_column_names, null_replacement, missing_patterns, column_value_separator
    )
    
    # 라벨 변환
    if target_column in df.columns:
        original_labels = df[target_column].astype(str).tolist()
        text2id = label_mapping.get('text2id', {})
        labels = [text2id.get(str(label), 0) for label in original_labels]
    else:
        # 타겟 컬럼이 없는 경우 더미 라벨 사용
        logger.warning(f"⚠️ 타겟 컬럼 '{target_column}'이 데이터에 없음. 더미 라벨 사용.")
        original_labels = ['UNKNOWN'] * len(df)
        labels = [0] * len(df)
    
    logger.info(f"   {split_name}: {len(texts):,} 샘플")
    
    return {
        'text': texts,
        'labels': labels,
        'original_labels': original_labels
    }


def create_text_representation(
    row: pd.Series, 
    feature_columns: List[str],
    separator: str = "^ ",
    include_column_names: bool = True,
    null_replacement: str = "None",
    column_value_separator: str = ":"
) -> str:
    """
    피처 값들을 텍스트로 결합
    
    Args:
        row: 데이터 행
        feature_columns: 피처 컬럼 리스트
        separator: 구분자
        include_column_names: 컬럼명 포함 여부
        null_replacement: NULL 값 대체 문자열
        column_value_separator: 칼럼명과 값 사이의 구분자
        
    Returns:
        텍스트 표현
    """
    text_parts = []
    
    for col in feature_columns:
        value = row[col]
        
        # 결측값 처리
        if pd.isna(value) or value == '' or value is None:
            value_str = null_replacement
        else:
            # float 타입이고 정수값인 경우 정수로 변환
            if isinstance(value, float) and pd.notna(value) and value.is_integer():
                value_str = str(int(value))
            else:
                value_str = str(value)
        
        # 컬럼명 포함 여부에 따라 처리
        if include_column_names:
            text_parts.append(f"{col}{column_value_separator}{value_str}")
        else:
            text_parts.append(value_str)
    
    return separator.join(text_parts)


def perform_tokenizer_analysis(
    texts: List[str],
    labels: List[int],
    tokenizer_analysis_config: Dict[str, Any],
    label_mapping: Dict[str, Any]
) -> Dict[str, Any]:
    """
    토크나이저 분석 수행 (선택적)
    
    Args:
        texts: 텍스트 리스트
        labels: 라벨 리스트
        tokenizer_analysis_config: 토크나이저 분석 설정
        label_mapping: 라벨 매핑 정보
        
    Returns:
        분석 결과
    """
    tokenizer_analysis = {}
    
    if not tokenizer_analysis_config.get("enabled", False):
        return tokenizer_analysis
    
    try:
        # 토크나이저 이름 가져오기
        tokenizer_name = tokenizer_analysis_config.get("tokenizer_name", "Qwen/Qwen3-4B")
        sample_size = tokenizer_analysis_config.get("sample_size", 1000)
        
        # 샘플링
        if sample_size > 0 and sample_size < len(texts):
            indices = np.random.choice(len(texts), sample_size, replace=False)
            sample_texts = [texts[i] for i in indices]
            sample_labels = [labels[i] for i in indices]
        else:
            sample_texts = texts
            sample_labels = labels
        
        # 실제 토크나이저 사용 여부 체크
        use_actual_tokenizer = tokenizer_analysis_config.get("use_actual_tokenizer", False)
        
        if use_actual_tokenizer:
            try:
                # 실제 토크나이저 로드 (선택적)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                
                # 토크나이징 및 분석
                tokenized_texts = []
                token_lengths = []
                
                for text in sample_texts[:10]:  # 처음 10개만 상세 분석
                    tokens = tokenizer.tokenize(text)
                    tokenized_texts.append({
                        'original': text,  # 전체 텍스트 표시
                        'tokens': tokens,  # 모든 토큰 표시
                        'token_count': len(tokens)
                    })
                
                # 전체 샘플에 대한 토큰 길이 계산
                for text in sample_texts:
                    tokens = tokenizer.tokenize(text)
                    token_lengths.append(len(tokens))
                
                tokenizer_analysis['tokenizer_name'] = tokenizer_name
                tokenizer_analysis['vocab_size'] = tokenizer.vocab_size
                tokenizer_analysis['tokenization_samples'] = tokenized_texts
                
            except Exception as e:
                logger.warning(f"실제 토크나이저 로드 실패, 단순 분석으로 대체: {e}")
                use_actual_tokenizer = False
        
        if not use_actual_tokenizer:
            # 간단한 공백 기반 토큰 분석
            token_lengths = [len(text.split()) for text in sample_texts]
            
            # 샘플 토크나이징 (공백 기반)
            tokenization_samples = []
            for i, text in enumerate(sample_texts[:5]):  # 처음 5개만
                words = text.split()
                tokenization_samples.append({
                    'index': i,
                    'original': text,  # 전체 텍스트 표시
                    'word_tokens': words,  # 모든 단어 표시
                    'word_count': len(words),
                    'label': label_mapping.get('id2text', {}).get(str(sample_labels[i]), str(sample_labels[i]))
                })
            
            tokenizer_analysis['tokenizer_name'] = 'whitespace_tokenizer'
            tokenizer_analysis['tokenization_samples'] = tokenization_samples
        
        # 토큰 길이 통계
        percentiles = tokenizer_analysis_config.get("token_length_stats", {}).get("percentiles", [25, 50, 75, 90, 95, 99])
        
        tokenizer_analysis['token_length_stats'] = {
            'sample_size': len(sample_texts),
            'mean': float(np.mean(token_lengths)),
            'std': float(np.std(token_lengths)),
            'min': int(np.min(token_lengths)),
            'max': int(np.max(token_lengths)),
            'percentiles': {f'p{p}': int(np.percentile(token_lengths, p)) for p in percentiles}
        }
        
        # 토큰 길이 분포
        tokenizer_analysis['token_length_distribution'] = {
            '0-50': sum(1 for l in token_lengths if l <= 50),
            '51-100': sum(1 for l in token_lengths if 50 < l <= 100),
            '101-200': sum(1 for l in token_lengths if 100 < l <= 200),
            '201-300': sum(1 for l in token_lengths if 200 < l <= 300),
            '301-400': sum(1 for l in token_lengths if 300 < l <= 400),
            '401-500': sum(1 for l in token_lengths if 400 < l <= 500),
            '500+': sum(1 for l in token_lengths if l > 500)
        }
        
        logger.info(f"🔤 토크나이저 분석 완료 (샘플 크기: {len(sample_texts)})")
        
    except Exception as e:
        logger.warning(f"토크나이저 분석 실패: {e}")
        tokenizer_analysis = {'error': str(e)}
    
    return tokenizer_analysis


def create_text_representation_vectorized(
    df: pd.DataFrame, 
    feature_columns: List[str],
    separator: str = "^ ",
    include_column_names: bool = True,
    null_replacement: str = "None",
    missing_patterns: List[str] = None,
    column_value_separator: str = ":"
) -> List[str]:
    """
    벡터화된 텍스트 표현 생성 (성능 최적화)
    
    Args:
        df: 데이터프레임
        feature_columns: 피처 컬럼 리스트
        separator: 구분자
        include_column_names: 컬럼명 포함 여부
        null_replacement: NULL 값 대체 문자열
        missing_patterns: 결측값으로 인식할 패턴들
        
    Returns:
        텍스트 표현 리스트
    """
    # 1. 결측값과 타입 처리를 벡터화로 수행
    df_processed = df[feature_columns].copy()
    
    # 2. 결측값 패턴 설정
    if missing_patterns is None:
        missing_patterns = [
            "N/A", "NULL", "nan", "NaN", "null", "None", "none", 
            "#N/A", "#DIV/0!", "<NA>", "NaT", ""
        ]
    
    # 3. 모든 컬럼을 문자열로 변환하면서 결측값 처리
    
    for col in feature_columns:
        series = df_processed[col]
        
        # 먼저 문자열로 변환
        series_str = series.astype(str)
        
        # 모든 결측값 패턴들을 null_replacement('None')로 대체 - 벡터화된 방식
        # replace를 한 번에 처리하여 성능 개선
        series_str = series_str.replace(missing_patterns, null_replacement)
        
        # float 타입의 정수값 처리
        if series.dtype in ['float64', 'float32']:
            # 원본에서 결측값이 아닌 것들만 처리
            mask = series.notna()
            if mask.any():
                # 정수로 변환 가능한 값들 처리
                float_values = series[mask]
                int_mask = (float_values == float_values.astype(int))
                series_str.loc[mask & int_mask] = float_values[int_mask].astype(int).astype(str)
        
        df_processed[col] = series_str
    
    # 4. 텍스트 결합 (벡터화)
    if include_column_names:
        # 컬럼명:값 형태로 변환 후 결합
        text_columns = []
        for col in feature_columns:
            text_columns.append(col + column_value_separator + df_processed[col])
        
        # 모든 컬럼을 separator로 결합
        result = pd.Series([separator.join(row) for row in zip(*text_columns)])
    else:
        # 값만 결합
        result = df_processed.apply(lambda row: separator.join(row), axis=1)
    
    return result.tolist()


# =============================================================================
# JSON 직렬화 유틸리티
# =============================================================================

def convert_to_json_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 타입으로 재귀적으로 변환
    
    Args:
        obj: 변환할 객체
        
    Returns:
        JSON 직렬화 가능한 객체
    """
    # 기본 타입은 그대로 반환
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # numpy 정수
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    
    # numpy 실수
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    
    # numpy 배열
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # datetime
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    
    # pandas Timestamp
    if hasattr(obj, 'isoformat') and callable(obj.isoformat):
        return obj.isoformat()
    
    # set, tuple
    if isinstance(obj, (set, tuple)):
        return list(convert_to_json_serializable(item) for item in obj)
    
    # list
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    
    # dict
    if isinstance(obj, dict):
        return {
            str(key): convert_to_json_serializable(value) 
            for key, value in obj.items()
        }
    
    # 기타 객체는 문자열로 변환
    return str(obj)


def prepare_metadata_for_json(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    메타데이터 딕셔너리를 JSON 저장용으로 준비
    
    Args:
        metadata: 원본 메타데이터
        
    Returns:
        JSON 직렬화 가능한 메타데이터
    """
    return convert_to_json_serializable(metadata)


def extract_stage_metadata_delta(
    current_metadata: Dict[str, Any], 
    previous_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    현재 스테이지에서 추가된 메타데이터만 추출
    
    Args:
        current_metadata: 현재 스테이지의 전체 메타데이터
        previous_metadata: 이전 스테이지의 메타데이터 (없으면 None)
        
    Returns:
        현재 스테이지에서 추가/변경된 메타데이터만 포함하는 딕셔너리
    """
    if previous_metadata is None:
        # Stage 0의 경우 모든 메타데이터가 새로운 것
        return current_metadata
    
    delta = {}
    
    for key, value in current_metadata.items():
        # 새로운 키이거나 값이 변경된 경우
        if key not in previous_metadata:
            delta[key] = value
        elif previous_metadata[key] != value:
            # 값이 변경된 경우
            if isinstance(value, dict) and isinstance(previous_metadata[key], dict):
                # 딕셔너리인 경우 재귀적으로 변경사항 추출
                nested_delta = {}
                for nested_key, nested_value in value.items():
                    if nested_key not in previous_metadata[key] or previous_metadata[key][nested_key] != nested_value:
                        nested_delta[nested_key] = nested_value
                if nested_delta:
                    delta[key] = nested_delta
            else:
                delta[key] = value
    
    return delta


def json_encoder(o: Any) -> Any:
    """
    numpy 타입과 datetime을 JSON 직렬화 가능한 타입으로 변환
    (json.dump의 default 파라미터용)
    
    Args:
        o: 변환할 객체
        
    Returns:
        JSON 직렬화 가능한 타입
    """
    # numpy 정수 타입
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    
    # numpy 실수 타입
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    
    # numpy 배열
    if isinstance(o, np.ndarray):
        return o.tolist()
    
    # datetime 타입
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    
    # pandas Timestamp
    if hasattr(o, 'isoformat'):
        return o.isoformat()
    
    # set, tuple을 list로
    if isinstance(o, (set, tuple)):
        return list(o)
    
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# =============================================================================
# 날짜 관련 피처 엔지니어링
# =============================================================================

def add_day_type_feature(df: pd.DataFrame, date_column: str = 'txn_date') -> pd.DataFrame:
    """
    날짜 타입 피처 추가 (weekday/weekend/holiday)
    
    우선순위: weekend > holiday > weekday
    
    Args:
        df: 입력 DataFrame
        date_column: 날짜 컬럼명
        
    Returns:
        day_type 컬럼이 추가된 DataFrame
    """
    if date_column not in df.columns:
        logger.warning(f"날짜 컬럼 '{date_column}'이 존재하지 않습니다.")
        return df
    
    df_copy = df.copy()
    
    # datetime 변환
    df_copy['_dt'] = pd.to_datetime(df_copy[date_column])
    
    # 한국 공휴일 객체 생성 (2020-2025)
    kr_holidays = holidays.KR(years=range(2020, 2026))
    
    # 요일 계산 (0=월요일, 6=일요일)
    df_copy['_dow'] = df_copy['_dt'].dt.dayofweek
    
    # day_type 결정
    def get_day_type(row):
        if row['_dow'] >= 5:  # 토요일(5) 또는 일요일(6)
            return 'weekend'
        elif row['_dt'].date() in kr_holidays:
            return 'holiday'
        else:
            return 'weekday'
    
    df_copy['day_type'] = df_copy.apply(get_day_type, axis=1)
    
    # 임시 컬럼 제거
    df_copy = df_copy.drop(columns=['_dt', '_dow'])
    
    # 통계 로깅
    counts = df_copy['day_type'].value_counts()
    logger.info(f"✅ day_type 피처 추가: {counts.to_dict()}")
    
    return df_copy


# =============================================================================
# Z-Score 변환 함수
# =============================================================================

def remove_outliers_iqr(series: pd.Series, iqr_coefficient: float = 1.5, 
                       remove_lower: bool = True, remove_upper: bool = True,
                       lower_quantile: float = 0.25, upper_quantile: float = 0.75,
                       iqr_coefficient_lower: float = None, iqr_coefficient_upper: float = None) -> pd.Series:
    """
    IQR 방법으로 이상치 제거
    
    Args:
        series: 이상치를 제거할 시리즈
        iqr_coefficient: IQR 계수 (기본 1.5) - 하위 호환성을 위해 유지
        remove_lower: 하한 이상치 제거 여부
        remove_upper: 상한 이상치 제거 여부
        lower_quantile: 하위 분위수 (기본 0.25)
        upper_quantile: 상위 분위수 (기본 0.75)
        iqr_coefficient_lower: 하한 IQR 계수 (None일 경우 iqr_coefficient 사용)
        iqr_coefficient_upper: 상한 IQR 계수 (None일 경우 iqr_coefficient 사용)
        
    Returns:
        이상치가 제거된 시리즈
    """
    Q1 = series.quantile(lower_quantile)
    Q3 = series.quantile(upper_quantile)
    IQR = Q3 - Q1
    
    # 하위 호환성: 개별 계수가 없으면 기본 계수 사용
    if iqr_coefficient_lower is None:
        iqr_coefficient_lower = iqr_coefficient
    if iqr_coefficient_upper is None:
        iqr_coefficient_upper = iqr_coefficient
    
    result = series
    
    if remove_lower:
        lower_bound = Q1 - iqr_coefficient_lower * IQR
        result = result[result >= lower_bound]
    
    if remove_upper:
        upper_bound = Q3 + iqr_coefficient_upper * IQR
        result = result[result <= upper_bound]
    
    return result


def apply_z_score_transformation(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    DataFrame에 Z-score 변환 적용
    
    Args:
        df: 입력 DataFrame
        config: Z-score 변환 설정
        
    Returns:
        변환된 DataFrame과 통계 정보
    """
    stats = {}
    
    if not config.get("enabled", False):
        return df, stats
    
    columns_config = config.get("columns", [])
    
    for col_config in columns_config:
        column_name = col_config.get("column")
        if not column_name or column_name not in df.columns:
            logger.warning(f"Column {column_name} not found in DataFrame")
            continue
        
        # 절대값 적용
        abs_column = f"{column_name}_abs"
        df[abs_column] = df[column_name].abs()
        
        group_by = col_config.get("group_by")
        outlier_config = col_config.get("outlier_removal", {})
        z_range_config = col_config.get("z_score_range", {})
        transform_config = col_config.get("linear_transform", {})
        output_suffix = col_config.get("output_suffix", "_zscore")
        keep_original = col_config.get("keep_original", True)
        
        # Z-score 클리핑 범위 먼저 가져오기
        clip_min = z_range_config.get("clip_min", -np.inf)
        clip_max = z_range_config.get("clip_max", np.inf)
        
        # 그룹별 또는 전체 처리
        if group_by and group_by in df.columns:
            # 그룹별 처리
            z_scores_list = []
            group_stats = []
            
            for group_name, group in df.groupby(group_by):
                abs_values = group[abs_column].dropna()
                
                if len(abs_values) > 0:
                    # 이상치 제거
                    cleaned_values = remove_outliers_iqr(
                        abs_values,
                        iqr_coefficient=outlier_config.get("iqr_coefficient", 1.5),
                        remove_lower=outlier_config.get("remove_lower", True),
                        remove_upper=outlier_config.get("remove_upper", True),
                        lower_quantile=outlier_config.get("lower_quantile", 0.25),
                        upper_quantile=outlier_config.get("upper_quantile", 0.75),
                        iqr_coefficient_lower=outlier_config.get("iqr_coefficient_lower"),
                        iqr_coefficient_upper=outlier_config.get("iqr_coefficient_upper")
                    )
                    
                    if len(cleaned_values) > 0:
                        # 로그 변환을 먼저 적용
                        log_base = transform_config.get("log_base", "e")
                        log_epsilon = transform_config.get("log_epsilon", 0.001)
                        
                        # 로그 변환 (cleaned values)
                        cleaned_log = None
                        if log_base == "10":
                            cleaned_log = np.log10(cleaned_values + log_epsilon)
                        elif log_base == "2":
                            cleaned_log = np.log2(cleaned_values + log_epsilon)
                        else:  # 자연로그
                            cleaned_log = np.log(cleaned_values + log_epsilon)
                        
                        # 로그 변환된 값의 평균과 표준편차
                        mean_log = cleaned_log.mean()
                        std_log = cleaned_log.std()
                        
                        if std_log > 0:
                            # 원본 값들도 로그 변환
                            abs_log = None
                            if log_base == "10":
                                abs_log = np.log10(abs_values + log_epsilon)
                            elif log_base == "2":
                                abs_log = np.log2(abs_values + log_epsilon)
                            else:
                                abs_log = np.log(abs_values + log_epsilon)
                            
                            # 로그 변환된 값에 대해 z-score 계산
                            z_scores = (abs_log - mean_log) / std_log
                            z_scores.index = group.index
                            z_scores_list.append(z_scores)
                        
                        group_stats.append({
                            group_by: group_name,
                            "original_count": len(abs_values),
                            "cleaned_count": len(cleaned_values),
                            "mean": mean_log if 'mean_log' in locals() else np.nan,
                            "std": std_log if 'std_log' in locals() else np.nan,
                            "outliers_removed": len(abs_values) - len(cleaned_values),
                            "log_base": log_base,
                            "log_epsilon": log_epsilon
                        })
            
            # 모든 z-scores 합치기
            if z_scores_list:
                all_z_scores = pd.concat(z_scores_list)
                all_z_scores = all_z_scores.reindex(df.index)
            else:
                all_z_scores = pd.Series(np.nan, index=df.index)
            
            # 간소화된 구조: co_vat_id를 키로 사용하여 빠른 조회 가능
            group_params = {}
            for stat in group_stats:
                co_vat_id = stat[group_by]
                group_params[str(co_vat_id)] = {
                    "mean": stat["mean"],
                    "std": stat["std"],
                    "log_base": stat["log_base"],
                    "log_epsilon": stat["log_epsilon"]
                }
            
            # 변환 설정 저장
            transform_params = {
                "clip_min": clip_min,
                "clip_max": clip_max,
                "target_min": transform_config.get("target_min", 1),
                "target_max": transform_config.get("target_max", 100),
                "round": transform_config.get("round", True)
            }
            
            stats[column_name] = {
                "group_params": group_params,
                "transform_config": transform_params
            }
        else:
            # 전체 데이터 처리
            abs_values = df[abs_column].dropna()
            
            cleaned_values = remove_outliers_iqr(
                abs_values,
                iqr_coefficient=outlier_config.get("iqr_coefficient", 1.5),
                remove_lower=outlier_config.get("remove_lower", True),
                remove_upper=outlier_config.get("remove_upper", True),
                lower_quantile=outlier_config.get("lower_quantile", 0.25),
                upper_quantile=outlier_config.get("upper_quantile", 0.75),
                iqr_coefficient_lower=outlier_config.get("iqr_coefficient_lower"),
                iqr_coefficient_upper=outlier_config.get("iqr_coefficient_upper")
            )
            
            if len(cleaned_values) > 0:
                # 로그 변환을 먼저 적용
                log_base = transform_config.get("log_base", "e")
                log_epsilon = transform_config.get("log_epsilon", 0.001)
                
                # 로그 변환 (cleaned values)
                if log_base == "10":
                    cleaned_log = np.log10(cleaned_values + log_epsilon)
                elif log_base == "2":
                    cleaned_log = np.log2(cleaned_values + log_epsilon)
                else:
                    cleaned_log = np.log(cleaned_values + log_epsilon)
                
                mean_log = cleaned_log.mean()
                std_log = cleaned_log.std()
                
                if std_log > 0:
                    # 전체 데이터도 로그 변환
                    if log_base == "10":
                        df_log = np.log10(df[abs_column] + log_epsilon)
                    elif log_base == "2":
                        df_log = np.log2(df[abs_column] + log_epsilon)
                    else:
                        df_log = np.log(df[abs_column] + log_epsilon)
                    
                    all_z_scores = (df_log - mean_log) / std_log
                else:
                    all_z_scores = pd.Series(np.nan, index=df.index)
            else:
                all_z_scores = pd.Series(np.nan, index=df.index)
            
            # 간소화된 구조: 전체 데이터의 경우
            transform_params = {
                "clip_min": clip_min,
                "clip_max": clip_max,
                "target_min": transform_config.get("target_min", 1),
                "target_max": transform_config.get("target_max", 100),
                "round": transform_config.get("round", True)
            }
            
            stats[column_name] = {
                "global_params": {
                    "mean": mean_log if 'mean_log' in locals() else np.nan,
                    "std": std_log if 'std_log' in locals() else np.nan,
                    "log_base": log_base if 'log_base' in locals() else "e",
                    "log_epsilon": log_epsilon if 'log_epsilon' in locals() else 0.001
                },
                "transform_config": transform_params
            }
        
        # Z-score 클리핑
        z_scores_clipped = all_z_scores.clip(lower=clip_min, upper=clip_max)
        
        # 선형 변환만 적용 (로그는 이미 z-score 계산 전에 적용됨)
        if transform_config.get("enabled", False):
            target_min = transform_config.get("target_min", 1)
            target_max = transform_config.get("target_max", 100)
            
            # 선형 변환
            range_original = clip_max - clip_min
            range_target = target_max - target_min
            
            transformed_values = target_min + (z_scores_clipped - clip_min) * range_target / range_original
            
            if transform_config.get("round", True):
                # NaN 값을 처리하면서 정수로 변환
                mask_finite = np.isfinite(transformed_values)
                transformed_values[mask_finite] = np.round(transformed_values[mask_finite]).astype(int)
                transformed_values[mask_finite] = transformed_values[mask_finite].clip(target_min, target_max)
            
            # 새 컬럼에 저장
            new_column_name = f"{column_name}{output_suffix}"
            df[new_column_name] = transformed_values
            
            # 추론에 필요없는 통계는 제거
        else:
            # 변환 없이 z-score만 저장
            new_column_name = f"{column_name}{output_suffix}"
            df[new_column_name] = z_scores_clipped
        
        # 임시 절대값 컬럼 제거
        df.drop(columns=[abs_column], inplace=True)
        
        # 원본 컬럼 제거 옵션
        if not keep_original and column_name in df.columns:
            df.drop(columns=[column_name], inplace=True)
        
        logger.info(f"✅ Z-score 변환 완료: {column_name} → {new_column_name}")
    
    return df, stats
"""
PEFT + DeepSpeed 프로젝트를 위한 유틸리티 클래스 및 함수들
"""

import os
import json
import torch
import pickle
import logging
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import deepspeed
import datasets
from datasets import Dataset as HuggingFaceDataset
import time
from omegaconf import OmegaConf, DictConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import mlflow


# ==================== 메트릭 관리 클래스 ====================

class MetricTracker:
    """모든 메트릭 계산 및 로깅 통합"""
    def __init__(self, local_rank=0):
        self.local_rank = local_rank
        self.metrics_history = []
    
    def calculate_and_log(self, predictions, labels, prefix="", step=None):
        """한 번에 모든 메트릭 계산 및 로깅"""
        # numpy 배열로 변환
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # 메트릭 계산
        metrics = {
            f"{prefix}accuracy": accuracy_score(labels, predictions),
            f"{prefix}f1_weighted": f1_score(labels, predictions, average='weighted', zero_division=0),
            f"{prefix}f1_macro": f1_score(labels, predictions, average='macro', zero_division=0),
            f"{prefix}precision": precision_score(labels, predictions, average='weighted', zero_division=0),
            f"{prefix}recall": recall_score(labels, predictions, average='weighted', zero_division=0)
        }
        
        # MLflow 로깅 (rank 0에서만)
        if self.local_rank == 0 and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"⚠️ MLflow 로깅 실패: {e}")
        
        # 기록 저장
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_summary(self):
        """전체 메트릭 히스토리 요약"""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history]
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_max"] = np.max(values)
        
        return summary


# ==================== GPU 모니터링 클래스 ====================

class GPUMonitor:
    """GPU 메모리 모니터링 통합"""
    def __init__(self, local_rank=0):
        self.local_rank = local_rank
        self.peak_memory = 0
    
    def get_memory_stats(self):
        """현재 GPU 메모리 상태 반환"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            # 피크 메모리 업데이트
            self.peak_memory = max(self.peak_memory, allocated)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'peak_memory_gb': self.peak_memory
            }
        return {}
    
    def log_memory(self, prefix=""):
        """메모리 상태 로그"""
        stats = self.get_memory_stats()
        if self.local_rank == 0 and stats:
            print(f"[GPU Memory {prefix}] "
                  f"Allocated: {stats['allocated_gb']:.2f}GB, "
                  f"Reserved: {stats['reserved_gb']:.2f}GB, "
                  f"Peak: {stats['peak_memory_gb']:.2f}GB")
        return stats
    
    def reset_peak_memory(self):
        """피크 메모리 통계 리셋"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0


# ==================== 로깅 클래스 ====================

class RankAwareLogger:
    """Rank-aware 로깅을 위한 래퍼"""
    def __init__(self, rank=0):
        self.rank = rank
    
    def info(self, message):
        if self.rank == 0:
            print(f"[INFO] {message}")
    
    def warning(self, message):
        if self.rank == 0:
            print(f"[WARNING] {message}")
    
    def error(self, message):
        if self.rank == 0:
            print(f"[ERROR] {message}")
    
    def debug(self, message):
        if self.rank == 0:
            print(f"[DEBUG] {message}")


# ==================== 경로 관리 클래스 ====================

class PathManager:
    """모든 경로를 중앙에서 관리"""
    
    def __init__(self, config, local_rank=0):
        self.config = config
        self.local_rank = local_rank
        self.base_dir = Path.cwd()
        
        # 모든 경로를 초기화 시점에 정의
        self.paths = {
            'data': self._resolve(config.data.dataset_all_path),
            'cache': self._resolve(config.cache_dir),
            'arrow_cache': self._resolve(config.cache_dir) / "hf_arrow",
            'model_output': self._resolve(config.model.save_path),
            'checkpoints': self._resolve(config.model.save_path) / "checkpoints",
            'best_model': self._resolve(config.model.save_path) / "best_model",
            'tensorboard': self._resolve("./tensorboard_logs"),
            'csv_logs': self._resolve("./csv_logs"),
            'mlflow': self._resolve("./mlruns")
        }
        
        # rank 0에서만 디렉토리 생성
        if self.local_rank == 0:
            self._create_all_directories()
            self._log_all_paths()
    
    def _resolve(self, path):
        """경로를 절대 경로로 변환"""
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p
    
    def _create_all_directories(self):
        """필요한 모든 디렉토리 생성"""
        for name, path in self.paths.items():
            if name != 'data':  # 데이터 파일은 디렉토리가 아님
                path.mkdir(parents=True, exist_ok=True)
    
    def _log_all_paths(self):
        """모든 경로 로깅"""
        print("=" * 50)
        print("📁 경로 설정:")
        for name, path in self.paths.items():
            print(f"  {name}: {path}")
        print("=" * 50)
    
    def get(self, key):
        """경로 가져오기"""
        if key not in self.paths:
            raise KeyError(f"Unknown path key: {key}")
        return self.paths[key]
    
    def validate_data_path(self):
        """데이터 파일 경로 검증"""
        data_path = self.get('data')
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if self.local_rank == 0:
            print(f"✅ Data file found: {data_path}")
        return True


# ==================== 유틸리티 함수들 ====================

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 외부 라이브러리 로깅 레벨 조정
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("deepspeed").setLevel(logging.INFO)
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(data: Dict[str, Any], filepath: str):
    """JSON 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_total_parameters(model) -> Dict[str, int]:
    """모델의 총 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
    }


def get_learning_rate(optimizer):
    """현재 학습률 가져오기"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def create_optimizer_grouped_parameters(model, weight_decay: float = 0.01):
    """옵티마이저를 위한 파라미터 그룹 생성"""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def print_rank_0(message: str, rank: int = 0):
    """Rank 0에서만 메시지 출력"""
    if rank == 0:
        print(message)


def get_gpu_memory_info():
    """GPU 메모리 정보를 문자열로 반환"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "GPU not available"
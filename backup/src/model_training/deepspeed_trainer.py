#!/usr/bin/env python3
"""
PEFT + DeepSpeed 훈련 스크립트 (DeepSpeed 런처용 최소화)
실행 예시: deepspeed --num_gpus=4 train.py --config conf/experiment.yaml
"""

import os, torch, deepspeed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf
import mlflow
import argparse
from datetime import datetime
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import time
import random
import torch.distributed as dist
from pathlib import Path

# HF Datasets
from datasets import DatasetDict, Dataset, load_from_disk
import pickle
import json

# 유틸리티 클래스 임포트
from utils import MetricTracker, GPUMonitor, RankAwareLogger, PathManager

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)



# ==================== 기존 코드 계속 ====================

# 통합된 토크나이저 설정 함수
def setup_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    토크나이저 생성 및 Qwen3 최적화 설정 통합 함수
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # 패딩 방향 설정 (훈련용)
    tokenizer.padding_side = "right"
    
    # 토큰 ID 확인 및 검증
    if local_rank == 0:
        print(f"🔍 토크나이저 토큰 정보:")
        print(f"   - PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"   - EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        
        # Qwen3는 기본적으로 PAD와 EOS가 구분되어 있음
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print("⚠️ 경고: PAD와 EOS 토큰이 동일합니다. 성능 저하 가능성 있음.")
        else:
            print("✅ PAD와 EOS 토큰이 올바르게 구분되어 있습니다.")
    
    return tokenizer

# 환경 자동 설정 함수 (최소화)
def auto_setup_environment():
    """DeepSpeed 런처 환경 자동 설정"""
    # Intel oneMKL 오류 방지 환경변수
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"  #oneMKL 서비스 레이어를 강제로 Intel 구현으로 고정. AMD/ARM 환경에서 불안정한 ‘ILP64 vs LP64’ 충돌을 예방.
    os.environ["MKL_THREADING_LAYER"] = "GNU"  #oneMKL 스레딩 레이어를 GNU로 설정. 멀티스레딩 환경에서 안정성을 위해 설정.
    os.environ["OMP_NUM_THREADS"] = "1"  #OpenMP 스레딩 레이어를 1로 설정. 멀티스레딩 환경에서 안정성을 위해 설정.
    os.environ["MKL_NUM_THREADS"] = "1"  #oneMKL 스레딩 레이어를 1로 설정. 멀티스레딩 환경에서 안정성을 위해 설정.
    
    # CUDA 관련 설정
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #CUDA 런타임 블로킹 모드 활성화. 디버깅 및 메모리 누수 추적에 도움.
    torch.backends.cuda.matmul.allow_tf32 = True  #CUDA 매트릭스 곱셈에서 TF32 활성화. 성능 향상을 위해 설정.
    torch.backends.cudnn.allow_tf32 = True  #CUDNN에서 TF32 활성화. 성능 향상을 위해 설정.
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  #Flash Attention 2 활성화. 성능 향상을 위해 설정.
    if local_rank == 0:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f"🚀 DeepSpeed 런처 환경:")
        print(f"   World Size: {world_size}")
        print(f"   Local Rank: {local_rank}")
        print(f"   🔧 Intel oneMKL 안정화 환경변수 설정됨")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.device_count()}개")

# 데이터 로딩 함수 (내장 함수 활용)
def pkl_to_arrow_once(pkl_path, arrow_dir, shard_size="500MB"):
    """PKL → Arrow(자동 샤딩) 한 번만 변환 with checksum-based caching"""
    import hashlib
    
    # 체크섬 계산 함수
    def calculate_file_checksum(filepath, chunk_size=8192):
        """파일의 MD5 체크섬 계산"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    # 메타데이터 경로
    meta_path = os.path.join(arrow_dir, "meta.json")
    
    # 전체 데이터셋 메타데이터 경로 (pkl 파일의 metadata 키 전체 저장)
    full_metadata_path = os.path.join(arrow_dir, "dataset_full_metadata.json")
    
    # 기존 캐시 확인
    if os.path.exists(arrow_dir) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            # 체크섬 및 경로 검증
            if "pkl_checksum" in meta and "pkl_path" in meta:
                current_checksum = calculate_file_checksum(pkl_path)
                if (meta["pkl_checksum"] == current_checksum and 
                    meta["pkl_path"] == pkl_path):
                    if local_rank == 0:
                        print(f"✅ Arrow 캐시 유효 (체크섬 일치): {arrow_dir}")
                    return  # 캐시 유효, 변환 스킵
                else:
                    if local_rank == 0:
                        print(f"🔄 Arrow 캐시 무효화 (체크섬/경로 변경)")
                        print(f"   이전 경로: {meta.get('pkl_path', 'N/A')}")
                        print(f"   현재 경로: {pkl_path}")
        except Exception as e:
            if local_rank == 0:
                print(f"⚠️ 메타데이터 읽기 실패: {e}")
    
    # 캐시 재생성
    os.makedirs(arrow_dir, exist_ok=True)
    
    if local_rank == 0:
        print(f"📦 PKL → Arrow 변환 시작: {pkl_path}")
        print(f"   대상: {arrow_dir}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old structure (nested under data['data']) and new structure (flat)
    # Check if this is the old structure with nested 'data'
    if 'data' in data and isinstance(data['data'], dict) and 'train' in data['data']:
        # Old structure: data is nested under data['data']
        actual_data = data['data']
    else:
        # New structure: train/val/test are at top level
        actual_data = data
    
    # Create datasets with proper split names
    dsets = DatasetDict({
        "train": Dataset.from_dict(actual_data["train"]),
        "validation": Dataset.from_dict(actual_data.get("validation", actual_data.get("val", {}))),
        "test": Dataset.from_dict(actual_data["test"])
    })
    dsets.save_to_disk(arrow_dir, max_shard_size=shard_size)  # 자동 분할
    
    # Extract num_classes from the correct location
    num_classes = None
    if 'metadata' in data:
        # Try direct access first
        if 'num_classes' in data['metadata']:
            num_classes = data['metadata']['num_classes']
        # If not found, check inside label_mapping
        elif 'label_mapping' in data['metadata'] and 'num_classes' in data['metadata']['label_mapping']:
            num_classes = data['metadata']['label_mapping']['num_classes']
    
    # If still not found, count unique labels
    if num_classes is None:
        all_labels = set()
        for split in ['train', 'val', 'validation', 'test']:
            if split in actual_data and 'labels' in actual_data[split]:
                all_labels.update(actual_data[split]['labels'])
        num_classes = len(all_labels)
        if local_rank == 0:
            print(f"⚠️ num_classes not found in metadata, calculated from labels: {num_classes}")
    
    # 메타데이터 저장 (체크섬 및 경로 포함)
    pkl_checksum = calculate_file_checksum(pkl_path)
    with open(meta_path, "w") as f:
        json.dump({
            "num_classes": num_classes,
            "created_at": datetime.now().isoformat(),
            "pkl_path": pkl_path,
            "pkl_checksum": pkl_checksum,
            "pkl_size": os.path.getsize(pkl_path)
        }, f)
    
    # 전체 데이터셋 메타데이터 저장 (pkl의 metadata 키 전체)
    # Ensure num_classes is at the top level of metadata
    full_metadata = data.get("metadata", {}).copy()
    if 'num_classes' not in full_metadata:
        full_metadata['num_classes'] = num_classes
    
    with open(full_metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False, default=str)
    
    if local_rank == 0:
        print(f"✅ Arrow 변환 완료 (체크섬: {pkl_checksum[:8]}...)")
        print(f"✅ 전체 메타데이터 저장: {full_metadata_path}")

def build_loaders(cfg):
    """HF Datasets API로 DataLoader 빌드 - PKL 파일 구조 그대로 사용"""
    arrow_dir = os.path.join(cfg.cache_dir, "hf_arrow")
    
    # rank 0에서만 PKL → Arrow 변환 수행
    if local_rank == 0:
        pkl_to_arrow_once(cfg.data.dataset_all_path, arrow_dir)   # ①
    
    # 다른 rank는 변환 완료까지 대기 (간단한 동기화)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dsdict = load_from_disk(arrow_dir, keep_in_memory=False)  # ② mmap 로드 with memory optimization
    train_ds = dsdict["train"]
    val_ds = dsdict["validation"]
    test_ds = dsdict["test"]

    # 🚫 강제 분할 로직 제거 - PKL 파일의 데이터 구조 그대로 사용
    if local_rank == 0:
        print(f"📊 PKL 파일 구조 그대로 사용:")
        print(f"   Train: {len(train_ds):,}개 샘플")
        print(f"   Validation: {len(val_ds):,}개 샘플")
        print(f"   Test: {len(test_ds):,}개 샘플")

    # ③ (선택) 서브셋 추출
    if cfg.data.extract_ratio < 1.0:
        n_train = int(len(train_ds) * cfg.data.extract_ratio)
        n_val = int(len(val_ds) * cfg.data.extract_ratio)
        n_test = int(len(test_ds) * cfg.data.extract_ratio)
        
        train_ds = train_ds.shuffle(seed=42).select(range(n_train))
        val_ds = val_ds.shuffle(seed=42).select(range(n_val))
        test_ds = test_ds.shuffle(seed=42).select(range(n_test))
        
        if local_rank == 0:
            print(f"📏 데이터 추출 비율 {cfg.data.extract_ratio}:")
            print(f"   Train: {len(train_ds):,}개")
            print(f"   Validation: {len(val_ds):,}개")
            print(f"   Test: {len(test_ds):,}개")

    # ④ 토크나이즈 batched map - 통합된 토크나이저 설정 사용
    tok = setup_tokenizer(cfg.model.name, trust_remote_code=True)
    
    def tok_fn(batch):
        return tok(
            batch["text"], 
            truncation=True, 
            max_length=cfg.data.max_length,
            add_special_tokens=True,  # 명시적으로 EOS 토큰 추가
            padding=False  # DataCollator가 배치 생성 시 패딩 처리
        )

    # 토크나이징 후 필요한 컬럼만 남기기 (input_ids, attention_mask, labels)
    cols_to_remove = ["text", "original_labels"]  # 문제가 되는 컬럼들 제거
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove, num_proc=os.cpu_count())
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove, num_proc=os.cpu_count())
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove, num_proc=os.cpu_count())

    # ⑤ 텐서 포맷 지정 → 커스텀 Dataset 필요 없음
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    collate = DataCollatorWithPadding(tok)

    # DistributedSampler 적용
    from torch.utils.data import DataLoader, DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.deepspeed.train_micro_batch_size_per_gpu,
                              sampler=train_sampler, collate_fn=collate,
                              num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory,
                              prefetch_factor=2, persistent_workers=True if cfg.hardware.num_workers > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=cfg.deepspeed.train_micro_batch_size_per_gpu,
                            sampler=val_sampler, collate_fn=collate,
                            num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory,
                            prefetch_factor=2, persistent_workers=True if cfg.hardware.num_workers > 0 else False)
    test_loader = DataLoader(test_ds, batch_size=cfg.deepspeed.train_micro_batch_size_per_gpu,
                             sampler=test_sampler, collate_fn=collate,
                             num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory,
                             prefetch_factor=2, persistent_workers=True if cfg.hardware.num_workers > 0 else False)

    # 최종 데이터셋 크기 확인 및 출력
    if local_rank == 0:
        print(f"📊 최종 데이터셋 크기:")
        print(f"   Train: {len(train_ds):,}개 ({len(train_loader):,} 배치)")
        print(f"   Validation: {len(val_ds):,}개 ({len(val_loader):,} 배치)")
        print(f"   Test: {len(test_ds):,}개 ({len(test_loader):,} 배치)")

    # num_classes 반환 (메타데이터에서 읽기)
    with open(os.path.join(arrow_dir, "meta.json")) as f:
        meta = json.load(f)
        num_classes = meta["num_classes"]
        if local_rank == 0 and "pkl_checksum" in meta:
            print(f"📋 데이터셋 메타정보:")
            print(f"   원본 PKL: {meta.get('pkl_path', 'N/A')}")
            print(f"   체크섬: {meta.get('pkl_checksum', 'N/A')[:8]}...")
            print(f"   클래스 수: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes, tok

# 향상된 평가 함수 (MetricTracker 활용)
def evaluate_model_comprehensive(model_engine, eval_dataloader, local_rank=0, metric_tracker=None, prefix=""):
    """종합적인 모델 평가 (loss, accuracy, f1, precision, recall)"""
    model_engine.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # MetricTracker가 없으면 임시 생성
    if metric_tracker is None:
        metric_tracker = MetricTracker(local_rank)
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=local_rank != 0):
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - eval_start_time
    
    # MetricTracker를 사용해 메트릭 계산 및 로깅
    metrics = metric_tracker.calculate_and_log(
        predictions=np.array(all_predictions),
        labels=np.array(all_labels),
        prefix=prefix
    )
    
    # 추가 메트릭
    metrics.update({
        f'{prefix}loss': total_loss / len(eval_dataloader),
        f'{prefix}eval_samples': len(all_predictions),
        f'{prefix}eval_time_sec': eval_time,
        f'{prefix}eval_samples_per_sec': len(all_predictions) / eval_time if eval_time > 0 else 0
    })
    
    return metrics

# 기존 호환성을 위한 래퍼 함수
def evaluate_model(model_engine, eval_dataloader):
    """기존 호환성을 위한 평가 함수"""
    return evaluate_model_comprehensive(model_engine, eval_dataloader, local_rank)

# 메타데이터 생성 함수 추가
def save_model_metadata(save_dir: str, model, tokenizer, config, num_classes: int):
    """모델 메타데이터 저장"""
    metadata = {
        "model_info": {
            "base_model_name": config.model.name,
            "num_classes": num_classes,
            "torch_dtype": config.model.torch_dtype,
            "model_type": "sequence_classification",
            "peft_type": "LoRA",
            "created_at": datetime.now().isoformat()
        },
        "lora_config": OmegaConf.to_container(config.lora),
        "data_info": {
            "max_length": config.data.max_length,
            "padding": config.data.padding,
            "truncation": config.data.truncation,
            "extract_ratio": config.data.extract_ratio,
            "dataset_path": config.data.dataset_all_path
        },
        "tokenizer_info": {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        },
        "training_config": {
            "epochs": config.training.epochs,
            "batch_size": config.deepspeed.train_micro_batch_size_per_gpu,
            "learning_rate": config.deepspeed.optimizer.params.lr,
            "weight_decay": config.deepspeed.optimizer.params.weight_decay
        }
    }
    
    metadata_path = os.path.join(save_dir, "model_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_path

# save_company_scaling_stats 함수 삭제됨 - 더 이상 필요하지 않음

def save_dataset_metadata_once(arrow_dir: str, checkpoint_root_dir: str, local_rank: int = 0):
    """전체 데이터셋 메타데이터를 checkpoints 폴더에 한 번만 저장"""
    if local_rank != 0:
        return None
    
    try:
        # Arrow 디렉토리에서 전체 메타데이터 파일 읽기
        full_metadata_path = os.path.join(arrow_dir, "dataset_full_metadata.json")
        
        if not os.path.exists(full_metadata_path):
            print(f"⚠️ 전체 메타데이터 파일을 찾을 수 없습니다: {full_metadata_path}")
            return None
        
        # 메타데이터 읽기
        with open(full_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 저장 시간 추가
        metadata['saved_at'] = datetime.now().isoformat()
        
        # checkpoints 폴더 바로 아래에 저장
        dataset_metadata_path = os.path.join(checkpoint_root_dir, "dataset_metadata.json")
        
        # 이미 존재하는 경우 덮어쓰기
        with open(dataset_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ 데이터셋 메타데이터 저장: checkpoints/dataset_metadata.json")
        
        # 메타데이터 요약 출력
        print(f"   📋 메타데이터 요약:")
        print(f"      - num_classes: {metadata.get('num_classes', 'N/A')}")
        print(f"      - total_samples: {metadata.get('total_samples', 'N/A')}")
        print(f"      - target_column: {metadata.get('target_column', 'N/A')}")
        print(f"      - feature_columns 수: {len(metadata.get('feature_columns', []))}")
        
        # company_scaling_stats는 더 이상 사용하지 않음
        
        return dataset_metadata_path
        
    except Exception as e:
        print(f"   ⚠️ 데이터셋 메타데이터 저장 실패: {e}")
        return None

# ==================== 체크포인트 관리 함수 ====================

def find_latest_checkpoint(checkpoint_dir: Path, local_rank: int = 0) -> tuple[str, str, int]:
    """
    DeepSpeed latest 파일을 활용하여 최신 체크포인트 찾기
    
    Returns:
        (checkpoint_dir, tag, epoch_number)
    """
    if not checkpoint_dir.exists():
        if local_rank == 0:
            logger.info(f"체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
        return None, None, 0
    
    # DeepSpeed latest 파일 확인 (DeepSpeed가 자동 생성)
    latest_file = checkpoint_dir / "latest"
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            latest_tag = f.read().strip()
            if local_rank == 0:
                logger.info(f"✅ DeepSpeed latest 파일에서 태그 발견: {latest_tag}")
            
            # epoch 번호 추출
            epoch_num = 0
            if "epoch_" in latest_tag:
                try:
                    epoch_num = int(latest_tag.split("epoch_")[1].split("_")[0])
                except:
                    epoch_num = int(latest_tag.split("_")[-1])
            
            return str(checkpoint_dir), latest_tag, epoch_num
    
    # latest 파일이 없으면 epoch_XX 패턴으로 검색
    epoch_dirs = sorted(checkpoint_dir.glob("epoch_*"), 
                       key=lambda x: int(x.name.split("_")[1]))
    
    if epoch_dirs:
        latest_dir = epoch_dirs[-1]
        tag = latest_dir.name
        epoch_num = int(tag.split("_")[1])
        
        if local_rank == 0:
            logger.info(f"✅ 최신 체크포인트 디렉토리 발견: {tag} (epoch {epoch_num})")
        
        return str(checkpoint_dir), tag, epoch_num
    
    if local_rank == 0:
        logger.info("체크포인트를 찾을 수 없습니다")
    return None, None, 0


def load_checkpoint_deepspeed(model_engine, scheduler, checkpoint_dir: str, tag: str = None, local_rank: int = 0):
    """
    DeepSpeed 체크포인트 로드 (내장 메소드 최대 활용)
    
    Args:
        model_engine: DeepSpeed model engine
        scheduler: Learning rate scheduler
        checkpoint_dir: 체크포인트 디렉토리 경로
        tag: 체크포인트 태그 (None이면 latest 사용)
        local_rank: 현재 프로세스의 rank
    
    Returns:
        (client_state, start_epoch, resume_run_id)
    """
    try:
        # DeepSpeed load_checkpoint 호출
        # tag=None이면 DeepSpeed가 자동으로 latest 파일을 읽어서 처리
        load_path, client_state = model_engine.load_checkpoint(
            checkpoint_dir,
            tag=tag,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,  # DeepSpeed가 lr_scheduler도 로드
            load_module_strict=True
        )
        
        if local_rank == 0:
            actual_tag = tag if tag else "latest"
            print(f"✅ DeepSpeed 체크포인트 로드 성공 (tag: {actual_tag})")
            print(f"   로드 경로: {load_path}")
            
            # DeepSpeed 0.14.4에서는 client_state가 딕셔너리로 반환됨
            if client_state:
                print(f"   Client state 타입: {type(client_state)}")
                if isinstance(client_state, dict):
                    print(f"   Client state keys: {list(client_state.keys())}")
        
        # client_state에서 정보 추출
        start_epoch = 0
        resume_run_id = None
        
        if client_state and isinstance(client_state, dict):
            # 에폭 정보
            if 'epoch' in client_state:
                start_epoch = client_state['epoch']
                print(f"   📍 에폭 {start_epoch} 완료됨, 에폭 {start_epoch + 1}부터 재개") if local_rank == 0 else None
            
            # MLflow run_id
            if 'mlflow_run_id' in client_state:
                resume_run_id = client_state['mlflow_run_id']
                print(f"   🏃 MLflow run ID: {resume_run_id}") if local_rank == 0 else None
            
            
            # 추가 메타데이터 출력
            if local_rank == 0:
                for key in ['train_loss', 'val_accuracy', 'val_loss', 'timestamp', 'num_classes']:
                    if key in client_state:
                        print(f"   📊 {key}: {client_state[key]}")
        else:
            print("⚠️ client_state가 비어있거나 예상된 형식이 아닙니다.") if local_rank == 0 else None
        
        return client_state, start_epoch, resume_run_id
        
    except Exception as e:
        if local_rank == 0:
            print(f"❌ 체크포인트 로드 실패: {e}")
            print(f"   오류 타입: {type(e).__name__}")
            print(f"   체크포인트 경로: {checkpoint_dir}")
            print(f"   태그: {tag}")
            if "size mismatch" in str(e):
                print(f"⚠️ 클래스 수가 다른 경우 체크포인트를 사용할 수 없습니다.")
                print(f"   체크포인트를 다시 학습하거나 동일한 데이터셋을 사용하세요.")
        return None, 0, None


# 메인 함수
def main():
    # 인수 파싱 (MLflow 프로젝트 호환성)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/experiment.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default="", help="DeepSpeed 체크포인트 경로 (자동 학습 재개)")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="체크포인트 경로 (DeepSpeed 런처 호환)")
    parser.add_argument("--checkpoint_tag", type=str, default="", help="체크포인트 태그 (epoch_N 또는 latest)")
    parser.add_argument("--mlflow_run_id", type=str, default="", help="MLflow run ID 재개 (호환성용)")
    parser.add_argument("--run_id", type=str, default="", help="MLflow run ID 재개 (DeepSpeed 인자용)")
    
    # MLflow 프로젝트 파라미터들 (YAML 설정 오버라이드용)
    parser.add_argument("--experiment_name", type=str, default="", help="실험 이름 (MLflow용)")
    parser.add_argument("--run_name", type=str, default="", help="실행 이름 (MLflow용)")
    parser.add_argument("--epochs", type=int, default=0, help="에폭 수 (0이면 YAML 설정 사용)")
    parser.add_argument("--batch_size", type=int, default=0, help="배치 크기 (0이면 YAML 설정 사용)")
    parser.add_argument("--learning_rate", type=float, default=0.0, help="학습률 (0이면 YAML 설정 사용)")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="데이터 사용 비율")
    parser.add_argument("--quick_test", action="store_true", help="빠른 테스트 모드")
    parser.add_argument("--mode", type=str, default="train", help="실행 모드 (train/evaluate/etc)")
    parser.add_argument("--output_dir", type=str, default="", help="출력 디렉토리")
    parser.add_argument("--detailed_metrics", action="store_true", help="상세 메트릭 출력")
    
    args = parser.parse_args()
    
    if local_rank == 0:
        print(f"🎯 MLflow 파라미터 수신:")
        print(f"   실험 이름: {args.experiment_name}")
        print(f"   실행 이름: {args.run_name}")
        print(f"   에폭: {args.epochs} (0이면 YAML 설정 사용)")
        print(f"   데이터 비율: {args.data_fraction}")
        print(f"   빠른 테스트: {args.quick_test}")
        print(f"   모드: {args.mode}")
        print(f"   🔑 run_id: {args.run_id or args.mlflow_run_id or '(없음)'}")
        print("")
    
    # DeepSpeed 초기화
    deepspeed.init_distributed()
    
    # 설정 로드
    config = OmegaConf.load(args.config)
    
    # 디버깅: 설정 확인
    if local_rank == 0:
        print(f"🔍 YAML 설정 로드 확인:")
        print(f"   - config.data.extract_ratio: {config.data.extract_ratio}")
        print(f"   - config.training.epochs: {config.training.epochs}")
        print(f"   - config.deepspeed.train_micro_batch_size_per_gpu: {config.deepspeed.train_micro_batch_size_per_gpu}")
        print(f"   - config.deepspeed.optimizer.params.lr: {config.deepspeed.optimizer.params.lr}")
        print("")
    
    # 환경 설정
    auto_setup_environment()
    
    # ==================== 통합 클래스 초기화 ====================
    # 경로 관리자 초기화
    path_manager = PathManager(config, local_rank)
    path_manager.validate_data_path()
    
    # 로거 초기화
    logger = RankAwareLogger(local_rank)
    
    # GPU 모니터 초기화
    gpu_monitor = GPUMonitor(local_rank)
    
    # 메트릭 트래커 초기화
    metric_tracker = MetricTracker(local_rank)
    
    logger.info("✅ 통합 관리 클래스 초기화 완료")
    gpu_monitor.log_memory("초기화 후")
    
    # 🚀 단순화된 데이터 로딩 (40줄 이내)
    train_dataloader, eval_dataloader, test_dataloader, num_classes, tokenizer = build_loaders(config)
    
    # Arrow 디렉토리 경로 저장 (에폭별 메타데이터 저장용)
    arrow_dir = os.path.join(config.cache_dir, "hf_arrow")
    
    # 데이터셋 메타데이터 로드 (신규 학습시 사용)
    dataset_metadata = None
    dataset_metadata_path = os.path.join(arrow_dir, "dataset_full_metadata.json")
    if os.path.exists(dataset_metadata_path):
        try:
            with open(dataset_metadata_path, 'r') as f:
                dataset_metadata = json.load(f)
                if local_rank == 0:
                    print(f"📊 데이터셋 메타데이터 로드 성공:")
                    print(f"   - 전체 샘플 수: {dataset_metadata.get('total_samples', 'N/A'):,}")
                    print(f"   - 타겟 컬럼: {dataset_metadata.get('target_column', 'N/A')}")
                    print(f"   - 피처 컬럼 수: {len(dataset_metadata.get('feature_columns', []))}")
                    if 'label_mapping' in dataset_metadata:
                        label_mapping = dataset_metadata['label_mapping']
                        print(f"   - 실제 클래스: {label_mapping.get('num_base_classes', 'N/A')}")
                        print(f"   - 더미 클래스: {label_mapping.get('num_dummy_classes', 'N/A')}")
        except Exception as e:
            if local_rank == 0:
                print(f"⚠️ 데이터셋 메타데이터 로드 실패: {e}")
    
    if local_rank == 0:
        print(f"✅ 데이터 로딩 완료: Train={len(train_dataloader.dataset)}, Val={len(eval_dataloader.dataset)}, Test={len(test_dataloader.dataset)}, Classes={num_classes}")
    
    # 🔄 체크포인트 경로 및 태그 처리 (디버깅 강화)
    checkpoint_path = args.checkpoint_path or args.resume_from_checkpoint
    checkpoint_tag = args.checkpoint_tag
    start_epoch = 0
    # run_id 파라미터 우선 처리 (DeepSpeed 인자 vs 호환성)
    resume_run_id = args.run_id or args.mlflow_run_id
    checkpoint_num_classes = None  # 체크포인트의 클래스 수
    is_resume_training = False  # 학습 재개 여부 플래그
    
    # 🔧 MLflow artifacts 전체 경로에서 태그 자동 추출
    original_checkpoint_path = checkpoint_path  # 원본 경로 보존
    if checkpoint_path and not checkpoint_tag:
        # MLflow artifacts 경로 패턴 감지: .../artifacts/checkpoints/epoch_XX
        if "artifacts/checkpoints/" in checkpoint_path:
            # 마지막 부분이 epoch_XX 형태인지 확인
            path_parts = checkpoint_path.rstrip('/').split('/')
            last_part = path_parts[-1]
            if last_part.startswith('epoch_'):
                checkpoint_tag = last_part
                # 원본 경로는 보존하고, 로딩 로직에서 처리
                if local_rank == 0:
                    print(f"🔍 MLflow artifacts 경로 자동 감지:")
                    print(f"   원본 경로에서 태그 추출: {last_part}")
                    print(f"   원본 경로 보존: {original_checkpoint_path}")
                    print(f"   추출된 태그: {checkpoint_tag}")

    if local_rank == 0:
        print(f"🔍 체크포인트 매개변수 확인:")
        print(f"   checkpoint_path: {checkpoint_path}")
        print(f"   original_checkpoint_path: {original_checkpoint_path}")
        print(f"   checkpoint_tag: {checkpoint_tag}")
        print(f"   초기 start_epoch: {start_epoch}")
        print(f"   🔑 resume_run_id: {resume_run_id}")
    
    # 체크포인트 경로 자동 감지 및 latest 파일 활용
    if checkpoint_path and os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"📥 체크포인트 경로 발견: {checkpoint_path}")
            
            # latest 파일 확인 및 태그 자동 설정 (태그가 명시되지 않은 경우만)
            latest_file = os.path.join(checkpoint_path, "latest")
            if not checkpoint_tag and os.path.exists(latest_file):
                try:
                    with open(latest_file, 'r') as f:
                        latest_tag = f.read().strip()
                        checkpoint_tag = latest_tag
                        print(f"📌 latest 파일에서 태그 자동 감지: {latest_tag}")
                except Exception as e:
                    print(f"⚠️ latest 파일 읽기 실패: {e}")
            
            # 태그가 여전히 없으면 latest 사용
            if not checkpoint_tag:
                checkpoint_tag = "latest"
                
            print(f"🔄 태그 '{checkpoint_tag}'로 자동 학습 재개를 시작합니다...")
            if resume_run_id:
                print(f"🔄 MLflow run_id '{resume_run_id}'로 로그 재개 예정")
            
            # 체크포인트 메타데이터 읽기 (model_metadata.json에서 모델 정보 확인)
            try:
                # model_metadata.json 파일 경로
                model_metadata_path = Path(checkpoint_path) / checkpoint_tag / "model_metadata.json"
                if model_metadata_path.exists():
                    with open(model_metadata_path, 'r') as f:
                        model_metadata = json.load(f)
                        # 모델 정보에서 클래스 수 확인
                        if 'model_info' in model_metadata and 'num_classes' in model_metadata['model_info']:
                            checkpoint_num_classes = model_metadata['model_info']['num_classes']
                            print(f"📊 model_metadata.json에서 클래스 수 발견: {checkpoint_num_classes}")
                            if checkpoint_num_classes != num_classes:
                                print(f"⚠️ 클래스 수 불일치: 데이터셋={num_classes}, 체크포인트={checkpoint_num_classes}")
                                print(f"🔧 체크포인트의 클래스 수({checkpoint_num_classes})로 모델을 초기화합니다.")
                                num_classes = checkpoint_num_classes
                            is_resume_training = True  # 체크포인트가 존재하므로 재개 학습
                            
                        
                        # 기타 모델 설정 정보 확인
                        if 'data_info' in model_metadata:
                            print(f"📊 체크포인트의 데이터 설정:")
                            print(f"   - max_length: {model_metadata['data_info'].get('max_length', 'N/A')}")
                            print(f"   - extract_ratio: {model_metadata['data_info'].get('extract_ratio', 'N/A')}")
                        
                        if 'tokenizer_info' in model_metadata:
                            print(f"📊 체크포인트의 토크나이저 정보:")
                            print(f"   - vocab_size: {model_metadata['tokenizer_info'].get('vocab_size', 'N/A')}")
                else:
                    print(f"⚠️ model_metadata.json 파일을 찾을 수 없습니다: {model_metadata_path}")
                    print(f"   신규 학습으로 간주하여 데이터셋 정보를 사용합니다.")
            except Exception as e:
                print(f"⚠️ model_metadata.json 읽기 실패: {e}")
                print(f"   신규 학습으로 간주하여 데이터셋 정보를 사용합니다.")
    
    # 재학습시 checkpoints 폴더에서 dataset_metadata.json 로드
    if is_resume_training and checkpoint_path:
        dataset_metadata_ckpt_path = os.path.join(checkpoint_path, "dataset_metadata.json")
        if os.path.exists(dataset_metadata_ckpt_path):
            try:
                with open(dataset_metadata_ckpt_path, 'r') as f:
                    dataset_metadata = json.load(f)
                    if local_rank == 0:
                        print(f"📊 체크포인트 폴더에서 데이터셋 메타데이터 로드 성공")
            except Exception as e:
                if local_rank == 0:
                    print(f"⚠️ 체크포인트의 데이터셋 메타데이터 로드 실패: {e}")
    
    # 학습 모드 확인 로그
    if local_rank == 0:
        if is_resume_training:
            print(f"\n🔄 === 학습 재개 모드 ===")
            print(f"   체크포인트 경로: {checkpoint_path}")
            print(f"   체크포인트 태그: {checkpoint_tag}")
            print(f"   클래스 수: {num_classes} (체크포인트 기준)")
            print(f"   메타데이터 소스: checkpoints/dataset_metadata.json")
        else:
            print(f"\n🆕 === 신규 학습 모드 ===")
            print(f"   데이터셋 경로: {config.data.dataset_all_path}")
            print(f"   클래스 수: {num_classes} (데이터셋 기준)")
            print(f"   메타데이터 소스: dataset_all.pkl")
        print(f"")
    
    # 모델 생성
    # Qwen3는 이미 올바른 pad_token_id를 가지고 있음 (151643)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=num_classes,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=config.model.trust_remote_code,
        pad_token_id=tokenizer.pad_token_id  # 명시적으로 설정 (151643)
    )
    
    # 모델이 올바른 토큰 ID를 사용하는지 확인
    if local_rank == 0:
        print(f"📌 모델 토큰 설정:")
        print(f"   - Model pad_token_id: {model.config.pad_token_id}")
        print(f"   - Model eos_token_id: {model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 'N/A'}")
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print("⚠️ 경고: 토크나이저에서 PAD와 EOS가 동일합니다!")
        else:
            print("✅ PAD와 EOS가 올바르게 구분되어 있습니다.")
    
    # PEFT 적용 (YAML 앵커 lora 설정 + 음수 인덱스 변환)
    lora_dict = OmegaConf.to_container(config.lora)
    
    # layers_to_transform의 음수 인덱스를 양수로 변환
    if 'layers_to_transform' in lora_dict:
        total_layers = model.config.num_hidden_layers  # Qwen3-4B: 36
        idxs = []
        for i in lora_dict['layers_to_transform']:
            idxs.append(total_layers + i if i < 0 else i)
        lora_dict['layers_to_transform'] = sorted(set(idxs))
        
        if local_rank == 0:
            print(f"🎯 LoRA layers_to_transform 변환: {config.lora.layers_to_transform} → {lora_dict['layers_to_transform']}")
    
    lora_config = LoraConfig(**lora_dict)
    model = get_peft_model(model, lora_config)
    
    # 모델 설정 동기화
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, 'base_model'):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # DeepSpeed 설정 준비 (warmup 설정 포함)
    ds_config = OmegaConf.to_container(config.deepspeed, resolve=True)
    
    if local_rank == 0:
        print(f"🔍 초기 ds_config scheduler 설정:")
        if 'scheduler' in ds_config:
            print(f"   - scheduler type: {ds_config['scheduler'].get('type', 'N/A')}")
            print(f"   - scheduler params: {ds_config['scheduler'].get('params', {})}")
    
    # 총 훈련 스텝 수 계산 (warmup 비율 적용)
    total_steps = len(train_dataloader) * config.training.epochs
    warmup_steps = 0
    
    # warmup_steps 자동 계산 처리
    if hasattr(config.training, 'warmup_steps'):
        if config.training.warmup_steps == "auto":
            # "auto"인 경우 warmup_ratio 사용
            if hasattr(config.training, 'warmup_ratio') and config.training.warmup_ratio > 0:
                warmup_steps = int(total_steps * config.training.warmup_ratio)
                if local_rank == 0:
                    print(f"🔥 Warmup 자동 설정: {warmup_steps} 스텝 (총 {total_steps} 스텝 중 {config.training.warmup_ratio*100:.1f}%)")
        elif isinstance(config.training.warmup_steps, int) and config.training.warmup_steps > 0:
            # 정수로 직접 지정된 경우
            warmup_steps = config.training.warmup_steps
            if local_rank == 0:
                print(f"🔥 Warmup 수동 설정: {warmup_steps} 스텝")
    elif hasattr(config.training, 'warmup_ratio') and config.training.warmup_ratio > 0:
        # warmup_steps가 없고 warmup_ratio만 있는 경우
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        if local_rank == 0:
            print(f"🔥 Warmup 비율 설정: {warmup_steps} 스텝 (총 {total_steps} 스텝 중 {config.training.warmup_ratio*100:.1f}%)")
    
    # scheduler 설정에 동적 warmup 적용 (DeepSpeed는 'scheduler' 키 사용)
    # warmup 설정과 관계없이 항상 scheduler 파라미터를 업데이트해야 함
    if 'scheduler' in ds_config:
        if 'params' in ds_config['scheduler']:
            # learning_rate 가져오기 (optimizer에서)
            learning_rate = ds_config.get('optimizer', {}).get('params', {}).get('lr', 1e-5)
            scheduler_type = ds_config['scheduler'].get('type', '')
            
            # 기본 warmup_steps 설정 (warmup이 없어도 scheduler는 작동해야 함)
            if warmup_steps == 0:
                warmup_steps = int(total_steps * 0.1)  # 기본값: 전체의 10%
            
            # WarmupCosineLR 파라미터 처리
            if scheduler_type == "WarmupCosineLR":
                # total_num_steps 자동 설정
                if ds_config['scheduler']['params'].get('total_num_steps') == "auto":
                    ds_config['scheduler']['params']['total_num_steps'] = total_steps
                
                # warmup_num_steps 자동 설정
                if ds_config['scheduler']['params'].get('warmup_num_steps') == "auto":
                    ds_config['scheduler']['params']['warmup_num_steps'] = min(200, warmup_steps)
                
                # warmup_min_ratio와 cos_min_ratio는 이미 설정되어 있음
                # WarmupCosineLR은 ratio를 사용하므로 absolute lr 값은 설정하지 않음
                
            # WarmupLR 파라미터 처리
            elif scheduler_type == "WarmupLR":
                if ds_config['scheduler']['params'].get('warmup_num_steps') == "auto":
                    ds_config['scheduler']['params']['warmup_num_steps'] = min(200, warmup_steps)
                
                if ds_config['scheduler']['params'].get('warmup_min_lr') == "auto":
                    ds_config['scheduler']['params']['warmup_min_lr'] = learning_rate * 0.1
                
                if ds_config['scheduler']['params'].get('warmup_max_lr') == "auto":
                    ds_config['scheduler']['params']['warmup_max_lr'] = learning_rate
            
            if local_rank == 0:
                lr_params = ds_config['scheduler']['params']
                actual_warmup = lr_params.get('warmup_num_steps', warmup_steps)
                actual_total = lr_params.get('total_num_steps', total_steps)
                
                if scheduler_type == "WarmupCosineLR":
                    warmup_min_ratio = lr_params.get('warmup_min_ratio', 0.1)
                    cos_min_ratio = lr_params.get('cos_min_ratio', 0.05)
                    print(f"   📊 스케줄러 설정 업데이트:")
                    print(f"      - warmup_num_steps: {actual_warmup}")
                    print(f"      - total_num_steps: {actual_total}")
                    print(f"      - warmup_min_ratio: {warmup_min_ratio}")
                    print(f"      - cos_min_ratio: {cos_min_ratio}")
                    print(f"      - 시작 LR: {learning_rate * warmup_min_ratio:.2e}")
                    print(f"      - 최대 LR: {learning_rate:.2e}")
                    print(f"      - 최종 LR: {learning_rate * cos_min_ratio:.2e}")
                else:
                    actual_min_lr = lr_params.get('warmup_min_lr', learning_rate * 0.1)
                    actual_max_lr = lr_params.get('warmup_max_lr', learning_rate)
                    print(f"   📊 스케줄러 설정 업데이트:")
                    print(f"      - warmup_num_steps: {actual_warmup}")
                    print(f"      - warmup_min_lr: {actual_min_lr:.2e}")
                    print(f"      - warmup_max_lr: {actual_max_lr:.2e}")
                    print(f"      - total_num_steps: {actual_total}")
                    
            if local_rank == 0:
                print(f"🔍 최종 ds_config scheduler 설정:")
                print(f"   - scheduler params: {ds_config['scheduler'].get('params', {})}")
    
    # DeepSpeed 초기화 (체크포인트 로딩 없이)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        training_data=train_dataloader.dataset,
        collate_fn=train_dataloader.collate_fn
    )
    
    if local_rank == 0:
        print("✅ DeepSpeed 엔진 초기화 완료 (체크포인트 로딩 전)")
        model.print_trainable_parameters()
        
        # Warmup 설정 정보 출력
        if hasattr(config.training, 'warmup_enabled') and config.training.warmup_enabled:
            # learning_rate 가져오기 (optimizer에서)
            learning_rate = ds_config.get('optimizer', {}).get('params', {}).get('lr', 1e-5)
            
            print(f"🔥 Warmup 활성화:")
            print(f"   - Warmup 스텝: {warmup_steps if 'warmup_steps' in locals() else 'N/A'}")
            print(f"   - Warmup 비율: {config.training.warmup_ratio*100:.1f}%" if hasattr(config.training, 'warmup_ratio') else "")
            print(f"   - Warmup 타입: {ds_config.get('optimizer', {}).get('params', {}).get('warmup_type', 'N/A')}")
            if 'scheduler' in ds_config and 'params' in ds_config['scheduler']:
                lr_params = ds_config['scheduler']['params']
                scheduler_type = ds_config['scheduler'].get('type', '')
                
                if scheduler_type == "WarmupCosineLR":
                    # WarmupCosineLR은 ratio를 사용
                    warmup_min_ratio = lr_params.get('warmup_min_ratio', 0.1)
                    cos_min_ratio = lr_params.get('cos_min_ratio', 0.05)
                    print(f"   - Warmup 시작 LR: {learning_rate * warmup_min_ratio:.2e} (lr × {warmup_min_ratio})")
                    print(f"   - Warmup 종료 LR: {learning_rate:.2e} (기본 lr)")
                    print(f"   - Cosine 최종 LR: {learning_rate * cos_min_ratio:.2e} (lr × {cos_min_ratio})")
                    print(f"   - Warmup 스텝: {lr_params.get('warmup_num_steps', 'N/A')}")
                    print(f"   - 총 스텝: {lr_params.get('total_num_steps', 'N/A')}")
                else:
                    # WarmupLR 등 다른 스케줄러
                    print(f"   - 최소 학습률: {lr_params.get('warmup_min_lr', 'N/A')}")
                    print(f"   - 최대 학습률: {lr_params.get('warmup_max_lr', 'N/A')}")
        
        # GPU 메모리 상태 확인
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"🔍 DeepSpeed 초기화 후 GPU 메모리: {memory_allocated:.1f}GB")
    
    # 🚀 체크포인트 경로 확인 (MLflow/DeepSpeed 초기화 전)
    checkpoint_dir = path_manager.get('checkpoints')
    
    # 체크포인트 정보 미리 확인
    checkpoint_to_load = None
    tag_to_load = None
    
    if checkpoint_path or checkpoint_tag or args.resume_from_checkpoint:
        # 명시적 체크포인트 경로가 없으면 최신 체크포인트 찾기
        if not checkpoint_path:
            found_dir, found_tag, found_epoch = find_latest_checkpoint(checkpoint_dir, local_rank)
            if found_dir:
                checkpoint_to_load = found_dir
                tag_to_load = found_tag if checkpoint_tag == "latest" else checkpoint_tag
                start_epoch = found_epoch
                logger.info(f"🔍 최신 체크포인트 발견: {found_tag} (epoch {found_epoch})")
            else:
                logger.warning("체크포인트를 찾을 수 없습니다. 처음부터 시작합니다.")
        else:
            checkpoint_to_load = checkpoint_path
            tag_to_load = checkpoint_tag
    
    # 체크포인트 로딩 전에 resume_run_id 초기화
    resume_run_id = None
    
    # MLflow 설정은 체크포인트 로딩 후로 이동
    run_id = None
    
    # 일단 체크포인트 정보만 확인 - MLflow 시작은 체크포인트 로딩 후로 이동
    if local_rank == 0:
        print("🔍 체크포인트 로딩을 위해 MLflow 초기화 지연...")
        
    
    # 모든 rank에서 run_id 공유 (DeepSpeed 분산 환경)
    if torch.distributed.is_initialized():
        # run_id를 모든 rank에 브로드캐스트
        if local_rank == 0:
            run_id_tensor = torch.tensor([hash(run_id) if run_id else 0], dtype=torch.long).cuda()
        else:
            run_id_tensor = torch.tensor([0], dtype=torch.long).cuda()
        torch.distributed.broadcast(run_id_tensor, 0)
        
        if local_rank == 0:
            print(f"🔄 MLflow run_id 모든 rank에 공유: {run_id}")
    
    # 체크포인트 로딩 변수 초기화
    loaded_run_id = None
    
    # 🚀 체크포인트 로딩 (DeepSpeed 초기화 후)
    if checkpoint_to_load and tag_to_load:
        logger.info(f"🔄 체크포인트 로딩 시작: {checkpoint_to_load}, tag: {tag_to_load}")
        loaded_client_state, loaded_start_epoch, loaded_run_id = load_checkpoint_deepspeed(
            model_engine, 
            scheduler, 
            checkpoint_to_load, 
            tag_to_load, 
            local_rank
        )
        
        if loaded_client_state:
            # 체크포인트에서 로드된 정보 사용
            start_epoch = loaded_start_epoch  # 완료된 에폭 번호
            if loaded_run_id and not resume_run_id:
                resume_run_id = loaded_run_id  # 체크포인트의 run_id 사용
            
            
            if local_rank == 0:
                print(f"✅ 체크포인트 로드 완료!")
                print(f"   - 완료된 에폭: {start_epoch}")
                print(f"   - 다음 에폭부터 재개: {start_epoch + 1}")
                print(f"   - MLflow run_id: {resume_run_id or 'N/A'}")
                if isinstance(loaded_client_state, dict):
                    print(f"   - 훈련 손실: {loaded_client_state.get('train_loss', 'N/A')}")
                    print(f"   - 검증 정확도: {loaded_client_state.get('val_accuracy', 'N/A')}")
            
            is_resume_training = True
    
    # 🚀 MLflow 초기화 (체크포인트 로딩 후)
    if local_rank == 0:
        # MLflow 프로젝트 환경에서 이미 run이 시작되었는지 확인
        active_run = mlflow.active_run()
        mlflow_run_id = os.environ.get('MLFLOW_RUN_ID')
        
        if active_run or mlflow_run_id:
            # 이미 MLflow 프로젝트에서 run이 시작됨
            if active_run:
                print(f"🎯 MLflow 프로젝트 run 사용: {active_run.info.run_id}")
                print(f"   실험 ID: {active_run.info.experiment_id}")
                print(f"   실행 이름: {active_run.info.run_name}")
                run_id = active_run.info.run_id
            else:
                print(f"🎯 MLflow 프로젝트 환경 변수 감지: {mlflow_run_id}")
                run_id = mlflow_run_id
        else:
            # CLI 인자로 전달받은 run_id 확인
            cli_run_id = args.run_id or args.mlflow_run_id
            if cli_run_id:
                resume_run_id = cli_run_id
                print(f"🎯 CLI에서 전달받은 run_id: {resume_run_id}")
            
            # 체크포인트에서 로드한 run_id가 있으면 우선 사용
            if loaded_run_id and not resume_run_id:
                resume_run_id = loaded_run_id
                print(f"🎯 체크포인트에서 MLflow run_id 추출: {resume_run_id}")
            
            # 일반 모드에서 새로운 run 시작 또는 기존 run 재개
            if resume_run_id:
                try:
                    print(f"🔄 기존 MLflow run 재개 시도: {resume_run_id}")
                    # MLflow 실험 설정
                    if args.experiment_name:
                        mlflow.set_experiment(args.experiment_name)
                    else:
                        mlflow.set_experiment("PEFT_DeepSpeed_Enhanced")
                    
                    # 기존 run에 이어서 기록
                    mlflow.start_run(run_id=resume_run_id)
                    print(f"✅ 기존 MLflow run 재개 성공: {resume_run_id}")
                    print(f"   → 기존 폴더에 이어서 저장됩니다.")
                    run_id = resume_run_id
                except Exception as e:
                    print(f"⚠️ 기존 MLflow run 재개 실패: {e}")
                    print("🔄 새로운 run을 생성합니다...")
                    resume_run_id = None
            
            if not resume_run_id:
                # 새 MLflow run 시작
                if args.run_name:
                    # 사용자가 run_name을 지정한 경우
                    run_name = args.run_name
                    
                    # 에폭 정보 동적 추가 (run_name에 :Epoch_ 포함된 경우)
                    if ":Epoch_" in run_name:
                        # 시작 에폭 기반으로 에폭 번호 업데이트
                        epoch_part = f"Epoch_{start_epoch + 1:02d}"
                        run_name = run_name.split(":Epoch_")[0] + f":{epoch_part}"
                    
                    print(f"🏷️ 사용자 지정 run_name: {run_name}")
                    if checkpoint_path:
                        print(f"   (체크포인트 재개: {checkpoint_tag})")
                else:
                    # 기본 run_name 생성
                    run_name = f"PEFT_DeepSpeed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if checkpoint_path:
                        run_name += f"_resume_from_{checkpoint_tag}"
                    
                    print(f"🏷️ 기본 run_name 생성: {run_name}")
                
                mlflow.set_experiment("PEFT_DeepSpeed_Enhanced")
                mlflow.start_run(run_name=run_name)
                print(f"🎯 새 MLflow run 시작: {run_name}")
                run_id = mlflow.active_run().info.run_id
        
        # 안전한 파라미터 로깅 (rank 0 전용)
        final_epochs = args.epochs if args.epochs > 0 else config.training.epochs
        final_batch_size = args.batch_size if args.batch_size > 0 else config.deepspeed.train_micro_batch_size_per_gpu
        final_learning_rate = args.learning_rate if args.learning_rate > 0.0 else config.deepspeed.optimizer.params.lr
        
        def safe_log_params(params_dict):
            """MLflow 파라미터 중복 로깅 방지 (rank 0 전용)"""
            for key, value in params_dict.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e:
                    if "already logged" in str(e):
                        print(f"   ⚠️ 파라미터 '{key}' 이미 로깅됨 (스킵)")
                    else:
                        print(f"   ❌ 파라미터 '{key}' 로깅 실패: {e}")
        
        # 파라미터 로깅 실행 (rank 0만)
        try:
            # Warmup 관련 파라미터 수집
            warmup_params = {}
            if hasattr(config.training, 'warmup_enabled') and config.training.warmup_enabled:
                warmup_params.update({
                    "warmup_enabled": config.training.warmup_enabled,
                    "warmup_steps": warmup_steps if 'warmup_steps' in locals() else 0,
                    "warmup_ratio": config.training.warmup_ratio if hasattr(config.training, 'warmup_ratio') else 0.1,
                    "total_steps": total_steps if 'total_steps' in locals() else 0
                })
                
                # DeepSpeed 설정에서 warmup 정보 추가
                if 'optimizer' in ds_config and 'params' in ds_config['optimizer']:
                    opt_params = ds_config['optimizer']['params']
                    warmup_params.update({
                        "warmup_type": opt_params.get('warmup_type', 'linear'),
                        "warmup_bias_lr": opt_params.get('warmup_bias_lr', 0.0),
                        "warmup_momentum": opt_params.get('warmup_momentum', 0.8)
                    })
                
                # 스케줄러 설정에서 warmup 정보 추가
                if 'scheduler' in ds_config and 'params' in ds_config['scheduler']:
                    lr_params = ds_config['scheduler']['params']
                    scheduler_type = ds_config['scheduler'].get('type', 'WarmupCosineLR')
                    
                    if scheduler_type == "WarmupCosineLR":
                        warmup_params.update({
                            "warmup_min_ratio": lr_params.get('warmup_min_ratio', 0.1),
                            "cos_min_ratio": lr_params.get('cos_min_ratio', 0.05),
                            "scheduler_type": scheduler_type
                        })
                    else:
                        warmup_params.update({
                            "warmup_min_lr": lr_params.get('warmup_min_lr', 1e-6),
                            "warmup_max_lr": lr_params.get('warmup_max_lr', 2e-5),
                            "scheduler_type": scheduler_type
                        })
            
            # 기본 파라미터와 warmup 파라미터 병합
            params_dict = {
                "model_name": config.model.name,
                "num_classes": num_classes,
                "epochs": final_epochs,
                "batch_size": final_batch_size,
                "learning_rate": final_learning_rate,
                "weight_decay": config.deepspeed.optimizer.params.weight_decay,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "extract_ratio": config.data.extract_ratio,
                "max_length": config.data.max_length,
                "torch_dtype": config.model.torch_dtype,
                "checkpoint_path": checkpoint_path or "",
                "checkpoint_tag": checkpoint_tag or "",
                "data_fraction": args.data_fraction,
                "quick_test": args.quick_test,
                "mode": args.mode,
                "resume_run_id": resume_run_id or "",
                "local_rank": local_rank,
                **warmup_params  # warmup 파라미터 추가
            }
            
            safe_log_params(params_dict)
            print(f"✅ MLflow 파라미터 로깅 완료 (warmup 정보 포함)")
        except Exception as e:
            print(f"⚠️ MLflow 파라미터 로깅 중 오류: {e}")
    
    # YAML 설정 오버라이드 (모든 rank에서 실행)
    if args.epochs > 0:
        config.training.epochs = args.epochs
        if local_rank == 0:
            print(f"🔧 에폭 오버라이드: {args.epochs}")
    
    if args.batch_size > 0:
        config.deepspeed.train_micro_batch_size_per_gpu = args.batch_size
        if local_rank == 0:
            print(f"🔧 배치 크기 오버라이드: {args.batch_size}")
    
    if args.learning_rate > 0.0:
        config.deepspeed.optimizer.params.lr = args.learning_rate
        if local_rank == 0:
            print(f"🔧 학습률 오버라이드: {args.learning_rate}")
    
    if local_rank == 0:
        print(f"📊 최종 설정: 에폭={config.training.epochs}, 배치={config.deepspeed.train_micro_batch_size_per_gpu}, 학습률={config.deepspeed.optimizer.params.lr}")
        if is_resume_training:
            print(f"🎯 체크포인트에서 에폭 {start_epoch} 완료 확인")
            print(f"🚀 학습 재개: 에폭 {start_epoch + 1}부터 {config.training.epochs}까지")
        else:
            print(f"🎯 시작 에폭: {start_epoch}")
            print(f"🚀 학습 범위: 에폭 {start_epoch + 1}부터 {config.training.epochs}까지")
        print(f"🔑 MLflow Run ID: {run_id}")
        print("")
    
    # 🔥 훈련 루프 시작
    # YAML 설정에서 progress_interval 가져오기
    progress_interval = config.deepspeed.steps_per_print if hasattr(config.deepspeed, 'steps_per_print') else config.training.progress_interval
    
    logger.info(f"📊 진행률 출력 간격: {progress_interval} 스텝마다")
    
    # GPU 모니터 초기 상태
    gpu_monitor.log_memory("학습 시작 전")
    
    for epoch in range(start_epoch, config.training.epochs):
        model_engine.train()
        
        # 분산 샘플러 에폭 설정
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        # 진행률바 설정 (재개 시 올바른 에폭 표시)
        current_epoch = epoch + 1
        if local_rank == 0:
            # 재개 시에도 정확한 진행 상황 표시
            desc = f"Epoch {current_epoch}/{config.training.epochs}"
            if start_epoch > 0 and epoch == start_epoch:
                desc += " (resumed)"
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=desc)
        else:
            pbar = enumerate(train_dataloader)
        
        total_loss = 0
        num_batches = 0
        
        for step, batch in pbar:
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # DeepSpeed 내장 메트릭과 진행률바 업데이트 (rank 0만)
            if local_rank == 0 and (step + 1) % progress_interval == 0:
                current_lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else 2e-5
                loss_val = loss.item()
                avg_loss = total_loss / num_batches
                # GPU 메모리 상태 가져오기
                gpu_stats = gpu_monitor.get_memory_stats()
                memory_allocated = gpu_stats.get('allocated_gb', 0)
                
                # 현재 글로벌 스텝 계산
                global_step = epoch * len(train_dataloader) + step + 1
                
                # Warmup 상태 확인
                warmup_status = ""
                if hasattr(config.training, 'warmup_enabled') and config.training.warmup_enabled:
                    if 'warmup_steps' in locals() and warmup_steps > 0:
                        if global_step <= warmup_steps:
                            warmup_progress = (global_step / warmup_steps) * 100
                            warmup_status = f" [Warmup: {warmup_progress:.1f}%]"
                        else:
                            warmup_status = " [Warmup: Complete]"
                
                postfix = {
                    'loss': f'{loss_val:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'mem': f'{memory_allocated:.1f}GB'
                }
                
                # warmup 상태를 postfix에 추가
                if warmup_status:
                    postfix['warmup'] = warmup_status
                
                pbar.set_postfix(postfix)
                
                # MLflow 실시간 로깅 (rank 0 전용)
                try:
                    metrics_dict = {
                        f'train_loss_step': loss.item(),
                        f'avg_train_loss': avg_loss,
                        f'learning_rate': current_lr,
                        f'memory_allocated_gb': memory_allocated,
                        f'global_step': global_step
                    }
                    
                    # Warmup 정보 추가
                    if hasattr(config.training, 'warmup_enabled') and config.training.warmup_enabled:
                        if 'warmup_steps' in locals() and warmup_steps > 0:
                            warmup_progress = min(global_step / warmup_steps, 1.0)
                            metrics_dict.update({
                                'warmup_progress': warmup_progress,
                                'is_warmup_phase': 1 if global_step <= warmup_steps else 0
                            })
                    
                    mlflow.log_metrics(metrics_dict, step=global_step)
                except Exception as e:
                    print(f"⚠️ MLflow 실시간 로깅 실패: {e}")
        
        # 에폭 평균 손실 계산
        avg_train_loss = total_loss / num_batches
        logger.info(f"📊 에폭 {epoch + 1} 훈련 평균 손실: {avg_train_loss:.4f}")
        gpu_monitor.log_memory(f"에폭 {epoch + 1} 종료")
        
        # 에폭 종료 후 평가
        metrics = {'loss': avg_train_loss, 'accuracy': 0.0}
        
        if eval_dataloader:
            logger.info(f"🔍 에폭 {epoch + 1} 평가 시작...")
            
            metrics = evaluate_model_comprehensive(model_engine, eval_dataloader, local_rank, metric_tracker, prefix="eval_")
            
            if local_rank == 0:
                logger.info(f"📈 에폭 {epoch + 1} 평가 결과:")
                logger.info(f"  📉 Loss: {metrics['eval_loss']:.4f}")
                logger.info(f"  🎯 Accuracy: {metrics['eval_accuracy']:.4f} ({metrics['eval_accuracy']*100:.2f}%)")
                logger.info(f"  📊 F1 (Weighted): {metrics['eval_f1_weighted']:.4f}")
                logger.info(f"  📊 F1 (Macro): {metrics['eval_f1_macro']:.4f}")
                logger.info(f"  📊 Precision: {metrics['eval_precision']:.4f}")
                logger.info(f"  📊 Recall: {metrics['eval_recall']:.4f}")
                logger.info(f"  🚀 Eval Speed: {metrics['eval_eval_samples_per_sec']:.2f} samples/sec")
                
                # MLflow 종합 로깅 (rank 0 전용)
                try:
                    mlflow.log_metrics({
                        f"train_loss": avg_train_loss,
                        f"eval_loss": metrics['eval_loss'],
                        f"eval_accuracy": metrics['eval_accuracy'],
                        f"eval_f1_weighted": metrics['eval_f1_weighted'], 
                        f"eval_f1_macro": metrics['eval_f1_macro'],
                        f"eval_precision_weighted": metrics['eval_precision'],
                        f"eval_recall_weighted": metrics['eval_recall'],
                        f"eval_samples_per_sec": metrics['eval_eval_samples_per_sec']
                    }, step=epoch + 1)
                    print(f"✅ MLflow 에폭 {epoch + 1} 메트릭 로깅 완료")
                except Exception as e:
                    print(f"⚠️ MLflow 종합 로깅 실패: {e}")
        
        # 🚀 DeepSpeed 내장 체크포인트 저장
        # PathManager를 사용하여 경로 가져오기
        ckpt_root = path_manager.get('checkpoints')
        
        # client_state 준비 (MLflow run_id 포함)
        client_state = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_accuracy': metrics.get('eval_accuracy', 0.0),
            'val_loss': metrics.get('eval_loss', 0.0),
            'timestamp': datetime.now().isoformat(),
            'mlflow_run_id': run_id if run_id else "",  # MLflow run_id 저장
            'num_classes': num_classes,  # 클래스 수 저장 (학습 재개 시 필수)
            'local_rank': local_rank
        }
        
        # 에폭 태그 설정
        epoch_tag = f"epoch_{epoch+1:02d}"
        
        logger.info(f"💾 DeepSpeed 체크포인트 저장: {ckpt_root}")
        logger.info(f"   태그: {epoch_tag}")
        logger.info(f"   MLflow Run ID: {run_id}")
        
        # DeepSpeed 체크포인트 저장 (루트 경로 + 태그)
        model_engine.save_checkpoint(str(ckpt_root), tag=epoch_tag, client_state=client_state)
        
        # 체크포인트 디렉터리 (DeepSpeed가 생성한 경로)
        checkpoint_dir = ckpt_root / epoch_tag
        
        # lr_scheduler와 rng_state 수동 저장 (rank 0만)
        try:
            if local_rank == 0 and scheduler is not None:
                scheduler_path = checkpoint_dir / "lr_scheduler.pt"
                torch.save(scheduler.state_dict(), scheduler_path)
                print(f"   ✅ lr_scheduler 저장: lr_scheduler.pt")
            
            if local_rank == 0:
                rng_state_path = checkpoint_dir / "rng_state.pth"
                rng_state = {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                }
                torch.save(rng_state, rng_state_path)
                print(f"   ✅ RNG 상태 저장: rng_state.pth")
        except Exception as e:
            if local_rank == 0:
                print(f"   ⚠️ lr_scheduler/RNG 저장 실패: {e}")
        
        # 모든 rank 동기화 - 모든 GPU 저장 완료 대기
        if dist.is_initialized():
            dist.barrier()
        
        # rank-0만 추가 상태 저장 및 MLflow artifacts 개선
        if local_rank == 0:
            checkpoint_dir = ckpt_root / epoch_tag
            
            # 추가 메타데이터 저장
            metadata_path = save_model_metadata(str(checkpoint_dir), model, tokenizer, config, num_classes)
            # company_scaling_stats 삭제됨
            
            # 데이터셋 전체 메타데이터 저장 (checkpoints 폴더에 한 번만)
            if epoch == 0:  # 첫 번째 에폭에서만 저장
                dataset_metadata_path = save_dataset_metadata_once(arrow_dir, str(ckpt_root), local_rank)
            
            # 🔥 DeepSpeed 호환 MLflow artifacts 저장 (rank 0 전용)
            import shutil
            
            try:
                # 1. 현재 에폭 폴더 MLflow artifacts 저장
                mlflow.log_artifacts(str(checkpoint_dir), f"checkpoints/{epoch_tag}")
                
                # 2. latest 파일 생성 및 저장
                latest_file = ckpt_root / "latest"
                latest_file.write_text(epoch_tag)
                mlflow.log_artifact(str(latest_file), "checkpoints")
                
                # 3. zero_to_fp32.py 스크립트 복사 및 저장
                zero_to_fp32_src = ckpt_root / "zero_to_fp32.py"
                if zero_to_fp32_src.exists():
                    mlflow.log_artifact(str(zero_to_fp32_src), "checkpoints")
                
                # 4. 첫 번째 에폭에서 dataset_metadata.json도 MLflow에 저장
                if epoch == 0:
                    dataset_metadata_file = ckpt_root / "dataset_metadata.json"
                    if dataset_metadata_file.exists():
                        mlflow.log_artifact(str(dataset_metadata_file), "checkpoints")
                        print(f"   ✅ MLflow dataset_metadata.json 저장: checkpoints/dataset_metadata.json")
                
                print(f"   ✅ MLflow artifacts 저장 완료: checkpoints/{epoch_tag}")
                print(f"   ✅ MLflow latest 파일 저장: checkpoints/latest -> {epoch_tag}")
                print(f"   ✅ MLflow zero_to_fp32.py 저장: checkpoints/zero_to_fp32.py")
                
            except Exception as e:
                print(f"   ⚠️ MLflow artifacts 저장 실패: {e}")
                print(f"   🔄 체크포인트는 로컬에 저장됨: {checkpoint_dir}")
    
    # 🏁 학습 완료 후 Test 데이터로 최종 평가
    if local_rank == 0:
        print(f"\n🏁 학습 완료! Test 데이터로 최종 평가를 진행합니다...")
        print(f"   Test 데이터: {len(test_dataloader.dataset):,}개 샘플, {len(test_dataloader):,}개 배치")
    
    test_metrics = evaluate_model_comprehensive(model_engine, test_dataloader, local_rank, metric_tracker=metric_tracker, prefix="")
    
    if local_rank == 0:
        print(f"\n🎯 최종 Test 평가 결과:")
        print(f"  📉 Test Loss: {test_metrics['loss']:.4f}")
        print(f"  🎯 Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  📊 Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  📊 Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        print(f"  📊 Test Precision (Weighted): {test_metrics['precision']:.4f}")
        print(f"  📊 Test Recall (Weighted): {test_metrics['recall']:.4f}")
        print(f"  🚀 Test Speed: {test_metrics['eval_samples_per_sec']:.2f} samples/sec")
        
        # MLflow에 Test 메트릭 로깅 (rank 0 전용)
        try:
            mlflow.log_metrics({
                "test_loss": test_metrics['loss'],
                "test_accuracy": test_metrics['accuracy'],
                "test_f1_weighted": test_metrics['f1_weighted'],
                "test_f1_macro": test_metrics['f1_macro'],
                "test_precision_weighted": test_metrics['precision'],
                "test_recall_weighted": test_metrics['recall'],
                "test_samples_per_sec": test_metrics['eval_samples_per_sec']
            })
            print("✅ MLflow Test 메트릭 로깅 완료")
        except Exception as e:
            print(f"⚠️ MLflow Test 메트릭 로깅 실패: {e}")
        
        try:
            mlflow.end_run()
            print("🎉 MLflow run 종료 및 훈련 완료!")
        except Exception as e:
            print(f"⚠️ MLflow run 종료 실패: {e}")
            print("🎉 훈련 완료!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Debugging Utilities for Training Interruption Analysis
pj_peft_deepspeed2_moniter_v2

패딩 검증 및 토크나이저 디버깅 함수들 포함
"""

import os
import sys
import time
import torch
import psutil
import logging
import traceback
import signal
import gc
from datetime import datetime
from typing import Dict, Any, Optional, List
import json


class TrainingDebugger:
    """학습 중단 원인 분석을 위한 디버깅 클래스"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.debug_config = config.get('debug', {})
        self.interruption_config = self.debug_config.get('interruption_analysis', {})
        
        self.enabled = self.debug_config.get('enabled', False)
        self.memory_tracking = self.interruption_config.get('memory_tracking', True)
        self.detailed_logging = self.interruption_config.get('detailed_logging', True)
        self.exception_handling = self.interruption_config.get('exception_handling', True)
        
        # 메모리 및 시스템 추적
        self.memory_history = []
        self.system_metrics = []
        self.step_timings = []
        
        # 시그널 핸들러 설정
        if self.exception_handling:
            self.setup_signal_handlers()
            
        self.logger.info(f"🐛 디버깅 모드: {'활성화' if self.enabled else '비활성화'}")
        if self.enabled:
            self.logger.info(f"   📊 메모리 추적: {self.memory_tracking}")
            self.logger.info(f"   📝 상세 로깅: {self.detailed_logging}")
            self.logger.info(f"   🛡️ 예외 처리: {self.exception_handling}")
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정 (SIGTERM, SIGINT 등)"""
        def signal_handler(signum, frame):
            self.logger.error(f"🚨 시그널 {signum} 수신됨 - 훈련 중단")
            self.save_debug_info(interruption_type="signal", signal_num=signum)
            sys.exit(1)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def track_memory_usage(self, step: int, phase: str = "training"):
        """메모리 사용량 추적"""
        if not self.enabled or not self.memory_tracking:
            return
        
        # GPU 메모리
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1024**3,
                    'reserved': torch.cuda.memory_reserved(i) / 1024**3,
                    'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3
                }
        
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        
        memory_info = {
            'step': step,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'gpu_memory': gpu_memory,
            'cpu_memory': {
                'used_gb': cpu_memory.used / 1024**3,
                'available_gb': cpu_memory.available / 1024**3,
                'percent': cpu_memory.percent
            }
        }
        
        self.memory_history.append(memory_info)
        
        # 메모리 부족 경고
        if cpu_memory.percent > 90:
            self.logger.warning(f"⚠️ CPU 메모리 사용률 높음: {cpu_memory.percent}%")
        
        for gpu_id, gpu_info in gpu_memory.items():
            if gpu_info['allocated'] > 20:  # 20GB 이상
                self.logger.warning(f"⚠️ {gpu_id} 메모리 사용량 높음: {gpu_info['allocated']:.2f}GB")
    
    def track_system_metrics(self, step: int):
        """시스템 메트릭 추적"""
        if not self.enabled:
            return
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 디스크 사용률
        disk_usage = psutil.disk_usage('/')
        
        # 네트워크 통계 (멀티 GPU 통신 추적)
        net_io = psutil.net_io_counters()
        
        system_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            },
            'process_count': len(psutil.pids())
        }
        
        self.system_metrics.append(system_info)
        
        # 시스템 부하 경고
        if cpu_percent > 95:
            self.logger.warning(f"⚠️ CPU 사용률 과부하: {cpu_percent}%")
    
    def track_step_timing(self, step: int, duration: float, phase: str = "training"):
        """스텝 시간 추적"""
        if not self.enabled:
            return
        
        timing_info = {
            'step': step,
            'phase': phase,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        self.step_timings.append(timing_info)
        
        # 비정상적으로 긴 스텝 경고
        if duration > 60:  # 60초 이상
            self.logger.warning(f"⚠️ 스텝 {step} 실행 시간 과도: {duration:.2f}초")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """예외 상황 로깅"""
        if not self.enabled:
            return
        
        self.logger.error(f"🚨 예외 발생 - {context}")
        self.logger.error(f"예외 타입: {type(exception).__name__}")
        self.logger.error(f"예외 메시지: {str(exception)}")
        self.logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        
        # 디버그 정보 저장
        self.save_debug_info(
            interruption_type="exception",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            context=context
        )
    
    def check_environment_issues(self) -> Dict[str, Any]:
        """환경 문제 체크"""
        issues = {}
        
        # CUDA 관련 체크
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                if device_props.total_memory < 8 * 1024**3:  # 8GB 미만
                    issues[f'gpu_{i}_memory'] = f"GPU {i} 메모리 부족: {device_props.total_memory / 1024**3:.1f}GB"
        
        # 디스크 공간 체크
        disk_usage = psutil.disk_usage('/')
        if disk_usage.free < 10 * 1024**3:  # 10GB 미만
            issues['disk_space'] = f"디스크 공간 부족: {disk_usage.free / 1024**3:.1f}GB 남음"
        
        # 메모리 체크
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024**3:  # 4GB 미만
            issues['system_memory'] = f"시스템 메모리 부족: {memory.available / 1024**3:.1f}GB 사용가능"
        
        return issues
    
    def save_debug_info(self, interruption_type: str = "unknown", **kwargs):
        """디버그 정보 저장"""
        if not self.enabled:
            return
        
        debug_info = {
            'interruption_type': interruption_type,
            'timestamp': datetime.now().isoformat(),
            'environment_issues': self.check_environment_issues(),
            'memory_history': self.memory_history[-10:],  # 최근 10개
            'system_metrics': self.system_metrics[-10:],   # 최근 10개
            'step_timings': self.step_timings[-20:],       # 최근 20개
            'additional_info': kwargs
        }
        
        # 디버그 파일 저장
        debug_file = f"debug_interruption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            self.logger.info(f"🐛 디버그 정보 저장됨: {debug_file}")
        except Exception as e:
            self.logger.error(f"디버그 정보 저장 실패: {e}")
    
    def cleanup(self):
        """디버깅 세션 정리"""
        if not self.enabled:
            return
        
        self.logger.info("🐛 디버깅 세션 정리 중...")
        
        # 최종 디버그 정보 저장
        self.save_debug_info(interruption_type="cleanup")
        
        # 메모리 정리
        self.memory_history.clear()
        self.system_metrics.clear()
        self.step_timings.clear()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Python 가비지 컬렉션
        gc.collect()
        
        self.logger.info("🐛 디버깅 세션 정리 완료")


def create_debugger(config: Dict[str, Any], logger: logging.Logger) -> TrainingDebugger:
    """디버거 인스턴스 생성"""
    return TrainingDebugger(config, logger) 


# =============================================================================
# 패딩 검증 및 토크나이저 디버깅 함수들
# =============================================================================

def validate_tokenizer_settings(tokenizer, model, phase: str = "train", local_rank: int = 0) -> bool:
    """
    토크나이저 설정을 검증하는 함수.
    
    Args:
        tokenizer: 토크나이저 객체
        model: 모델 객체
        phase: 단계 ("train" 또는 "inference")
        local_rank: 로컬 랭크 (분산 학습용)
        
    Returns:
        bool: 검증 성공 여부
    """
    if local_rank == 0:  # 메인 프로세스에서만 출력
        print(f"🔍 토크나이저 설정 검증 ({phase})")
        print(f"   - pad_token: {tokenizer.pad_token}")
        print(f"   - pad_token_id: {tokenizer.pad_token_id}")
        print(f"   - eos_token: {tokenizer.eos_token}")
        print(f"   - eos_token_id: {tokenizer.eos_token_id}")
        print(f"   - padding_side: {tokenizer.padding_side}")
        print(f"   - model.config.pad_token_id: {model.config.pad_token_id}")
    
    # 검증 로직
    validation_errors = []
    
    # 1. 패드 토큰 ID 일관성 검증 (Qwen3: pad_token_id == eos_token_id 여야 함)
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        validation_errors.append("Qwen3에서는 패드 토큰 ID와 EOS 토큰 ID가 같아야 합니다")
    
    # 2. 모델-토크나이저 설정 일관성 검증
    if model.config.pad_token_id != tokenizer.pad_token_id:
        validation_errors.append("모델과 토크나이저의 패드 토큰 ID가 다릅니다")
    
    # 3. 단계별 패딩 방향 검증
    if phase == "train" and tokenizer.padding_side != "right":
        validation_errors.append("훈련 시에는 오른쪽 패딩이 권장됩니다")
    elif phase == "inference" and tokenizer.padding_side != "left":
        validation_errors.append("추론 시에는 왼쪽 패딩이 필수입니다")
    
    # 4. 패드 토큰 존재성 검증
    if tokenizer.pad_token is None:
        validation_errors.append("패드 토큰이 설정되지 않았습니다")
    
    # 5. 토큰 ID 유효성 검증
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        validation_errors.append("유효하지 않은 패드 토큰 ID입니다")
    
    # 검증 결과 출력
    if validation_errors:
        if local_rank == 0:
            print("❌ 토크나이저 설정 검증 실패:")
            for error in validation_errors:
                print(f"   - {error}")
        return False
    else:
        if local_rank == 0:
            print("✅ 토크나이저 설정 검증 성공")
        return True


def verify_batch_padding(batch, tokenizer, max_samples: int = 3, local_rank: int = 0):
    """
    배치의 패딩 상태를 검증하는 함수.
    
    Args:
        batch: 토크나이저로 처리된 배치
        tokenizer: 토크나이저 객체
        max_samples: 검증할 최대 샘플 수
        local_rank: 로컬 랭크
    """
    if local_rank != 0:  # 메인 프로세스에서만 실행
        return
    
    print("🔍 배치 패딩 상태 검증:")
    
    # 배치 정보 출력
    if 'input_ids' in batch:
        batch_size = batch['input_ids'].shape[0]
        seq_length = batch['input_ids'].shape[1]
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 시퀀스 길이: {seq_length}")
        
        # 샘플별 패딩 상태 확인
        for i in range(min(max_samples, batch_size)):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i] if 'attention_mask' in batch else None
            
            # 패드 토큰 위치 찾기
            pad_positions = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            non_pad_positions = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            
            print(f"   - 샘플 {i+1}:")
            print(f"     * 패드 토큰 수: {len(pad_positions)}")
            print(f"     * 실제 토큰 수: {len(non_pad_positions)}")
            
            if len(pad_positions) > 0:
                # 패딩 방향 확인
                if pad_positions[0] < non_pad_positions[0]:
                    padding_side = "left"
                else:
                    padding_side = "right"
                print(f"     * 패딩 방향: {padding_side}")
                
                # 어텐션 마스크 일치성 확인
                if attention_mask is not None:
                    pad_mask_zeros = (attention_mask == 0).sum().item()
                    if pad_mask_zeros == len(pad_positions):
                        print(f"     * 어텐션 마스크: ✅ 일치")
                    else:
                        print(f"     * 어텐션 마스크: ❌ 불일치 ({pad_mask_zeros} vs {len(pad_positions)})")


def test_tokenizer_padding(tokenizer, test_texts: List[str], phase: str = "train", local_rank: int = 0):
    """
    토크나이저 패딩 동작을 테스트하는 함수.
    
    Args:
        tokenizer: 토크나이저 객체
        test_texts: 테스트할 텍스트 리스트
        phase: 단계 ("train" 또는 "inference")
        local_rank: 로컬 랭크
    """
    if local_rank != 0:  # 메인 프로세스에서만 실행
        return
    
    print(f"🧪 토크나이저 패딩 테스트 ({phase}):")
    
    # 테스트 텍스트 토크나이징
    encoded = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # 결과 분석
    print(f"   - 입력 텍스트 수: {len(test_texts)}")
    print(f"   - 출력 배치 크기: {encoded['input_ids'].shape}")
    
    # 패딩 상태 확인
    verify_batch_padding(encoded, tokenizer, max_samples=min(3, len(test_texts)), local_rank=0)
    
    # 디코딩 테스트
    print("   - 디코딩 테스트:")
    for i, text in enumerate(test_texts[:3]):
        decoded = tokenizer.decode(encoded['input_ids'][i], skip_special_tokens=False)
        print(f"     * 원본 {i+1}: {text[:50]}...")
        print(f"     * 디코딩: {decoded[:100]}...") 
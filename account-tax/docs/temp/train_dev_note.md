# conf/base/train.yaml
model:
name_or_path: "Qwen/Qwen2.5-0.5B"
data:
train_path: "data/03_primary/train.parquet"
eval_path: "data/03_primary/valid.parquet"
text_field: "text"
max_length: 1024
training:
epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
lr: 2.0e-5
weight_decay: 0.0
warmup_ratio: 0.03
seed: 42
bf16: true
use_flash_attn: false
num_workers: 4
num_workers_eval: 2
lora:
r: 16
alpha: 32
dropout: 0.05
target_modules: ["q_proj","k_proj","v_proj","o_proj"]
logging:
out_dir: "data/05_model/run_yaml"
metrics_path: "data/05_model/run_yaml/metrics.json"
# ← DeepSpeed 설정을 JSON 파일이 아닌, YAML 섹션으로 직접 제공
deepspeed:
train_micro_batch_size_per_gpu: 1
gradient_accumulation_steps: 8
optimizer:
type: AdamW
params:
lr: 2.0e-5
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0
zero_optimization:
stage: 3
offload_optimizer:
device: "none" # 필요 시 "cpu"
offload_param:
device: "none" # 필요 시 "cpu"
overlap_comm: true
contiguous_gradients: true
reduce_bucket_size: 5.0e7
stage3_prefetch_bucket_size: 5.0e7
stage3_param_persistence_threshold: 1.0e5
bf16:
enabled: true
gradient_clipping: 1.0
wall_clock_breakdown: false




# src/train/main_yaml.py
import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import pandas as pd
import yaml
# -------- Dataset (텍스트 컬럼명은 YAML에서 받음) ----------
class ParquetTextDataset(Dataset):
def __init__(self, path: str, tokenizer: AutoTokenizer, text_field: str, max_len: int):
df = pd.read_parquet(path)
self.texts = df[text_field].astype(str).tolist()
self.tok = tokenizer
self.max_len = max_len
def __len__(self): return len(self.texts)
def __getitem__(self, idx: int):
enc = self.tok(
self.texts[idx],
truncation=True,
max_length=self.max_len,
padding="max_length",
return_tensors="pt",
)
item = {k: v.squeeze(0) for k, v in enc.items()}
item["labels"] = item["input_ids"].clone()
return item
def parse_args():
ap = argparse.ArgumentParser()
ap.add_argument("--config_yml", type=str, required=True)
return ap.parse_args()
def is_rank0() -> bool:
r = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
try:
return int(r) == 0
except:
return True
def main():
args = parse_args()
with open(args.config_yml, "r", encoding="utf-8") as f:
cfg = yaml.safe_load(f)
# --- 섹션별 단축 참조
model_cfg = cfg.get("model", {})
data_cfg = cfg.get("data", {})
train_cfg = cfg.get("training", {})
lora_cfg_yml = cfg.get("lora", {})
ds_cfg = cfg.get("deepspeed", {}) # ← dict 그대로 deepspeed.initialize에 전달
log_cfg = cfg.get("logging", {})
# 재현성
seed = int(train_cfg.get("seed", 42))
torch.manual_seed(seed)
# 디바이스 매핑
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
# 토크나이저/모델
model_id = model_cfg["name_or_path"]
bf16 = bool(train_cfg.get("bf16", True))
use_flash = bool(train_cfg.get("use_flash_attn", False))
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
model_id,
torch_dtype=torch.bfloat16 if bf16 else torch.float32,
attn_implementation="flash_attention_2" if use_flash else "eager",
)
# LoRA
lcfg = LoraConfig(
r = int(lora_cfg_yml.get("r", 16)),
lora_alpha = int(lora_cfg_yml.get("alpha", 32)),
lora_dropout = float(lora_cfg_yml.get("dropout", 0.05)),
target_modules = [m.strip() for m in lora_cfg_yml.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"])],
bias = "none",
task_type = "CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
# 옵티마이저 (DeepSpeed가 감싼다)
optimizer = torch.optim.AdamW(
model.parameters(),
lr=float(train_cfg.get("lr", 2e-5)),
weight_decay=float(train_cfg.get("weight_decay", 0.0)),
)
# DeepSpeed 초기화 — JSON 파일이 아니라, YAML에서 읽은 dict를 그대로 사용
engine, optimizer, _, _ = deepspeed.initialize(
model=model,
optimizer=optimizer,
model_parameters=model.parameters(),
config=ds_cfg, # ← 여기!
)
# 데이터
text_field = data_cfg.get("text_field", "text")
max_len = int(data_cfg.get("max_length", 1024))
train_path = data_cfg["train_path"]
eval_path = data_cfg.get("eval_path", "")
per_device_bs= int(train_cfg.get("per_device_train_batch_size", 1))
train_ds = ParquetTextDataset(train_path, tokenizer, text_field, max_len)
train_samp = DistributedSampler(train_ds, shuffle=True)
train_dl = DataLoader(
train_ds, batch_size=per_device_bs, sampler=train_samp,
pin_memory=True, num_workers=int(train_cfg.get("num_workers", 4)), drop_last=True
)
eval_dl = None
if eval_path:
eval_ds = ParquetTextDataset(eval_path, tokenizer, text_field, max_len)
eval_samp = DistributedSampler(eval_ds, shuffle=False)
eval_dl = DataLoader(
eval_ds, batch_size=per_device_bs, sampler=eval_samp,
pin_memory=True, num_workers=int(train_cfg.get("num_workers_eval", 2)), drop_last=False
)
# 간단 스케줄러(warmup)
total_steps = int(train_cfg.get("epochs", 3)) * len(train_dl)
warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.03)))
scheduler = torch.optim.lr_scheduler.LinearLR(
optimizer, start_factor=0.0 if warmup_steps > 0 else 1.0,
total_iters=warmup_steps if warmup_steps > 0 else 1
)
# 학습 루프
metrics = {"train_loss": []}
engine.train()
global_step = 0
epochs = int(train_cfg.get("epochs", 3))
out_dir = Path(log_cfg.get("out_dir", "data/05_model/run_yaml"))
metrics_path = log_cfg.get("metrics_path", "")
for ep in range(epochs):
train_samp.set_epoch(ep)
for batch in train_dl:
batch = {k: v.to(engine.local_rank) for k, v in batch.items()}
out = engine(**batch)
loss = out.loss
engine.backward(loss)
engine.step()
if warmup_steps > 0 and global_step < warmup_steps:
scheduler.step()
if is_rank0():
metrics["train_loss"].append(float(loss.detach().item()))
global_step += 1
if eval_dl is not None:
engine.eval()
s, n = 0.0, 0
with torch.no_grad():
for eb in eval_dl:
eb = {k: v.to(engine.local_rank) for k, v in eb.items()}
eo = engine(**eb)
s += float(eo.loss.detach().item()); n += 1
if is_rank0():
metrics.setdefault("eval_loss", []).append(s / max(n, 1))
engine.train()
# 저장(ZeRO-3 권장: 엔진 체크포인트 사용)
if is_rank0():
out_dir.mkdir(parents=True, exist_ok=True)
engine.save_checkpoint(str(out_dir / "ds_ckpt"))
if is_rank0():
# LoRA 어댑터만 별도 저장(선택)
try:
engine.module.save_pretrained(str(out_dir / "lora_adapter"))
except Exception:
pass
if metrics_path:
with open(metrics_path, "w", encoding="utf-8") as f:
json.dump(metrics, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
main()



ZeRO 메모리 최적화, DataLoader 메커니즘, 그리고 훈련 성능 튜닝 레시피 등 분산 훈련 환경변수와 세부 구현 지침을 함께 제공하여 실질적인 운영 가이드라인 역할을 수행합니다.











1) Kedro vs DeepSpeed: 철학과 “원자 단위(atomic unit)”
Kedro: 노드=원자 단위. 경계는 직렬화 가능한 산출물(파일/테이블/모델 가중치)로 끊는다. 목표는 유연성/결정성/재현성/부분 재실행/캐시/라인리지.
DeepSpeed/torchrun: 런처가 스폰한 랭크 프로세스 집합=원자 단위. 한 번의 스폰 수명 안에서 init→train→eval→save가 연속되어야 한다. 목표는 분산 동기·통신 일관성과 장치/통신 핸들의 생명주기 보장.
2) DeepSpeed 런처가 정확히 하는 일
프로세스 스폰: --num_gpus=N이면 N개의 랭크 프로세스(0..N−1)를 동시 실행.
환경변수 세팅: WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT, NCCL_*.
프로세스그룹 초기화: torch.distributed.init_process_group(backend="nccl") 가능한 상태를 보장.
3) 왜 “노드 쪼개기(여러 노드로 학습 분산)”가 문제를 만든다
엔진/통신/디바이스 핸들 전달 불가
핸들은 해당 프로세스의 살아있는 자원(CUDA 컨텍스트, NCCL communicator 등) 참조.
직렬화 불가 + 다른 프로세스(PID)로 이전 불가 → 노드 경계 넘어 재사용 불가.
초기화 타이밍 불일치
분산 init은 모든 랭크가 동시에 같은 구성으로 들어와야 함.
노드 경계마다 init/dispose가 반복되면 포트 충돌/NCCL 타임아웃/데드락.
Kedro 병렬러너와 충돌
ParallelRunner는 노드들을 별도 프로세스로 돌림 → DeepSpeed/torch.distributed와 이중 스폰/충돌.
MLflow Hook 충돌
훅이 모든 랭크에서 실행되면 같은 run/artifact에 동시 기록 → 중복/경합/락.
해법: rank0만 로깅.
ZeRO 체크포인트 불일치
ZeRO-2/3는 샤딩. 엔진 규칙대로 저장/복원하지 않으면 state_dict 누락/불일치/meta tensor.
핵심 메커니즘: 노드 종료 시 보통 랭크 프로세스도 함께 종료 → 핸들 정리. 설령 유지해도 프로세스 경계를 넘는 전달은 불가.
4) 권장 설계(충돌 없는 패턴)
**훈련 전체를 Kedro “한 노드”**로 묶고, 그 노드 내부에서 런처 1회 호출:
deepspeed --num_gpus=4 src/train/main_yaml.py --config_yml cfg.yml
Kedro는 SequentialRunner 사용(병렬러너 금지).
중간 메트릭/체크포인트/평가는 노드 내부에서 수행하고, rank0만 로깅.
노드 반환값은 직렬화 가능한 결과(최종 메트릭 요약, 체크포인트 경로 등)만.
5) “핸들(handle)”과 “직렬화” — 왜 못 넘기는가
핸들 = OS/드라이버가 관리하는 ‘열려 있는 연결’의 참조표(CUDA 컨텍스트/스트림, NCCL communicator, 통신 큐, cuBLAS 핸들 등).
→ 프로세스-귀속(process-bound): 해당 PID의 주소공간·커널 오브젝트에 매여 있어 복사·직렬화·이관 불가.
데이터(정적)와 대비
OK: CPU 텐서, 파라미터 state_dict()(단, ZeRO-3는 엔진 API로 병합/로드 필요)
NG: 엔진/그룹/커뮤니케이터 같은 라이브 핸들
6) DataLoader/분산 배치 메커니즘
DistributedSampler: 전체 인덱스를 랭크별로 균등 분할(에폭마다 셔플), 각 랭크는 자기 리스트만 순회.
DataLoader: 랭크별 인덱스를 배치 단위로 묶어 for batch in train_dl로 공급.
batch = {k: v.to(engine.local_rank)}: RAM→VRAM 전송(디바이스로 복사).
collate_fn: 샘플들을 텐서로 묶는 함수(명시 안 하면 기본 동작).
루프: out = engine(**batch) → loss → engine.backward(loss) → engine.step().
7) ZeRO-2/3, 통신·업데이트 흐름(숫자/개념)
전역 평균(gradient global average): 모든 랭크의 grad를 원소별 평균.
reduce-scatter: 평균을 샤드별로 소유 랭크에 직접 분배.
업데이트: 각 랭크는 자기 shard만 in-place 업데이트(Adam이면 m/v shard 갱신).
전파: 업데이트된 shard를 브로드캐스트/집합해 모든 랭크의 파라미터 복제본을 동일화.
accumulation 고려: 전역 배치 = per_device × WORLD_SIZE × accumulation.
메모리 직관(활성/버퍼 제외)
ZeRO-2: 대략 P + 3P/W (P=파라미터 용량)
예: P=16GB, W=4 → 16 + 3×4 = 28GB → 24GB VRAM이면 OOM 위험 → 옵티마이저 CPU 오프로딩 또는 ZeRO-3, QLoRA/accum↑/seq_len↓.
CPU 오프로딩 vs GPU-only
GPU-only: 빠름, VRAM 압박 큼
옵티마이저 오프로딩: VRAM 절약, 속도 손실(전송/CPU연산)
8) 어텐션/포지셔널 요약(입력 합성 vs 어텐션 스코어)
입력 합성: X = word_emb(input_ids) + pos_emb(positions) + (token_type_emb)
X ∈ ℝ^{B×L×d}
token_type_ids는 보통 0/1(문장쌍 구분), attention_mask는 0/1(마스킹)이며 X에 더하지 않음.
상대 위치(T5-RPB/ALiBi): **어텐션 스코어 L×L**에 거리별 바이어스 추가(입력 합성의 X 크기는 그대로).
RoPE: Q/K를 위치에 따라 회전 → 점수가 상대 거리(j−i) 에 의존(역시 X 크기 불변).
learned vs sinusoidal:
learned: P[t]가 학습 파라미터(위치별로 값 다름).
sinusoidal: 수식 기반(학습 안 함), 위치마다 다른 벡터 생성.
9) 성능/안정화 튜닝 레시피(즉효)
overlap_comm: true(통신↔연산 오버랩)
reduce_bucket_size / allgather_bucket_size 키워서 왕복 지연↓
전역 배치(= per_device × WORLD_SIZE × accumulation) ↑
bf16/FP16, activation checkpointing
토폴로지: NVLink/InfiniBand + NCCL_SOCKET_IFNAME, NCCL_IB_HCA
DataLoader: pin_memory=True, num_workers=k, .to(..., non_blocking=True)
drop_last=True로 스텝 불균형 방지
10) 환경변수 핵심 정의
WORLD_SIZE: 총 랭크 수(=총 GPU)
RANK: 전역 랭크 id(0..WORLD_SIZE−1)
LOCAL_RANK: 머신 내부 GPU id(0..nGPU−1)
MASTER_ADDR/MASTER_PORT: 랭크0 주소/포트
NCCL_*: 통신 인터페이스/튜닝(소켓/IB/버퍼 등)
11) MLflow/Hook 운용 원칙
훅은 노드 종료 시점에만 개입(after_node_run).
중간 메트릭/체크포인트 스트리밍은 노드 내부에서 직접 기록.
중복 방지: 반드시 rank0 가드로 로깅.
파일 드롭을 훅이 실시간 감시하진 않는다(노드 종료 후 집계는 가능).
12) 제공 코드(main_yaml.py)와 운영 포인트
YAML에서 모델/데이터/훈련/LoRA/DeepSpeed 설정 수용(DeepSpeed는 dict 직접 전달, JSON 불필요).
AutoTokenizer/AutoModelForCausalLM + LoRA(peft) + deepspeed.initialize 흐름.
DistributedSampler.set_epoch(ep), drop_last=True, pin_memory, num_workers.
학습/평가 루프에서 rank0만 메트릭/저장.
체크포인트: 특히 ZeRO-3는 engine.save_checkpoint(...) 사용. 필요 시 full-state export로 외부 평가/추론에 활용.
스케줄러(warmup)와 bf16/flash-attn 옵션도 YML로 제어.
13) “모순처럼 보였던 지점” 해소
SequentialRunner=한 프로세스는 맞다.
하지만 노드 내부에서 런처가 랭크 프로세스들을 추가로 스폰하고, 엔진은 그 랭크들 안에 산다.
노드가 끝나면 랭크들이 종료되며 핸들도 정리. 다음 노드로 엔진 전달 불가.
따라서 훈련=단일 노드, 런처 1회가 정석.
14) 최종 운영 체크리스트
Kedro SequentialRunner 고정.
훈련=단일 노드, 내부에서 deepspeed --num_gpus=4 ... 1회 호출.
중간 메트릭/체크포인트/평가 = 노드 내부, rank0 가드.
ZeRO 단계에 맞춘 정식 체크포인트 저장/복원(필요 시 full-state export).
I/O·통신·배치·버킷·오버랩 등 튜닝 파라미터는 YAML로 관리.



학습 (Training)
무엇을 기록?
train/loss (Trainer 자동)
train/ppl = exp(평균 loss) (LM)
lr (Trainer 자동 또는 optimizer.param_groups[0]["lr"])
speed/tokens_per_sec (스텝 시간/배치 토큰으로 계산)
어떻게 쓰나 (Transformers Trainer + MLflow)
# TrainingArguments(report_to=["mlflow"])로 기본 로깅 자동
import math, time, os, mlflow
from transformers import TrainerCallback
class SpeedCallback(TrainerCallback):
def on_step_begin(self, args, state, control, **kw):
self.t0 = time.perf_counter()
self.pad = kw["model"].config.pad_token_id if hasattr(kw["model"].config, "pad_token_id") else 0
def on_step_end(self, args, state, control, **kw):
if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) != 0: return
dt = time.perf_counter() - self.t0
batch = kw["inputs"]
tokens = (batch["input_ids"] != self.pad).sum().item()
mlflow.log_metric("speed/tokens_per_sec", tokens/dt, step=state.global_step)
# ppl은 평가 루프나 콜백에서: mlflow.log_metric("train/ppl", math.exp(avg_loss), step=global_step)
평가 (Evaluation)
무엇을 기록?
eval/loss, eval/ppl
태스크별: accuracy/f1/precision/recall/roc_auc/rouge/sacrebleu (Hugging Face evaluate 권장)
어떻게 쓰나 (evaluate + Trainer.compute_metrics)
import evaluate, math
acc = evaluate.load("accuracy"); f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
logits, labels = eval_pred
preds = logits.argmax(-1)
acc.add_batch(predictions=preds, references=labels)
f1.add_batch(predictions=preds, references=labels)
return {"eval/accuracy": acc.compute()["accuracy"],
"eval/f1": f1.compute()["f1"]} # 여기에 loss→ppl은 별도 콜백/평가 후 exp(loss)
DeepSpeed (구간별 시간/통신·FLOPs)
무엇을 기록?
스텝 벽시계 분해: time/forward, time/backward, time/optimizer, time/comm, time/step_total
처리량: speed/tokens_per_sec (사용자 콜백)
FLOPs/매개변수/지연: FLOPs 프로파일러 스냅샷
튜닝 해석 지표: time/comm 비중↓, gpu/max_alloc_mb 여유↑, 처리량↑
어떻게 쓰나
1) 벽시계 분해(설정만으로 로그 출력)
# deepspeed config
wall_clock_breakdown: true
overlap_comm: true
# reduce_bucket_size, allgather_bucket_size 등은 환경 맞춰 조정
2) FLOPs 프로파일러(워밍업 뒤 1~N 스텝만)
from deepspeed.profiling.flops_profiler import get_model_profile
macs, params, flops, duration = get_model_profile(
model, input_shape=(batch_size, seq_len),
as_string=False, print_profile=False, detailed=False
)
if is_rank0:
mlflow.log_metric("flops/tflops_per_step", flops/1e12, step=global_step)
mlflow.log_metric("params/total_m", params/1e6, step=global_step)
PyTorch (메모리/프로파일)
무엇을 기록?
메모리: gpu/alloc_mb, gpu/max_alloc_mb
프로파일(선택): 커널/연산 단위 성능(trace)
어떻게 쓰나
import torch, mlflow, os
def is_rank0(): return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0
if is_rank0():
alloc = torch.cuda.memory_allocated()/1e6
peak = torch.cuda.max_memory_allocated()/1e6
mlflow.log_metric("gpu/alloc_mb", alloc, step=global_step)
mlflow.log_metric("gpu/max_alloc_mb", peak, step=global_step)
# 선택: torch.profiler
# with torch.profiler.profile(activities=[...], record_shapes=True) as prof:
# ...
# prof.export_chrome_trace("trace.json") # MLflow artifact로 올려도 됨
운용 규칙(분산 공통)
항상 rank=0만 로깅/아티팩트 저장.
체크포인트 주기 저장(engine.save_checkpoint) + 재시작 시 load_checkpoint 후 step/tokens_per_sec 등 지표 이어가기.
Kedro→DeepSpeed 서브프로세스라면 부모 run_id/URI를 env로 전달하고, rank0에서 mlflow.start_run(run_id=..., nested=True)로 같은 run에 계속 기록.
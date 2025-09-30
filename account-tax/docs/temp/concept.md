# Core Concepts for Qwen Fine-Tuning & Kedro/MLflow Integration

## 1. Concept Tree Overview
```
AI Training Pipeline
├─ Data Preparation
│  ├─ Text Serialisation
│  └─ Tokenization
├─ Model Initialization
│  ├─ Base Model (Qwen)
│  └─ PEFT Adapter (LoRA)
└─ Training Dynamics
   ├─ Tensor / Module / Model hierarchy
   ├─ Autograd Graph & Memory Control
   └─ Logging & Tracking (Kedro ↔ MLflow)
```

---

## 2. Tokenizer: Role and Execution Flow

### Definition
Hugging Face 토크나이저는 한 번의 호출로 **토큰 단위 분해 → 정수 ID 매핑 → 길이 제한(잘라내기) → 패딩**까지 처리하는 전처리기입니다.

### Workflow
```
입력 텍스트 배치
  ↓ (DataLoader에서 collate_fn 호출)
Tokenizer(
    texts,
    truncation=True,
    max_length=L,
    padding="longest" or "max_length",
    return_tensors="pt"
)
  ↓
{input_ids, attention_mask, ...}  # 모델 입력 텐서
```
- `truncation=True`와 `max_length`를 지정하면 정수화된 토큰 시퀀스를 지정 길이로 자릅니다.
- 토크나이저는 직렬화된 텍스트를 배치 단위로 받아, 바로 모델이 사용할 수 있는 텐서를 반환합니다.
- DataLoader 또는 Trainer가 배치마다 토큰화를 호출하며, torch 텐서로 된 입력을 생성합니다.

---

## 3. Model Initialization: What and How

### Definition
모델 초기화는 **사전학습된 가중치를 로딩하고, 작업(Task)에 맞게 헤드와 설정을 조정**하는 단계입니다.

### Workflow
```
model_name = "Qwen/Qwen1.5-4B"

# 1. 토크나이저 로드
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 사전학습 모델 로드 + 분류 헤드 설정
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=NUM_CLASSES,
    problem_type="single_label_classification",
)

# 3. (선택) label 매핑, dropout 등 구성 수정
model.config.id2label = {...}
model.config.label2id = {...}

# 4. PEFT / LoRA 어댑터 적용 (LoRAConfig → get_peft_model)
```
- **Base 모델**은 20개의 Transformer 블록(`model.model.layers[0..19]`)으로 구성됩니다.
- 필요한 경우 아래와 같이 **인덱스로 특정 블록을 참조**하거나, 모듈 이름을 활용해 접근합니다.
  ```python
  block_0 = model.model.layers[0]
  block_last = model.model.layers[-1]

  # 이름으로 특정 서브모듈 찾기 (예: LoRA target)
  for name, module in model.named_modules():
      if "q_proj" in name:
          print("LoRA candidate:", name)
  ```

---

## 4. LoRA & Partial Fine-Tuning Concepts

### Core Idea
LoRA는 **원본 가중치 W를 그대로 두고, 저차원 행렬 A, B로 표현한 보정항 ΔW = B·A를 추가**해 미세조정합니다. 이는 GPU 메모리와 연산량을 크게 줄여 줍니다.

### Computation
```
Original forward : y = W x
LoRA forward     : y = W x + B (A x)    # A, B만 학습
```
- W는 freeze, A와 B만 `requires_grad=True`로 설정합니다.
- 역전파 시 A, B에 대해서만 gradient가 계산되어 업데이트됩니다.
- 추론 시에는 `W + B·A`를 합쳐 단일 가중치처럼 사용할 수도 있고, runtime에 덧셈을 유지해도 됩니다.

### Selective Layer Training
- Qwen 4B의 20개 블록 중 **예: 16~20번 블록에만 LoRA를 적용**하고 나머지는 완전히 freeze하면 메모리를 아낄 수 있습니다.
- 레이어 인덱싱 예시:
  ```python
  frozen_layers = model.model.layers[:15]
  train_layers = model.model.layers[15:]

  # 1) 파라미터에 requires_grad=False 플래그를 일괄 적용
  for block in frozen_layers:
      block.requires_grad_(False)  # 내부 텐서 플래그가 False로 바뀜

  # 2) Forward에서 불필요한 그래프 생성을 막기 위해 no_grad 컨텍스트 사용
  with torch.no_grad():
      for block in frozen_layers:
          hidden = block(hidden, attention_mask=mask)[0]

  # 3) 그래프를 완전히 끊고 뒷부분만 새 그래프로 시작
  hidden = hidden.detach()
  hidden.requires_grad_(True)

  # 4) 학습 대상 블록만 그래프에 포함
  for block in train_layers:
      hidden = block(hidden, attention_mask=mask)[0]
  ```

#### Qwen 3-4B Block Layout (요약)
```
Embedding & Rotary Positional Encoding
   ↓
Transformer Layer 0
Transformer Layer 1
...
Transformer Layer 18
Transformer Layer 19
   ↓
Normalization + Output Head (e.g., lm_head or classifier)
```
- `model.model.layers[i]`로 0~19 블록에 직접 접근할 수 있습니다.
- 블록 내부 주요 서브모듈 예시 (이름 패턴)
  - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`
  - `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- LoRA 타깃을 지정할 때는 위와 같은 이름을 활용해 특정 투영층에만 어댑터를 삽입합니다.

---

## 5. Tensor → Block → Model : Hierarchy Definitions

- **Tensor**: PyTorch `torch.Tensor` 혹은 `torch.nn.Parameter` 객체. 예) weight, bias, intermediate activation.
- **Block/Module**: 여러 텐서를 묶어 연산을 정의한 `torch.nn.Module`. 예) Transformer layer, FFN.
- **Model**: 여러 블록을 연결한 상위 구조. 예) Qwen 전체 네트워크 (`model.model.layers`, embedding, lm_head 등).

### Autograd Flags & Memory Control
| 도구 | 적용 범위 | 효과 |
|------|-----------|------|
| `requires_grad` | 개별 텐서 | True이면 gradient 계산; False면 gradient 및 optimizer 업데이트 대상에서 제외 |
| `torch.no_grad()` | 코드 블록 (여러 연산 포함) | Forward 계산은 수행하되, 그 연산을 그래프에 기록하지 않음 (activation 저장 X) |
| `tensor.detach()` | 특정 텐서 | 기존 그래프와 연결 끊고, 이후 연산을 새 그래프로 시작 |

> **Tip**: `module.requires_grad_(False)`는 모듈 내부의 모든 파라미터 텐서에 대해 `requires_grad=False`를 일괄 적용하는 **편의 함수**입니다. 즉, 실제로는 “블록 전체에 표시”해 보이지만, 결국 내부 개별 파라미터 텐서의 플래그가 바뀌는 것입니다.

#### Flag 조합이 만들어내는 인과 관계
```
Input
  │
  ├─ Frozen Block (requires_grad=False + no_grad)
  │      • forward 계산 → 그래프 기록 X → activation 저장 X
  │
  └─ detach() → hidden.requires_grad_(True)
         • 이전 블록과 그래프 연결 완전 차단
         • 이후 블록은 새로운 그래프로 시작

  └─ Trainable Block (requires_grad=True)
         • forward 계산 → 필요한 activation만 저장 → backward 시 gradient 계산

Loss.backward()
  • Frozen Block: 그래프가 없으므로 연산·메모리 0
  • Trainable Block: 그래프가 존재하므로 gradient 계산 후 업데이트
```
- `requires_grad=False`는 “이 텐서/파라미터는 gradient 대상이 아니다”라는 플래그.
- `torch.no_grad()`는 “이 구간의 forward를 그래프에 기록하지 말라”는 컨텍스트. forward 결과는 얻되 activation을 저장하지 않음.
- `detach()`는 “지금까지 만들어진 그래프와 완전히 단절”시켜 이후 연산을 새 그래프로 시작하게 만듭니다.
- 세 도구는 역할이 다르며, **조합해서** 사용하면 “앞 블록은 완전히 inference 모드, 뒤 블록만 학습” 구조를 만들 수 있습니다. 하나만 쓰면 원하는 효과를 얻기 어렵습니다.

#### 실제 그래프 비교 예 (TinyNet)
```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

net = TinyNet()
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# Helper: grad_fn 체인을 출력
from collections import deque
def print_graph(y):
    visited = set()
    queue = deque([(y.grad_fn, None)])
    while queue:
        node, parent = queue.popleft()
        if node is None or node in visited:
            continue
        visited.add(node)
        print(f"{node}  ← produced by {parent}")
        for next_fn, _ in node.next_functions:
            queue.append((next_fn, node))

# Case 1: 모든 레이어를 학습
out_full = net.fc2(torch.relu(net.fc1(x)))
print_graph(out_full)

# Case 2: fc1을 no_grad + detach → fc2만 학습
with torch.no_grad():
    h = torch.relu(net.fc1(x))
h = h.detach()
h.requires_grad_(True)
out_partial = net.fc2(h)
print_graph(out_partial)
```

출력에서 확인할 수 있듯:
- **Case 1**: `fc1`과 `fc2` 관련 `AddmmBackward`, `ReluBackward` 노드가 모두 등장 → 그래프가 전부 연결 → 두 레이어 모두 gradient 계산 & activation 저장.
- **Case 2**: `fc2` 관련 노드만 남고, `fc1`에 해당하는 노드는 사라짐 → 앞 레이어는 그래프에서 제외돼 backward/메모리가 필요 없다.

아래는 실제로 노드 이름을 나열해 본 결과입니다 (`TinyNet`에 bias=False를 두고 실행):

| 케이스 | 적용한 플래그/컨텍스트 | 그래프에 남은 노드(`grad_fn` 이름) |
|--------|------------------------|--------------------------------------|
| A | 아무 것도 적용 X | `['MmBackward0', 'ReluBackward0', 'TBackward0', 'MmBackward0', 'AccumulateGrad', ...]`<br>→ `fc1`·`relu`·`fc2`가 모두 그래프에 기록 |
| B | `fc1.weight.requires_grad=False` | `['MmBackward0', 'ReluBackward0', 'TBackward0', 'MmBackward0', ...]`<br>→ 파라미터는 업데이트 안 되지만, `fc1` 연산은 그래프에 남아 activation을 저장함 |
| C | `with torch.no_grad(): fc1(...)` + `detach()` | `['MmBackward0', 'AccumulateGrad', 'TBackward0', 'AccumulateGrad']`<br>→ `fc2` 관련 노드만 남아, `fc1` 부분은 그래프에서 완전히 사라짐 |

이처럼 **freeze만 하면 그래프는 여전히 남지만**, `no_grad`와 `detach`까지 적용해야 앞 구간이 진짜로 그래프에서 분리되어 메모리 사용이 0에 가깝게 줄어든다는 것을 눈으로 확인할 수 있습니다.

> **정리**
> - VRAM 절약을 위해서는 `torch.no_grad()`가 가장 결정적입니다. 이 컨텍스트 안에서 실행된 forward는 그래프에 기록되지 않아 activation이 생성되지 않습니다.
> - `requires_grad=False`만 적용하면 파라미터 업데이트는 막지만, 그래프와 activation은 계속 남습니다.
> - `detach()`는 이미 만들어진 그래프를 후속 구간과 분리해 backward에서 다시 추적하지 않게 하지만, 앞 구간의 저장 공간을 즉시 0으로 만들지는 않습니다. 그러나 `no_grad`와 함께 쓰면 그래프 자체를 만들지 않으므로 메모리 사용을 거의 없앨 수 있습니다.

##### 계산 그래프 노드가 생긴 모습은?
```python
import torch
import torch.nn as nn
from collections import deque

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

net = TinyNet()
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

out = net.fc2(torch.relu(net.fc1(x)))

def adjacency_list(fn):
    graph = {}
    queue = deque([fn])
    visited = set()
    while queue:
        node = queue.popleft()
        if node is None or node in visited:
            continue
        visited.add(node)
        graph[id(node)] = {
            "name": type(node).__name__,
            "children": []
        }
        for child, _ in node.next_functions:
            if child is not None:
                graph[id(node)]["children"].append(id(child))
                queue.append(child)
    return graph

print(adjacency_list(out.grad_fn))
```
출력 예시는 다음과 같습니다.
```
{
  139633967594752: {'name': 'MmBackward0', 'children': [139633967594800, 139633967592016]},
  139633967594800: {'name': 'ReluBackward0', 'children': [139633967589712]},
  139633967592016: {'name': 'TBackward0', 'children': [139633967589568]},
  139633967589712: {'name': 'MmBackward0', 'children': [139633967589472, 139633967585488]},
  ...
}
```
- 각 노드는 **연산 이름(`MmBackward0`, `ReluBackward0` 등)**과 **다음에 연결된 노드(children)**를 포함합니다.
- 이 관계망이 바로 autograd가 역전파 때 순회하는 “지도”이며, 노드는 메타정보만 갖고 있어서 메모리는 거의 차지하지 않습니다. 대신, 이 노드들이 가리키는 **텐서(activation)**가 더 큰 VRAM을 소비합니다.

##### 메모리 구성 관점 (Tensor vs. Graph Metadata)
- VRAM의 대부분은 **텐서(파라미터, activation)**가 차지합니다. 예를 들어 4,096×4,096 float16 텐서 하나만으로도 약 32MB가 필요합니다.
- Autograd 그래프의 노드/연산 정보는 포인터, 함수 레퍼런스 등 가벼운 메타데이터라서, 수백 개의 노드여도 메모리 사용량은 텐서와 비교하면 미미합니다.
- 따라서 메모리를 줄이고 싶다면 **activation이 생성되지 않도록(no_grad) 하거나, 필요 없는 텐서의 그래프 연결을 끊는(detach) 방식**이 가장 효과적입니다.

### Practical Pattern for Partial Fine-Tuning
```
with torch.no_grad():
    for block in model.model.layers[:15]:
        hidden = block(hidden, attention_mask=mask)[0]

hidden = hidden.detach()          # 앞 블록 activation 완전 제거
hidden.requires_grad_(True)       # 뒷 블록은 학습 대상

for block in model.model.layers[15:]:
    hidden = block(hidden, attention_mask=mask)[0]
```
- 앞 블록은 inference 모드처럼 메모리를 거의 쓰지 않으며, 뒷 블록만 그래프에 남아 OOM을 방지합니다.

---

## 6. Computational Graph: Definition & Memory Implication

### Definition
- Autograd 계산 그래프는 **“Tensor → Operation → Tensor”**로 연결된 **관계망**입니다.
- Forward pass에서 그래프가 만들어지고, backward pass에서 이를 역순으로 순회하여 gradient가 계산됩니다.

### Memory Note
- 그래프에 올라간 텐서는 역전파를 위해 activation을 VRAM에 저장합니다.
- `torch.no_grad()`나 `detach()`로 그래프에서 제외된 연산은 activation을 저장하지 않으므로 메모리를 아낄 수 있습니다.
- “관계망에 없다”는 것은 곧 **역전파·gradient 계산이 전혀 일어나지 않음**을 의미하고, 이는 메모리 절약으로 직결됩니다.

---

## 7. Kedro ↔ MLflow Logging Context (요약)
- 노드는 Kedro가 통제 가능한 최소 단위입니다. 노드 출력이 카탈로그에 연결되면 MLflow Dataset(Artifact/Metric)을 통해 자동으로 기록됩니다.
- MLflow run 구조는 다음과 같습니다:
  - `params/`: 데이터 split, 하이퍼파라미터 등의 설정값
  - `metrics/`: 로그된 수치(정확도, 손실, 데이터 품질)
  - `artifacts/`: 모델 가중치, 데이터셋, 리포트 파일 등
  - `tags/`: 파이프라인 이름, 실행 환경 등 메타데이터
  - `meta.yaml`: run 요약 정보

---

## 8. Torch vs. PyTorch 명칭
- **PyTorch**: 프로젝트/프레임워크 전체 이름.
- **`torch`**: PyTorch에서 사용하는 핵심 Python 패키지(`import torch`).
- `torchvision`, `torchaudio` 등은 도메인별 부가 패키지입니다.

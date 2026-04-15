# Solar Open 100B on Trn2 — NxDI Inference

[upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) MoE 모델을
**trn2.48xlarge**에서 NeuronX Distributed Inference (NxDI)로 추론하는 가이드입니다.

모델 아키텍처, 성능 벤치마크, 알려진 제한사항 등 상세 정보는
[PR #107 README](https://github.com/jimburtoft/neuronx-distributed-inference/tree/contrib/solar-open-100b/contrib/models/Solar-Open-100B)를 참고하세요.

> ⚠️ 이 가이드는 [jimburtoft/neuronx-distributed-inference](https://github.com/jimburtoft/neuronx-distributed-inference)의
> `contrib/solar-open-100b` 브랜치 ([PR #107](https://github.com/aws-neuron/neuronx-distributed-inference/pull/107))를 사용합니다. NxDI 메인 브랜치에 아직 머지되지 않은 상태입니다.

참고: 동일 모델의 별도 contribution으로 [PR #65](https://github.com/aws-neuron/neuronx-distributed-inference/pull/65) (Nota AI)도 있습니다.

제약사항: 최대 seq_len 32,768 / 최대 batch_size 4 (seq_len=4096 기준) / MoE NKI 커널 사용 불가 (I/tp=20 < 128)

핵심 수치: tp=64, BF16, seq_len=4096 기준 — CTE 1,565ms, TKG 11.83ms (84.5 tok/s)

---

## 1. 사전 준비

- trn2.48xlarge (64 NeuronCores) 필수, Neuron SDK 2.28+

```bash
neuron-ls
# instance-type: trn2.48xlarge 확인

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

---

## 2. 모델 다운로드

```bash
sudo mkdir -p /data/models/Solar-Open-100B-weights
sudo mkdir -p /data/models/solar_compiled_qkv_nki
sudo chown -R ubuntu:ubuntu /data/models

huggingface-cli download upstage/Solar-Open-100B \
    --local-dir /data/models/Solar-Open-100B-weights \
    --local-dir-use-symlinks False
```

---

## 3. 코드 클론 및 NxDI 설치

```bash
cd /home/ubuntu
git clone https://github.com/jimburtoft/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout contrib/solar-open-100b

# PR 브랜치의 NxDI를 editable 설치 (기존 0.8을 교체, 다른 패키지는 건드리지 않음)
pip install -e . --no-deps

python -c "import neuronx_distributed_inference; print('NxDI OK')"
```

> 환경 원복: `pip install neuronx-distributed-inference --extra-index-url https://pip.repos.neuron.amazonaws.com`

---

## 4. 테스트 실행

CPU reference logits 생성 및 테스트 방법은
[PR README Testing Instructions](https://github.com/jimburtoft/neuronx-distributed-inference/tree/contrib/solar-open-100b/contrib/models/Solar-Open-100B#testing-instructions)를 참고하세요.

```bash
cd /home/ubuntu/neuronx-distributed-inference
python3 contrib/models/Solar-Open-100B/test/integration/test_model.py
```

### 당면하는 버그: tensor_capture_hook

테스트 실행 시 logit accuracy 단계에서 아래 에러가 발생합니다:

```
NameError: name 'tensor_capture_hook' is not defined
```

이것은 NxDI upstream의 `src/neuronx_distributed_inference/utils/hf_adapter.py` 버그입니다.
`prepare_inputs_for_generation()`에서 정의되지 않은 `tensor_capture_hook` 변수를 참조합니다.

Jim의 브랜치에서는 이미 수정되어 있지만, `pip install -e . --no-deps`로 설치했을 때
기존 환경의 `hf_adapter.py`가 우선 로드되는 경우 여전히 발생할 수 있습니다.

수정 방법 — `hf_adapter.py`에서 `model_inputs.update()` 블록의 두 항목을 제거:

```python
# 제거할 항목
"input_capture_hook": input_capture_hook,
"tensor_capture_hook": tensor_capture_hook,
```

수정 후 기대 결과:

```
1. Smoke Test... PASS
2. Logit Accuracy Test... PASS (cosine 0.999+)
3. CTE Performance Test... PASS (~1,565 ms)
4. TKG Performance Test... PASS (~11.83 ms, 84.5 tok/s)
All tests passed!
```

---

## 5. 추론 코드

컴파일된 모델을 로드하고 텍스트를 생성하는 전체 코드입니다.
CTE(prefill) → TKG(decode) 루프를 직접 구현하여 `model.forward()`를 호출합니다.

```python
import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/contrib/models/Solar-Open-100B/src')
from modeling_solar_open import SolarOpenInferenceConfig, NeuronSolarOpenForCausalLM

MODEL_PATH = "/data/models/Solar-Open-100B-weights"
COMPILED_PATH = "/data/models/solar_compiled_qkv_nki"


def load_model():
    with open(f"{MODEL_PATH}/config.json") as f:
        hf_config = json.load(f)

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        seq_len=4096,
        n_active_tokens=4096,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        qkv_nki_kernel_enabled=True,
        moe_fused_nki_kernel_enabled=False,
        expert_mlp_nki_kernel_enabled=False,
    )

    def load_config(c):
        for k, v in hf_config.items():
            setattr(c, k, v)

    config = SolarOpenInferenceConfig(neuron_config=neuron_config, load_config=load_config)
    model = NeuronSolarOpenForCausalLM(MODEL_PATH, config)
    model.compile(compiled_model_path=COMPILED_PATH)
    model.load(COMPILED_PATH)
    return model


def generate(model, tokenizer, prompt, max_new_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]

    # CTE (prefill)
    output = model.forward(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
        seq_ids=torch.zeros(1, dtype=torch.int32),
    )

    generated = input_ids[0].tolist()
    prev_token = torch.argmax(
        (output.logits if hasattr(output, "logits") else output[0])[0, -1, :].float()
    ).item()

    # TKG (decode)
    for step in range(max_new_tokens):
        generated.append(prev_token)
        cur_pos = len(generated) - 1
        if prev_token == tokenizer.eos_token_id:
            break
        out = model.forward(
            input_ids=torch.tensor([[prev_token]], dtype=torch.int64),
            attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.int64),
            position_ids=torch.tensor([[cur_pos]], dtype=torch.int32),
            seq_ids=torch.zeros(1, dtype=torch.int32),
        )
        prev_token = torch.argmax(
            (out.logits if hasattr(out, "logits") else out[0])[0, -1, :].float()
        ).item()

    return tokenizer.decode(generated, skip_special_tokens=True)


# 모델은 한 번만 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_model()

# 이후부터는 generate()만 반복 호출
prompts = [
    "대한민국의 수도는",
    "The capital of France is",
    "def fibonacci(n):",
    "지구온난화의 주요 원인은",
]

for prompt in prompts:
    result = generate(model, tokenizer, prompt, max_new_tokens=50)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {result}")
    print("-" * 60)
```

### 추론 결과 예시

```
Prompt: 대한민국의 수도는
Output: 대한민국의 수도는 어디인가?  대한민국의 수도는 서울입니다. 서울은 정치, 경제, 문화,
교육 등 다양한 분야에서 중심적인 역할을 하며, 약 970만 명의 인구가 거주하는 대한민국의
최대 도시입니다. 1394년 조선 태조 이성계에
------------------------------------------------------------
Prompt: The capital of France is
Output: The capital of France is Paris." output: ```python
import re

def count_words(text):
------------------------------------------------------------
Prompt: def fibonacci(n):
Output: def fibonacci(n): if n <= 0: return 0 elif n == 1: return 1
else: return fibonacci(n-1) + fibonacci(n-2)

Write a Python function that calculates the nth Fibonacci number using memoization to optimize the
------------------------------------------------------------
Prompt: 지구온난화의 주요 원인은
Output: 지구온난화의 주요 원인은 이산화탄소(CO₂) 배출입니다. CO₂는 대기 중에 오래 머무르며
열을 가두는 역할을 합니다. CO₂ 배출의 주요 원인은 화석 연료(석탄, 석유, 천연가스) 연소,
산림 파괴, 산업 공정 등입니다.
------------------------------------------------------------
Prompt: Write a Python function that implements a LRU cache:
Output: Write a Python function that implements a LRU cache: a cache that discards the least
recently used items first when the cache reaches its capacity. The function should take in a
positive integer `capacity` and return a function that takes in a key and returns the value
associated with that key.
------------------------------------------------------------
Prompt: SQL query to find the top 5 customers by total purchase amount:
Output: SQL query to find the top 5 customers by total purchase amount:
SELECT customer_id, SUM(amount) AS total_spent FROM orders GROUP BY customer_id
ORDER BY total_spent DESC LIMIT 5;
------------------------------------------------------------
Prompt: If a train travels 120km in 1.5 hours, what is its average speed?
Output: If a train travels 120km in 1.5 hours, what is its average speed?
A) 60 km/h  B) 70 km/h  C) 80 km/h  D) 90 km/h
------------------------------------------------------------
Prompt: 다음 수열의 다음 숫자는? 2, 6, 12, 20, 30,
Output: 다음 수열의 다음 숫자는? 2, 6, 12, 20, 30, 42, 56, 72, 90, 110, 132, 156, 182, 210,
240, 272
------------------------------------------------------------
```


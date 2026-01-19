# AWS Trainium2 기반 Llama 3 8B Fine-tuning

이 가이드는 **AWS Trainium2 (Trn2)** 인스턴스에서 **LoRA (Low-Rank Adaptation)** 및 **Tensor Parallelism (텐서 병렬화)**을 사용하여 [**Llama 3 8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 모델을 파인튜닝하는 단계별 가이드를 제공합니다. 또한 Neuron 기반 **vLLM**을 사용하여 파인튜닝된 모델을 서빙하는 방법도 다룹니다.

## Prerequisites

1.  **인스턴스 실행:**
    * `trn2.48xlarge` 인스턴스가 활성화(`Running`) 상태여야 합니다.
    * 👉 **[가이드: Capacity Block 기반 XC 인스턴스 실행](https://github.com/leesjpe/compute-foundation-on-aws/blob/main/ec2/ec2-dlami-neuron.md)** 
2.  **고속 스토리지 설정(선택 사항) :**
    * 모델 로딩 속도와 체크포인트 저장 속도를 높이기 위해 로컬 NVMe SSD (RAID 0) 사용을 권장합니다.
    * 아직 설정하지 않으셨다면, 아래 가이드를 먼저 진행해 주세요.
    * 👉 **[가이드: 고속 스토리지 설정 (NVMe RAID 0)](https://github.com/leesjpe/compute-foundation-on-aws/blob/main/storage/local-nvme-setup.md)**
    * *참고: 이 과정을 건너뛴다면, 루트 EBS 볼륨에 모델을 저장할 충분한 공간이 있는지 확인하세요.*

---

## 1. 환경 설정 (Environment Setup)

### 1.1. Neuron 가상환경 활성화

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_training/bin/activate

```

### 1.2. 학습 스크립트 다운로드

`aws-neuron/neuronx-distributed` 리포지토리에서 필요한 학습 및 유틸리티 스크립트를 다운로드합니다.

```bash
# 실험을 위한 디렉토리 생성
mkdir -p /data/tp_llama3_8b_lora_finetune
cd /data/tp_llama3_8b_lora_finetune

# 스크립트 다운로드
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/data_module.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/module_llama.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/tp_llama_hf_finetune_ptl.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lr.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/modeling_llama_nxd.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements.txt
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements_ptl.txt
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/training_utils.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/convert_checkpoints.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/test/integration/modules/lora/test_llama_lora_finetune.sh
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

```

### 1.3. 패키지 설치

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_ptl.txt

# 쉘 스크립트 실행 권한 부여
chmod +x test_llama_lora_finetune.sh
```

### 1.4 NLTK 라이브러리 설치 및 punkt 다운로드
* NLTK(Natural Language Toolkit) 라이브러리가 텍스트를 "문장 단위로 쪼개기(Sentence Splitting)" 위해 필요한 학습된 모델 데이터를 다운로드.
```
pip install nltk
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');"

```
다운로드가 완료되면 아래와 같은 경로와 구조 확인 가능
```bash
/home/ubuntu/nltk_data/  <-- NLTK data 디렉터리
└── tokenizers/          <-- 토크나이저 모델 디렉터리
    └── punkt/           <-- 다운받은 모델 디렉터리
        ├── english.pickle
        └── ...
```

---

## 2. 모델 및 데이터셋 준비 (Prepare Model & Dataset)

### 2.1. Llama-3-8B 체크포인트 다운로드 (Hugging Face)

Base 모델을 로컬 디렉토리로 다운로드합니다.

```bash
hf auth login
# ./models/llama3-8b 경로에 다운로드
hf download meta-llama/Meta-Llama-3-8B --local-dir /data/models/hf-llama3-8b-bf16

```
허깅페이스에서 다운로드 하였기에 체크포인트 변환은 생략 하지만 만약 Meta 형식으로 다운로드 하였다면 다운로드 받은 * [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) 로 아래와 같이 HF 형식으로 변경 필요
```bash
pip install blobfile tiktoken
cd /data/tp_llama3_8b_lora_finetune
python convert_llama_weights_to_hf.py --input_dir /data/models/hf-llama3-8b-bf16/ --model_size 8B --llama_version 3 --output_dir /data/models/conv_hf-llama3-8b-bf16
```

### 2.2. 체크포인트를 Neuron 포맷(NXD)으로 변환

변환 과정 중 메모리 효율성을 높이기 위해, `AutoModelForCausalLM`을 사용하여 모델을 직접 로드하는 커스텀 변환 스크립트를 사용합니다.

1. **`convert_checkpoints.py` 생성/수정:**
[기존 스크립트](https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama/convert_checkpoints.py)를 아래의 최적화된 코드로 덮어씁니다.
기존 다운로드 받은 스크립트는 .bin 파일을 기준으로 작성되었으나 현재 HF 는 model.safetensors 포맷이기에 아래 스크립트로 변경하여 Hugging Face(.safetensors)를 **메모리(RAM)**에 로드하자마자 → 즉시 Neuron 포맷으로 변환해서 저장 함.

```python
import argparse
import torch
from transformers import AutoModelForCausalLM
from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase

class CheckpointConverterLlama(CheckpointConverterBase):
    def load_full_state(self, args):
        print(f"Loading model directly from {args.input_dir} using Transformers...")
        # 메모리 효율적으로 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            args.input_dir, 
            torch_dtype="auto", 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        return model.state_dict()

if __name__ == "__main__":
    checkpoint_converter = CheckpointConverterLlama()
    parser = checkpoint_converter.get_arg_parser()
    args, _ = parser.parse_known_args()
    checkpoint_converter.run(args)
```


2. **변환 실행:**
Hugging Face 포맷을 Neuron Distributed (Megatron 스타일) 포맷으로 변환합니다.
Hugging Face Transformers가 모델을 메모리에 효율적으로 로드하기 위해 사용하는 accelerate 라이브러리를 먼저 설치 합니다.

```bash
pip install accelerate

python3 convert_checkpoints.py \
--hw_backend trn2 \
--tp_size 32 \
--qkv_linear 1 \
--kv_size_multiplier 4 \
--convert_from_full_state \
--config config.json \
--input_dir /data/models/hf-llama3-8b-bf16 \
--output_dir /data/models/llama3_8b_bf16-tp32/pretrained_weight/

```

* `--tp_size 32`: 타겟 Tensor Parallelism 크기 (Trn2 노드 사양에 맞춤).
* `--hw_backend trn2`: 타겟 하드웨어 설정.
* `--qkv_linear`: GQA(Grouped-Query Attentioin) 모델은 1, Non GQA 모델은 0

[실행결과]
<img width="1107" height="630" alt="Screenshot 2026-01-16 at 3 56 55 PM" src="https://github.com/user-attachments/assets/505bbe66-4282-4cd5-9b34-77bced621c41" />


---

## 3. LoRA 파인튜닝 (Fine-tuning with LoRA)

### 3.1. 학습 스크립트 설정

`test_llama_lora_finetune.sh` 파일을 열어 경로 및 학습 파라미터를 수정합니다.

```bash
# 경로 설정
PRETRAINED_PATH=/data/models/llama3_8b_bf16-tp32
BASE_MODEL=/data/models/hf-llama3-8b-bf16
HF_TOKEN='your_token_here'

# 학습 파라미터 (Full Fine-tuning)
# 전체 Epoch를 돌리기 위해 step 제한을 해제(-1)합니다.
TOTAL_STEPS=-1 
TOTAL_EPOCHS=3
```
> **주의:** 스크립트 내에 `max_train_samples` 옵션이 있다면 주석 처리하거나 삭제하여 전체 데이터셋을 학습하도록 해야 합니다.

### 3.2. 학습 데이터
이 예시에서는 InstructGPT 논문에서 설명된 범주(브레인스토밍, 분류, 폐쇄형 질의응답, 생성, 정보 추출, 개방형 질의응답 및 요약 포함)에 대한 지시 따르기 기록으로 구성된 오픈 소스 데이터셋인 Dolly를 사용합니다.
데이터셋을 설정하려면 test_llama_lora_finetune.sh 파일에서 다음 플래그들을 구성합니다. 

```bash
--data_dir "databricks/databricks-dolly-15k" \
--task "open_qa" \
```

### 3.3. 학습 시작

```bash
./test_llama_lora_finetune.sh
```

**완료 확인:**

* 로그 메시지: `Training finished!`
* 로그 메시지: `synced saving of checkpoint lora completed`

Trn2.32xlarge 의 경우 수분내로 마무리 됩니다. 

### 3.4. 결과 확인

LoRA 어댑터는 출력 디렉토리에 저장됩니다. NXD는 체크포인트를 분할된(sharded) `.pt` 파일들로 저장합니다.

```bash
lora_adapter/
├── adapter_config.json
└── lora/model/
    ├── dp_rank_00_tp_rank_00_pp_rank_00.pt
    ... (TP=32인 경우 32개 파일)
```

---

## 4. vLLM을 이용한 추론 (Inference with vLLM)

### 4.1. vLLM용 LoRA 어댑터 준비 (중요)

Neuron 기반 vLLM은 분할된 `.pt` 파일들이 `lora/model`과 같은 하위 폴더가 아닌, 어댑터 디렉토리 최상위에 위치 해야 합니다.

1. **체크포인트 파일 이동:**
```bash
cd /home/ubuntu/tp_llama3_8b_lora_finetune/lora_adapter/lora/model
cp *.pt ../../
```

*이제 `adapter_config.json`과 `*.pt` 파일들이 같은 위치에 있어야 합니다.*

2. **`adapter_config.json` 수정:**

Neuron vLLM은 Q, K, V 레이어를 물리적으로 분리하여 처리합니다. 따라서 `target_modules` 이름을 이에 맞춰 수정해야 합니다.

**변경 전:**
```json
"target_modules": ["qkv_proj"],
```

**변경 후:**
```json
"target_modules": ["q_proj", "k_proj", "v_proj"],
```

3. 매핑 파일 생성 (lora_serving_config.json)
아래 내용을 담은 json 파일을 생성

```bash
vi /data/tp_llama3_8b_lora_finetune/lora_serving_config.json

{
  "lora-ckpt-paths": {
    "llama3_adapter": "/data/tp_llama3_8b_lora_finetune/lora_adapter"
  },
  "lora-ckpt-paths-cpu": {}
}
```
* 설명: "llama3_adapter"는 사용자가 지정 이름, 뒤에는 .pt 파일들이 들어있는 실제 폴더 경로를 지정.


### 4.2.1 vllm 환경 사용
3번까지 단계에서 사용한 가상환경이 Enable 되어있다면 deactivate 후 사전 구성된 vllm 환경 활성화
```bash
deactivate
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate

```

### 4.2.2 vLLM 도커 컨테이너 실행

* vllm 환경을 위한 Docker container 실행

```bash
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:<image_tag>
# neuron 2.27 vllm 0.11.0 기준 tag
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04

docker run \
-d -it \
-v /home/ubuntu/:/home/ubuntu/ \
-v /data:/data \
--privileged \
--cap-add SYS_ADMIN \
--cap-add IPC_LOCK \
-p 8000:8000 \
--name <server name> \
<Image ID>
```

### 4.3. 추론 TEST

* **LoRA 어댑터를 적용하여 모델을 서빙하는 Python 스크립트를 작성합니다. (`test_lora_inference.py`)**

```python
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# 사용자 환경 경로 반영
# ==========================================
MODEL_PATH = "/data/models/llama3-8b"
LORA_CKPT_JSON = "/data/tp_llama3_8b_lora_finetune/lora_serving_config.json"
COMPILED_MODEL_PATH = "/data/cache/llama3-8b-lora-finetuned-neuron_cache"

# 환경 변수 설정
os.environ["NEURON_COMPILED_ARTIFACTS"] = COMPILED_MODEL_PATH
os.environ["VLLM_USE_V1"] = "1"

# Sample prompts.
prompts = [
    "The president of the United States is"
]

# Create a sampling params object.
sampling_params = SamplingParams(top_k=1, max_tokens=4096)

# override_neuron_config 구조
override_neuron_config = {
    "skip_warmup": True,
    "lora_ckpt_json": LORA_CKPT_JSON,
}

# Create an LLM with multi-LoRA serving.
# additional_config 포함한 초기화 코드
llm = LLM(
    model=MODEL_PATH,
    max_num_seqs=2,
    max_model_len=4096,           # 64는 너무 짧아 4096으로 수정함 (안정성 위함)
    tensor_parallel_size=32,
    additional_config={
        "override_neuron_config": override_neuron_config
    },
    enable_lora=True,
    max_loras=2,
    max_cpu_loras=4,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)

"""
Only the lora_name needs to be specified.
The lora_id and lora_path are supplied at the LLM class/server initialization, after which the paths are
handled by NxD Inference.
"""

# lora_id_1 is in HBM (Defined in JSON)
lora_req_1 = LoRARequest("lora_id_1", 1, lora_path="/home/ubuntu/tp_llama3_8b_lora_finetune/lora_adapters/llama3_8b_lora") # Path is empty as per JSON usage
# lora_id_3 is in host memory (Defined in JSON)
#lora_req_2 = LoRARequest("lora_id_3", 2, lora_path="") # Path is empty as per JSON usage

#outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1, lora_req_2])
outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1])

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```


### 4.4. 트러블슈팅 체크리스트

* **답변이 중간에 잘리나요?** `SamplingParams`의 `max_tokens` 값을 늘려보세요.
* **결과가 Base 모델과 똑같나요?**
* 여러 번 학습했다면 `checkpoint-xxx` 폴더가 새로 생성되었는지 확인하고 경로를 최신으로 변경하세요.
* `adapter_config.json`의 `target_modules`가 `["q_proj", "k_proj", "v_proj"]`로 정확히 수정되었는지 확인하세요.
---

## Additional: Offline inference

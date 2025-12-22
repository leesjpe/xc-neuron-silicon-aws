작성해주신 영문 매뉴얼을 바탕으로, 한국 개발자들이 이해하기 쉽도록 기술 용어를 다듬어 한글 버전으로 번역했습니다.

Github `README.md` 등에 그대로 복사해서 사용하시면 됩니다.

---

# AWS Trainium2 기반 Llama 3 8B 파인튜닝 및 vLLM 서빙 가이드

이 리포지토리는 **AWS Trainium2 (Trn2)** 인스턴스에서 **LoRA (Low-Rank Adaptation)** 및 **Tensor Parallelism (텐서 병렬화)**을 사용하여 **Llama 3 8B** 모델을 파인튜닝하는 단계별 가이드를 제공합니다. 또한 Neuron 기반 **vLLM**을 사용하여 파인튜닝된 모델을 서빙하는 방법도 다룹니다.

## Prerequisites

* **인스턴스:** AWS EC2 `trn2.48xlarge` (또는 유사한 Trn2 인스턴스)
* **AMI:** AWS Deep Learning AMI (Neuron)
* **소프트웨어:** Neuron SDK 설치 완료 (PyTorch 환경)

---

## 1. 환경 설정 (Environment Setup)

### 1.1. Neuron 가상환경 활성화

```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/activate

```

### 1.2. 학습 스크립트 다운로드

`aws-neuron/neuronx-distributed` 리포지토리에서 필요한 학습 및 유틸리티 스크립트를 다운로드합니다.

```bash
# 실험을 위한 디렉토리 생성
mkdir -p ~/examples/tp_llama3_8b_lora_finetune
cd ~/examples/tp_llama3_8b_lora_finetune

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
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/test/integration/modules/lora/test_llama_lora_finetune.sh

```

### 1.3. 패키지 설치

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_ptl.txt
python3 -m pip install nltk blobfile tiktoken "huggingface_hub<1.0"

# 쉘 스크립트 실행 권한 부여
chmod +x test_llama_lora_finetune.sh

# NLTK 데이터 준비
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');"

```

---

## 2. 모델 및 데이터셋 준비 (Prepare Model & Dataset)

### 2.1. Llama-3-8B 체크포인트 다운로드 (Hugging Face)

Base 모델을 로컬 디렉토리로 다운로드합니다.

```bash
huggingface-cli login
# ./models/llama3-8b 경로에 다운로드
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir /home/ubuntu/models/llama3-8b

```

### 2.2. 체크포인트를 Neuron 포맷(NXD)으로 변환

변환 과정 중 메모리 효율성을 높이기 위해, `AutoModelForCausalLM`을 사용하여 모델을 직접 로드하는 커스텀 변환 스크립트를 사용합니다.

1. **`convert_checkpoints.py` 생성/수정:**
기존 스크립트를 아래의 최적화된 코드로 덮어씁니다.
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
```bash
python3 convert_checkpoints.py \
  --hw_backend trn2 \
  --tp_size 32 \
  --qkv_linear 1 \
  --kv_size_multiplier 4 \
  --convert_from_full_state \
  --config config.json \
  --input_dir /home/ubuntu/models/llama3-8b \
  --output_dir /home/ubuntu/models/llama3-8b-nxdt-tp32/pretrained_weight/

```


* `--tp_size 32`: 타겟 Tensor Parallelism 크기 (Trn2 노드 사양에 맞춤).
* `--hw_backend trn2`: 타겟 하드웨어 설정.



---

## 3. LoRA 파인튜닝 (Fine-tuning with LoRA)

### 3.1. 학습 스크립트 설정

`test_llama_lora_finetune.sh` 파일을 열어 경로 및 학습 파라미터를 수정합니다.

```bash
# 경로 설정
PRETRAINED_PATH=/home/ubuntu/models/llama3-8b-nxdt-tp32/
BASE_MODEL=/home/ubuntu/models/llama3-8b
HF_TOKEN='your_token_here'

# 학습 파라미터 (Full Fine-tuning)
# 전체 Epoch를 돌리기 위해 step 제한을 해제(-1)합니다.
TOTAL_STEPS=-1 
TOTAL_EPOCHS=3

```

> **주의:** 스크립트 내에 `max_train_samples` 옵션이 있다면 주석 처리하거나 삭제하여 전체 데이터셋을 학습하도록 해야 합니다.

### 3.2. 학습 시작

```bash
./test_llama_lora_finetune.sh

```

**완료 확인:**

* 로그 메시지: `Training finished!`
* 로그 메시지: `synced saving of checkpoint lora completed`

### 3.3. 결과 확인

LoRA 어댑터는 출력 디렉토리에 저장됩니다. NXD는 체크포인트를 분할된(sharded) `.pt` 파일들로 저장합니다.

```text
lora_adapter/
├── adapter_config.json
└── lora/model/
    ├── dp_rank_00_tp_rank_00_pp_rank_00.pt
    ... (TP=32인 경우 32개 파일)

```

---

## 4. vLLM을 이용한 추론 (Inference with vLLM)

### 4.1. vLLM 도커 컨테이너 실행

Neuron 디바이스 접근 권한을 포함하여 vLLM 컨테이너를 실행합니다.

```bash
docker run -d -it --privileged --shm-size=32g \
  -v /home/ubuntu/:/home/ubuntu/ \
  -v /dev:/dev \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK \
  -p 8000:8000 \
  --name vllm_llama8 \
  vllm-neuron:latest  # 사용 중인 이미지 ID 또는 태그로 교체

```

### 4.2. vLLM용 LoRA 어댑터 준비 (중요)

Neuron 기반 vLLM은 분할된 `.pt` 파일들이 `lora/model`과 같은 하위 폴더가 아닌, 어댑터 디렉토리 최상위에 위치하기를 기대합니다.

1. **체크포인트 파일 이동:**
```bash
cd /home/ubuntu/lora_adapters/llama3-8b-dolly-lora/
mv lora/model/*.pt .
rm -rf lora

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



### 4.3. 추론 스크립트 실행 (`neuron_multi_lora.py`)

LoRA 어댑터를 적용하여 모델을 서빙하는 Python 스크립트를 작성합니다.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 경로 설정
MODEL_PATH = "/home/ubuntu/models/llama3-8b/"
LORA_PATH_1 = "/home/ubuntu/lora_adapters/llama3-8b-dolly-lora/" # .pt 파일들과 config가 있는 경로

# 프롬프트 (Instruction 포맷 준수)
prompts = ["""### Instruction:
Write a romantic love letter about a potato.

### Response:
"""]

# 샘플링 파라미터
sampling_params = SamplingParams(top_k=1, max_tokens=1024)

# LLM 초기화 (Multi-LoRA)
llm = LLM(
    model=MODEL_PATH,
    max_num_seqs=2,
    max_model_len=4096,
    tensor_parallel_size=32,
    device="neuron",
    override_neuron_config={
        "sequence_parallel_enabled": False,
        "lora_modules": {"lora_id_1": LORA_PATH_1}, # 정적(Static) 로딩
    },
    enable_lora=True,
    max_loras=2,
)

# 생성 요청
lora_req_1 = LoRARequest("lora_id_1", 1, LORA_PATH_1)
outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1])

# 결과 출력
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

```

### 4.4. 트러블슈팅 체크리스트

* **답변이 중간에 잘리나요?** `SamplingParams`의 `max_tokens` 값을 늘려보세요.
* **결과가 Base 모델과 똑같나요?**
* 여러 번 학습했다면 `checkpoint-xxx` 폴더가 새로 생성되었는지 확인하고 경로를 최신으로 변경하세요.
* `adapter_config.json`의 `target_modules`가 `["q_proj", "k_proj", "v_proj"]`로 정확히 수정되었는지 확인하세요.

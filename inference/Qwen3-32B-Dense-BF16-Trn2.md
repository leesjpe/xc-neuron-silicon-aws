# Serving Qwen3 30B BF16 on Trainium2 with vLLM (NxD)

이 가이드는 **AWS Trainium2 (`trn2.48xlarge`)** 인스턴스에서 **vLLM**과 **Neuronx Distributed (NxD)** 아키텍처를 사용하여 **Qwen 3 (32B)** 모델을 서빙하는 방법을 설명합니다.
aws neuron 공식문서의 [Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/sd-inference-tutorial.html) 을 기반으로 구성 되었습니다. 

**Quickstart using Docker** vLLM이 사전 설치된 AWS Neuron 포크 버전의 사전 구성된 딥 러닝 컨테이너(DLC)를 활용합니다. 

* aws-neuron Github 의 [deep-learning-contianers](https://github.com/aws-neuron/deep-learning-containers?tab=readme-ov-file#vllm-inference-neuronx) 의 vllm-inference-neuronx 에서 컨테이너 별 vLLM Framework 버전, Neuron SDK 버전, ECR Public URL을 확인 할 수 있습니다.

* vLLM V1 방식을 따르며 [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron) 기반으로 vLLM 서버를 배포합니다. ([Neuron SDK 2.27](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.27.0/index.html) 버전 이상부터 적용 가능)

* 기존 vLLM V0 방식은 Neuron SDK 2.28 에서 Deprecated 될 예정입니다. [Inference update](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html#id6)에서 공지. 
V0를 사용하고 있다면 V1 으로의 마이그레이션을 고려하시길 권장합니다.  

---
### 📋 Prerequisites (사전 준비)

진행하기 전에 다음 사항들을 확인하세요.

1.  **인스턴스 실행:** `trn2.48xlarge` 인스턴스가 활성화(`Running`) 상태여야 합니다.
    * 👉 **[가이드: Capacity Block 기반 XC 인스턴스 실행](https://github.com/leesjpe/compute-foundation-on-aws/blob/main/ec2/ec2-dlami-neuron.md)** 
3.  **(선택 사항이지만 권장) 고속 스토리지 설정:**
    * 모델 로딩 속도와 체크포인트 저장 속도를 높이기 위해 로컬 NVMe SSD (RAID 0) 사용을 강력히 권장합니다.
    * 아직 설정하지 않으셨다면, 아래 가이드를 먼저 진행해 주세요.
    * 👉 **[가이드: 고속 스토리지 설정 (NVMe RAID 0)](https://github.com/leesjpe/compute-foundation-on-aws/blob/main/storage/local-nvme-setup.md)**
    * *참고: 이 과정을 건너뛴다면, 루트 EBS 볼륨에 모델을 저장할 충분한 공간이 있는지 확인하세요.*

---

### 1. 🐳 Docker 기반 vLLM 서버 배포
#### Step 1-1: Neuron vLLM 컨테이너 실행

라이브러리 의존성 충돌을 방지하기 위해 AWS 공식 **Neuron Deep Learning Container (DLC)**를 사용합니다.

아래 명령어를 사용하여 컨테이너를 실행하고 내부 쉘로 진입합니다.
*(고속 스토리지를 설정했다면 `/data` 마운트가 가능하며, EBS만 사용한다면 경로는 상황에 맞게 조정하세요.)*

```bash
# Docker 실행 및 진입
# -v /data:/data : 고속 스토리지(또는 모델 경로) 마운트
# -p 8000:8000 : API 서버 포트 개방

docker run -d -it \
  --privileged \
  -v /home/ubuntu/:/home/ubuntu/ \
  -v /data:/data \
  -p 8000:8000 \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04
```

#### 📥 Step 1-2: 가상환경 진입

```bash
docker exec -it <Container ID> bash
```

#### ⚙️ Step 1-3 vllm server 실행

1-3: Model download

```bash
mkdir /home/ubuntu/qwen3_32b_dense_bf16
hf download Qwen/Qwen3-32B --local-dir /home/ubuntu/qwen3_32b_dense_bf16
```

1-3 는 BF16 기준의 [Qwen3 32B Model](https://huggingface.co/Qwen/Qwen3-32B) 추론을 위한 vllm 서버를 실행 합니다. 

1-3 과정은  10~15 소요되며 아래와 같이 로그가 보이면 컴파일 및 서버 시작 완료 ☕️

기존 vLLM V0 방식에서는 VLLM_USE_V1=0 를 사용했지만 V1 에서는 제외 합니다.

```bash
# 환경 변수 설정
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/qwen3_32b_dense_bf16_artifacts"
export MODEL_ID="/home/ubuntu/qwen3_32b_dense_bf16"

# 서버 실행 (8000 포트 통한 외부 접속 허용, 특정 IP 로 제한 권장)
vllm serve $MODEL_ID \
    --tensor-parallel-size 16 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --block-size 32 \
    --host 0.0.0.0 \
    --port 8000
```
<img width="1294" height="545" alt="Screenshot 2025-12-06 at 9 17 48 PM" src="https://github.com/user-attachments/assets/4cf45802-3e9a-4290-b0c0-e5303f384e40" />

vLLM Model ID 확인
```bash
curl http://localhost:8000/v1/models
```

#### 🧪 Step 1-4: 추론 테스트 (Inference)

**Host Machine**에서 컨테이너에서 실행 중인 vLLM 서버로의 추론 테스트를 수행합니다.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
"model": "/home/ubuntu/qwen3_32b_dense_bf16",
"prompt": "What is machine learning?",
"max_tokens": 100,
"temperature": 0.7
}'
```
<img width="1295" height="216" alt="Screenshot 2025-12-06 at 10 38 58 PM" src="https://github.com/user-attachments/assets/451cf358-9bc8-45ab-bc77-fb491cb57a6d" />

#### 📊 Step 1-5: Performance Benchmarking (via Host)

이 가이드에서는 **Host Machine**에서 vLLM 서버를 벤치마킹하는 방법을 설명합니다. 호스트에서 벤치마크를 실행하면 Python 버전 충돌을 방지하고 리소스 분리를 ​​보장할 수 있습니다.

##### Prerequisites
* `llmperf` 라이브러리는 **Python 3.8 ~ 3.10** 환경을 권장합니다. 시스템 기본 Python 버전이 너무 높거나(3.11+), 패키지가 꼬이는 것을 방지하기 위해 `conda` 가상환경 사용을 권장합니다.

본 가이드는 llmperf 를 위한 별도 가상환경 사용하지 않고 사전 인스톨된 neuron 의 가상환경을 사용할 경우 llmperf 를 설치하는 방법을 안내합니다. 

```bash
# 1. llmperf 소스 코드 다운로드 (git clone)
git clone https://github.com/ray-project/llmperf.git
cd llmperf

# 2. pyproject.toml 파일에서 Python 버전 제한(<3.11)을 <3.13으로 수정 (sed 명령어 사용)
sed -i 's/<3.11/<3.13/g' pyproject.toml

# 3. 수정된 소스로 설치 (-e 옵션 사용)
pip install -e . --no-deps
```

* 버전 수정: llmperf에 설정된 파이썬 상한선( <3.11)을 제거하여 현재 사용 중인 3.12.3 환경에서 설치가 거부되는 것을 방지.
* --no-deps 설정: 이미 vLLM 환경에 설치된 패키지들과 llmperf 간의 불필요한 버전 충돌(Pydantic 등) 검사를 건너뛰고 설치.

![alt text](<Screenshot 2026-01-14 at 3.23.54 PM.png>)

설치 완료 후에 아래 명령어를 Host 에서 실행

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY=dummy

python token_benchmark_ray.py \
    --model "/home/ubuntu/qwen3_32b_dense_bf16" \
    --mean-input-tokens 128 \
    --stddev-input-tokens 0 \
    --mean-output-tokens 512 \
    --stddev-output-tokens 0 \
    --max-num-completed-requests 10 \
    --timeout 1200 \
    --num-concurrent-requests 1 \
    --results-dir /tmp/results \
    --llm-api openai \
    --additional-sampling-params '{}'

```

<img width="364" height="1172" alt="Screenshot 2025-12-08 at 9 12 32 PM" src="https://github.com/user-attachments/assets/ce17baea-c904-42a4-bf9d-b848db455af5" />



## 2.🖥️ Host machine 의 가상환경을 활용한 vLLM 서버 배포

### Step 2-1: vLLM V1 가상환경 활성화 
```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
```

* 이후 서버 실행 및 추론 테스트, Benchmarking 과정은 Docker 방식에서 안내 한 내용과 동일하게 진행

[⚙️ Step 1-3 vllm server 실행](#️-step-1-3-vllm-server-실행)

[🧪 Step 1-4: 추론 테스트 (Inference)](#-step-1-4-추론-테스트-inference)

[📊 Step 1-5: Performance Benchmarking (via Host)](#-step-1-5-performance-benchmarking-via-host)

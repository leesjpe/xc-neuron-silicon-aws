# Serving Qwen 3/2.5 on Trainium2 with vLLM (NxD)

이 가이드는 **AWS Trainium2 (`trn2.48xlarge`)** 인스턴스에서 **vLLM**과 **Neuronx Distributed (NxD)** 아키텍처를 사용하여 **Qwen 3 (32B)** 모델을 서빙하는 방법을 설명합니다.

기존 Legacy 방식이 아닌, **NxD 기반의 vLLM (v1 파이프라인)**을 사용하여 대규모 모델에 대한 안정성과 최적화된 성능을 제공합니다.

---

## 📋 Prerequisites (사전 준비)

진행하기 전에 다음 사항들을 확인하세요.

1.  **인스턴스 실행:** `trn2.48xlarge` 인스턴스가 활성화(`Running`) 상태여야 합니다.
2.  **(선택 사항이지만 권장) 고속 스토리지 설정:**
    * 모델 로딩 속도와 체크포인트 저장 속도를 높이기 위해 로컬 NVMe SSD (RAID 0) 사용을 강력히 권장합니다.
    * 아직 설정하지 않으셨다면, 아래 가이드를 먼저 진행해 주세요.
    * 👉 **[가이드: 고속 스토리지 설정 (NVMe RAID 0)](../storage/local-nvme-setup.md)**
    * *참고: 이 과정을 건너뛴다면, 루트 EBS 볼륨에 모델을 저장할 충분한 공간이 있는지 확인하세요.*

---

## 🐳 Step 1: Neuron vLLM 컨테이너 실행

라이브러리 의존성 충돌을 방지하기 위해 AWS 공식 **Neuron Deep Learning Container (DLC)**를 사용합니다.

아래 명령어를 사용하여 컨테이너를 실행하고 내부 쉘로 진입합니다.
*(고속 스토리지를 설정했다면 `/data` 마운트가 필수이며, EBS만 사용한다면 경로는 상황에 맞게 조정하세요.)*

```bash
# Docker 실행 및 진입
# -v /data:/data : 고속 스토리지(또는 모델 경로) 마운트
# -p 8000:8000 : API 서버 포트 개방

docker run -d -it \
  --privileged \
  -v /home/ubuntu/:/home/ubuntu/ \
  -v /data:/data \
  -p 8000:8000 \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py311-sdk2.26.1-ubuntu22.04
```

## 📥 Step 2: 가상환경 진입

```bash
docker exec -it <Container ID> bash
```

## ⚙️ Step 3: 환경 변수 설정 및 vllm server 실행
3-2 과정은  10~15 소요되며 아래와 같이 로그가 보이면 컴파일 및 서버 시작 완료 ☕️

```bash
# 3-1. 환경 변수 설정
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS="/data/Qwen-32B-BS1-SL6k-TP64"
export MODEL_ID="Qwen/Qwen3-32B"

# 3-2. 서버 실행 (외부 접속 허용)
VLLM_USE_V1=0 vllm serve $MODEL_ID \
    --tensor-parallel-size 64 \
    --max-num-seqs 1 \
    --max-model-len 6400 \
    --override-neuron-config '{"save_sharded_checkpoint": true}' \
    --host 0.0.0.0 \
    --port 8000
```
<img width="1294" height="845" alt="Screenshot 2025-12-06 at 9 17 48 PM" src="https://github.com/user-attachments/assets/4cf45802-3e9a-4290-b0c0-e5303f384e40" />


## 🧪 Step 4: 추론 테스트 (Inference)
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
"model": "Qwen/Qwen3-32B",
"prompt": "What is machine learning?",
"max_tokens": 100,
"temperature": 0.7
}'
```
<img width="1295" height="316" alt="Screenshot 2025-12-06 at 10 38 58 PM" src="https://github.com/user-attachments/assets/451cf358-9bc8-45ab-bc77-fb491cb57a6d" />



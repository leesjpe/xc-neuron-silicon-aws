---

# vLLM V1 가이드 Neuron Plugin

이 문서는 AWS Neuron 환경에서 vLLM V1 설정에 대한 내용을 설명합니다. 
vLLM V0 과 vLLM V1 의 차이점에 대해서는 vLLM 공식 블로그 [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) 를 참고해 주세요.

* v0.x (In-tree 방식): 예전에는 NVIDIA GPU, AMD GPU, AWS Neuron, OpenVINO 등의 코드가 모두 vLLM이라는 하나의 거대한 프로젝트(Repository) 안에 포함
* v0.6.0 ~ v1.x (Out-of-tree/Plugin 방식): vLLM 팀은 핵심 엔진(스케줄러, API 서버 등)만 남기고, 각 하드웨어별 최적화 코드는 외부로 분리



## 🚀 Neuron 의 운영 방식 변화
vLLM V1(최신 실행 엔진)은 처음부터 이런 **모듈화(Modularity)**를 염두에 두고 설계 되었습니다.
AWS Neuron도 vllm-neuron이라는 플러그인을 독립적으로 개발하고 배포할 수 있게 된 것입니다.
기존 v0에서는 vLLM 메인 소스 코드 안에 Neuron 코드가 포함되어 있었으나 Upstreaming 방식으로 AWS 의 Fork 리포지토리로 관리되었으나, vLLM V1 에서 현재는 **플러그인 방식**으로 운영됩니다.


* **독립적 배포:** vLLM 본체의 업데이트와 상관없이 AWS Neuron의 최신 SDK(Neuron SDK 2.27 이상) 지원을 빠르게 반영할 수 있습니다.
(Neuron 2.28 버전부터 vLLM v0 버전은 지원이 중단됩니다. v1 버전으로 vLLM 을 시작하길 권장합닏)
* **경량화:** 사용자는 자신의 하드웨어(NVIDIA, AMD, Neuron 등)에 맞는 플러그인만 선택적으로 설치하여 환경을 최적화할 수 있습니다.

---

## 📦 설치 방법 (Installation)

플러그인 방식에서는 `vllm` 기본 패키지와 `vllm-neuron` 플러그인을 함께 설치해야 합니다.

```bash
# 1. AWS Neuron용 PyTorch 및 관련 SDK 설치 (AWS 가이드 권장 버전)
pip install torch-neuronx torchvision torchaudio --extra-index-url https://pip.repos.neuron.amazonaws.com

# 2. vLLM Neuron 플러그인 설치 (vLLM 핵심 엔진이 함께 설치됨)
pip install vllm-neuron

```

---

## 🔍 버전 확인 가이드 (Version Check)

플러그인 구조에서는 각 구성 요소의 버전 호환성이 중요합니다. 환경 문제 발생 시 아래 명령어를 통해 실질적인 버전을 확인하세요.

### 1. 소프트웨어 패키지 버전 확인

가장 먼저 확인해야 할 세 가지 핵심 패키지입니다.

```bash
# 플러그인(vllm-neuron) 버전 확인
pip show vllm-neuron

# 베이스 엔진(vllm) 버전 확인
pip show vllm

# 전체 Neuron 관련 패키지 리스트 확인
pip list | grep -E "neuron|vllm"

```

### 2. 하드웨어 및 드라이버 상태 확인

vLLM이 실제 Neuron Core를 인식하고 있는지 확인합니다.

```bash
# 설치된 Neuron 디바이스(Inf2, Trn1 등) 및 SDK 버전 확인
neuron-ls

# (실행 중일 때) 실시간 NeuronCore 및 HBM 메모리 점유율 확인
neuron-top

```

---

## 🛠️ 사용 방법 (Usage)

vLLM을 실행할 때 플러그인이 자동으로 로드되도록 환경 변수를 설정하거나 실행 옵션을 지정합니다.

### 환경 변수 설정

```bash
export VLLM_TARGET_DEVICE=neuron

```

### 서버 실행 예시

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <모델_경로_또는_ID> \
    --max-model-len 4096 \
    --block-size 128 \
    --device neuron

```

*주의: Neuron 환경에서는 `--block-size`를 128로 설정하는 것이 권장됩니다.*

---

## 💡 자주 묻는 질문 (FAQ)

**Q: 플러그인을 따로 설치한 적이 없는데 왜 플러그인 방식이라고 하나요?**
A: `pip install vllm-neuron`을 수행했다면 이미 플러그인을 설치한 것입니다. vLLM은 실행 시 내부적으로 `entry_points`를 검색하여 설치된 `vllm-neuron` 패키지를 찾아 하드웨어 가속기로 자동 등록합니다.

**Q: 모델 로딩 중 "No neuron-specific executor found" 에러가 납니다.**
A: `vllm-neuron` 패키지가 정상적으로 설치되었는지 `pip show vllm-neuron`으로 확인하고, `VLLM_TARGET_DEVICE=neuron` 환경 변수가 설정되었는지 점검하세요.

---

## 🔗 참고 링크

* [vLLM 공식 문서](https://docs.vllm.ai/)
* [AWS Neuron 공식 문서](https://awsdocs-neuron.readthedocs-hosted.com/)
* [vLLM-Neuron GitHub Repository](https://github.com/vllm-project/vllm-neuron)

---

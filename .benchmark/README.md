# Neuron Model Benchmark Suite

설정 파일 기반의 유연한 모델 컴파일 및 벤치마크 도구입니다.

## 📁 디렉토리 구조

```
benchmark/
├── configs/                    # 모델별 설정 파일
│   ├── llama31-70b.conf       # Llama 3.1 70B 설정
│   └── qwen3-8b.conf          # Qwen3 8B 설정
├── scripts/                    # 실행 스크립트
│   ├── compile_model.sh       # 모델 컴파일
│   ├── run_benchmark.sh       # Performance 벤치마크 (vLLM Bench)
│   ├── run_llmperf.sh         # Performance 벤치마크 (LLMPerf)
│   └── run_accuracy.sh        # Accuracy 테스트
├── reports/                    # 리포트 생성 도구
│   ├── generate_report.py     # 텍스트 리포트
│   └── generate_html_report.py # HTML 리포트
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 0. 사전 준비

**자동 설정 (추천):**
```bash
cd benchmark
chmod +x setup.sh
./setup.sh
# → 안전한 의존성만 설치하는 옵션 선택 (Option 1)
```

**수동 설정 (안전):**
```bash
# AWS Neuron Samples 리포지토리 클론
cd ~
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git

# 안전한 의존성만 설치 (기존 vLLM 환경 보호)
pip install lm-eval datasets tiktoken openai psutil botocore

# 스크립트 실행 권한
chmod +x benchmark/scripts/*.sh
```

**⚠️ 중요: 의존성 충돌 방지**

aws-neuron-samples의 requirements.txt에는 다음 패키지들이 포함되어 있습니다:
- `torch`, `transformers`, `pydantic>2.10`, `pyarrow==20.0.0`

이들은 기존 vLLM 환경과 충돌할 수 있으므로, **안전한 의존성만 설치**하는 것을 권장합니다:
```bash
# 안전한 패키지만 설치
pip install lm-eval datasets tiktoken openai psutil botocore

# 위험한 패키지는 설치하지 않음 (기존 버전 사용)
# - torch (Neuron torch 사용)
# - transformers (기존 버전 사용)
# - pydantic (vLLM 호환 버전 사용)
# - pyarrow (기존 버전 사용)
```

### 1. 모델 컴파일

```bash
cd benchmark/scripts

# Llama 3.1 70B 컴파일 (Light: 3개 모델)
./compile_model.sh ../configs/llama31-70b.conf light

# Qwen3 8B 컴파일 (Medium: 6개 모델)
./compile_model.sh ../configs/qwen3-8b.conf medium
```

**컴파일 레벨:**
- `light`: 3개 모델 (BS1, BS2, BS4 각 1개)
- `medium`: 6개 모델 (각 배치 사이즈별 2개)
- `heavy`: 9개 모델 (각 배치 사이즈별 3개)

### 2. 벤치마크 실행

#### Performance Tests

##### vLLM Bench (빠른 검증)

```bash
# Llama 3.1 70B 벤치마크 (Light: 6개 테스트)
./run_benchmark.sh ../configs/llama31-70b.conf light

# Qwen3 32B 벤치마크 (Medium: 15개 테스트)
./run_benchmark.sh ../configs/qwen3-32b.conf medium
```

**테스트 레벨:**
- `light`: 6개 테스트 (~30-40분)
- `medium`: 15개 테스트 (~1-2시간)
- `heavy`: 30개 테스트 (~2-4시간)

#### LLMPerf (상세 분석)

```bash
# Llama 3.1 70B LLMPerf (Light: 9개 테스트)
./run_llmperf.sh ../configs/llama31-70b.conf light

# Qwen3 32B LLMPerf (Medium: 25개 테스트)
./run_llmperf.sh ../configs/qwen3-32b.conf medium
```

**LLMPerf 특징:**
- 정규분포 기반 실제 워크로드 시뮬레이션
- P50, P90, P95, P99 상세 메트릭
- 다양한 동시성 레벨 테스트

**테스트 레벨:**
- `light`: 9개 테스트 (3 variations × 3 concurrency)
- `medium`: 25개 테스트 (5 variations × 5 concurrency)
- `heavy`: 56개 테스트 (7 variations × 8 concurrency)

#### Accuracy Tests

```bash
# Llama 3.1 70B 정확도 테스트 (Light: 2 datasets)
./run_accuracy.sh ../configs/llama31-70b.conf light

# Qwen3 8B 정확도 테스트 (Medium: 4 datasets)
./run_accuracy.sh ../configs/qwen3-8b.conf medium
```

**Accuracy 특징:**
- AWS Neuron 공식 accuracy.py 사용 (aws-neuron-samples)
- lm-eval 기반 표준 벤치마크
- MMLU, GSM8K, HellaSwag, ARC 등
- 자동 서버 관리 및 결과 수집

**사전 준비:**
```bash
# 첫 실행 시 자동으로 클론되고 안전한 의존성만 설치됨
# 또는 수동 설치:
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git ~/aws-neuron-samples
pip install lm-eval datasets tiktoken openai psutil botocore
```

**⚠️ 의존성 충돌 주의:**
- requirements.txt의 torch, transformers, pydantic는 설치하지 마세요
- 기존 vLLM 환경과 충돌할 수 있습니다
- 스크립트가 자동으로 안전한 패키지만 설치합니다

**테스트 레벨:**
- `light`: 2개 데이터셋 (MMLU 100샘플, GSM8K 50샘플) - ~10-15분
- `medium`: 4개 데이터셋 (각 200-500샘플) - ~30-60분
- `heavy`: 6개 데이터셋 (전체 데이터셋) - ~1-2시간

### 3. 리포트 생성

```bash
cd benchmark/reports

# Performance 결과 리포트 (vLLM Bench)
python3 generate_report.py ../scripts/benchmark_results/20260211_040816_light_llama-3-1-70b-instruct
python3 generate_html_report.py ../scripts/benchmark_results/20260211_040816_light_llama-3-1-70b-instruct

# Performance 결과 리포트 (LLMPerf)
python3 generate_report.py ../scripts/llmperf_results/20260211_050123_light_llama-3-1-70b-instruct
python3 generate_html_report.py ../scripts/llmperf_results/20260211_050123_light_llama-3-1-70b-instruct

# Accuracy 결과 리포트
python3 generate_report.py ../scripts/accuracy_results/20260211_060123_light_llama-3-1-70b-instruct
python3 generate_html_report.py ../scripts/accuracy_results/20260211_060123_light_llama-3-1-70b-instruct
```

## 🔍 벤치마크 도구 비교

### Performance Tests

#### vLLM Bench
- ✅ 빠른 실행 (내장 도구)
- ✅ 간단한 설정
- ✅ 기본 메트릭 (Throughput, TTFT, TPOT)
- 📊 고정 입력/출력 길이

**사용 시나리오:** 빠른 성능 검증, 배치 사이즈 비교

#### LLMPerf
- ✅ 상세한 메트릭 (P50, P90, P95, P99)
- ✅ 실제 워크로드 시뮬레이션 (정규분포)
- ✅ 다양한 동시성 레벨
- 📊 통계적 분석

**사용 시나리오:** 프로덕션 평가, 상세 성능 분석

### Accuracy Tests

#### AWS Neuron Accuracy Suite
- ✅ 공식 AWS Neuron 도구
- ✅ lm-eval 기반 표준 벤치마크
- ✅ 자동 서버 관리
- 📊 MMLU, GSM8K, HellaSwag, ARC 등

**사용 시나리오:** 모델 정확도 검증, 컴파일 영향 평가

### 권장 워크플로우

```bash
# 1. 모델 컴파일
./compile_model.sh ../configs/llama31-70b.conf light

# 2. 빠른 성능 검증 (vLLM Bench)
./run_benchmark.sh ../configs/llama31-70b.conf light

# 3. 정확도 검증 (Accuracy)
./run_accuracy.sh ../configs/llama31-70b.conf light

# 4. 상세 성능 분석 (LLMPerf) - 선택사항
./run_llmperf.sh ../configs/llama31-70b.conf light

# 5. 결과 확인
ls -lh benchmark_results/
ls -lh accuracy_results/
ls -lh llmperf_results/
```

## ⚙️ 새 모델 추가

### 1. 설정 파일 생성

`configs/your-model.conf` 파일을 생성:

```bash
# Model Information
MODEL_NAME="Your-Model-Name"
MODEL_PATH="/home/ubuntu/models/Your-Model/"
MODEL_TYPE="llama"  # llama, qwen2, mistral 등
TASK_TYPE="causal-lm"

# Compilation Settings
TORCH_DTYPE="bfloat16"
TP_DEGREE=64
PAD_TOKEN_ID=2

# Batch Size Configurations
BS1_CONFIG="12288 12800"  # CONTEXT_LENGTH SEQ_LENGTH
BS2_CONFIG="8192 8704"
BS4_CONFIG="4096 4608"

# Bucketing Configuration
BS1_CONTEXT_BUCKETS="2048 4096 8192 12288"
BS1_TOKEN_BUCKETS="2048 4096 8192 12800"
BS2_CONTEXT_BUCKETS="2048 4096 8192"
BS2_TOKEN_BUCKETS="2048 4096 8192 8704"
BS4_CONTEXT_BUCKETS="2048 4096"
BS4_TOKEN_BUCKETS="2048 4096 4608"

# Compilation Options
COMPILE_OPTS="--on-device-sampling \
--top-k 1 \
--do-sample \
--fused-qkv \
--sequence-parallel-enabled \
--qkv-kernel-enabled \
--attn-kernel-enabled \
--mlp-kernel-enabled \
--cc-pipeline-tiling-factor 1 \
--enable-bucketing"

# Neuron Runtime Settings
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_RT_NUM_CORES=64
NEURON_RT_EXEC_TIMEOUT=1800
XLA_DENSE_GATHER_FACTOR=0
NEURON_RT_INSPECT_ENABLE=0

# vLLM Server Settings
VLLM_BLOCK_SIZE=16
VLLM_RPC_TIMEOUT=100000

# Accuracy Test Settings
ACCURACY_LIGHT_TESTS=(
    "mmlu:100"
    "gsm8k:50"
)

ACCURACY_MEDIUM_TESTS=(
    "mmlu:500"
    "gsm8k:200"
    "hellaswag:500"
    "arc_challenge:200"
)

ACCURACY_HEAVY_TESTS=(
    "mmlu:0"
    "gsm8k:0"
    "hellaswag:0"
    "arc_challenge:0"
    "truthfulqa:0"
    "winogrande:0"
)

ACCURACY_MAX_CONCURRENT_REQUESTS=1
ACCURACY_TIMEOUT=3600
ACCURACY_SERVER_PORT=8000
ACCURACY_N_VLLM_THREADS=16
ACCURACY_CLIENT_PARAMS_BATCH_SIZE=1
ACCURACY_CLIENT_PARAMS_NUM_FEW_SHOT=5
```

### 2. 컴파일 및 벤치마크

```bash
cd benchmark/scripts

# 컴파일
./compile_model.sh ../configs/your-model.conf light

# Performance 벤치마크
./run_benchmark.sh ../configs/your-model.conf light

# Accuracy 테스트
./run_accuracy.sh ../configs/your-model.conf light
```

## 📊 결과 파일

벤치마크 실행 후 생성되는 파일들:

### Performance Results (vLLM Bench / LLMPerf)
```
benchmark_results/20260211_040816_light_llama-3-1-70b-instruct/
├── test_metadata.json                  # 전체 테스트 메타데이터
├── summary_light.csv                   # CSV 요약
├── failures.log                        # 실패 로그
├── result_*.json                       # 개별 테스트 결과
├── benchmark_*.json                    # vLLM 벤치마크 원본
├── benchmark_*.log                     # 벤치마크 로그
└── server_*.log                        # 서버 로그
```

### Accuracy Results
```
accuracy_results/20260211_060123_light_llama-3-1-70b-instruct/
├── test_metadata.json                  # 전체 테스트 메타데이터
├── summary_light.csv                   # CSV 요약
├── result_*.json                       # 개별 테스트 결과
├── config_*.yaml                       # 테스트별 설정 파일
└── accuracy_*.log                      # 정확도 테스트 로그
```

## 🔧 고급 사용법

### 백그라운드 실행

```bash
# nohup으로 백그라운드 실행
nohup ./run_benchmark.sh ../configs/llama31-70b.conf light > benchmark.log 2>&1 &

# 진행 상황 확인
tail -f benchmark.log
```

### 특정 배치 사이즈만 컴파일

설정 파일을 수정하여 원하는 배치 사이즈만 설정:

```bash
# configs/llama31-70b-bs1-only.conf
BS1_CONFIG="12288 12800"
BS2_CONFIG=""  # 비활성화
BS4_CONFIG=""  # 비활성화
```

### 결과 비교

여러 모델의 결과를 비교:

```bash
# Llama 벤치마크
./run_benchmark.sh ../configs/llama31-70b.conf light

# Qwen 벤치마크
./run_benchmark.sh ../configs/qwen3-32b.conf light

# 결과 비교
ls -lh benchmark_results/
```

## 📝 주의사항

1. **컴파일 시간**: 첫 컴파일은 20-60분 소요
2. **디스크 공간**: 컴파일된 모델당 수 GB 필요
3. **메모리**: 배치 사이즈가 클수록 메모리 사용량 증가
4. **동시 실행 금지**: 한 번에 하나의 벤치마크만 실행
5. **Accuracy 테스트**: aws-neuron-samples 리포지토리 필요 (자동 클론됨)
6. **의존성 충돌**: requirements.txt의 일부 패키지는 기존 환경과 충돌 가능

### 의존성 관리

**안전한 패키지 (설치 권장):**
```bash
pip install lm-eval datasets tiktoken openai psutil botocore
```

**위험한 패키지 (설치 금지):**
- `torch` - Neuron torch와 충돌
- `transformers` - 버전 충돌 가능
- `pydantic>2.10` - vLLM 호환성 문제
- `pyarrow==20.0.0` - 버전 고정으로 충돌

**확인 방법:**
```bash
# 현재 설치된 버전 확인
pip list | grep -E "torch|transformers|pydantic|pyarrow"

# lm-eval 설치 확인
python3 -c "import lm_eval; print('lm-eval:', lm_eval.__version__)"
```

## 🐛 문제 해결

### 컴파일 실패

```bash
# 로그 확인
cat /home/ubuntu/compiled_models/llama31-70b-bs1-ctx12288/compile.log

# 타임아웃 증가
# configs/*.conf 파일에서:
NEURON_RT_EXEC_TIMEOUT=3600  # 60분으로 증가
```

### 벤치마크 실패

```bash
# 실패 로그 확인
cat benchmark_results/*/failures.log

# 서버 로그 확인
cat benchmark_results/*/server_*.log
```

### 메타데이터 업데이트 실패

```bash
cd benchmark_results/20260211_040816_light_llama-3-1-70b-instruct/

# 수동 업데이트
python3 << 'EOF'
import json, glob
from datetime import datetime

with open('test_metadata.json', 'r') as f:
    metadata = json.load(f)

metadata['tests'] = []
for result_file in sorted(glob.glob('result_*.json')):
    with open(result_file, 'r') as f:
        metadata['tests'].append(json.load(f))

metadata['end_time'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
metadata['total_tests'] = len(metadata['tests'])
metadata['successful_tests'] = sum(1 for t in metadata['tests'] if t['status'] == 'SUCCESS')
metadata['failed_tests'] = sum(1 for t in metadata['tests'] if t['status'] == 'FAILED')

with open('test_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
EOF
```

### Accuracy 스크립트 찾기 실패

```bash
# aws-neuron-samples 확인
ls -la ~/aws-neuron-samples/inference-benchmarking/accuracy.py

# 없으면 클론
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git ~/aws-neuron-samples

# 의존성 설치
pip install -r ~/aws-neuron-samples/inference-benchmarking/requirements.txt

# 스크립트가 자동으로 클론하도록 설정되어 있음
# 첫 실행 시 자동으로 처리됨
```

## 📚 참고 문서

- [Llama 3.1 70B 가이드](../inference/Llama3.1-70B-Instruct-Trn2.md)
- [Qwen3 32B 가이드](../inference/Qwen3-32B-Dense-BF16-Trn2.md)
- [AWS Neuron 공식 문서](https://awsdocs-neuron.readthedocs-hosted.com/)
- [AWS Neuron Accuracy Evaluation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/accuracy-eval-with-datasets.html)
- [lm-eval Documentation](https://github.com/EleutherAI/lm-evaluation-harness)

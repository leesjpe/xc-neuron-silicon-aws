# Benchmark Suite Quick Start

## 사전 준비

**방법 1: 자동 설정 (추천)**
```bash
cd benchmark
chmod +x setup.sh
./setup.sh
# → Option 1 선택 (안전한 의존성만 설치)
```

**방법 2: 수동 설정 (안전)**
```bash
# 1. AWS Neuron 가상환경 활성화
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 2. AWS Neuron Samples 클론
cd ~
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git

# 3. 안전한 의존성만 설치 (기존 vLLM 환경 보호)
pip install lm-eval datasets tiktoken openai psutil botocore

# 4. 스크립트 실행 권한
chmod +x benchmark/scripts/*.sh
```

**⚠️ 중요: 의존성 충돌 방지**

requirements.txt에 포함된 다음 패키지들은 설치하지 마세요:
- `torch` → Neuron torch와 충돌
- `transformers` → 기존 버전과 충돌 가능
- `pydantic>2.10` → vLLM과 충돌 가능
- `pyarrow==20.0.0` → 버전 고정으로 충돌 가능

안전한 패키지만 설치:
- `lm-eval` (필수: accuracy 평가)
- `datasets`, `tiktoken` (데이터 로딩)
- `openai`, `psutil`, `botocore` (유틸리티)

## 전체 워크플로우

```bash
cd benchmark/scripts

# 1단계: 모델 컴파일 (20-60분)
./compile_model.sh ../configs/llama31-70b.conf light

# 2단계: Performance 테스트 (30-40분)
./run_benchmark.sh ../configs/llama31-70b.conf light

# 3단계: Accuracy 테스트 (10-15분)
./run_accuracy.sh ../configs/llama31-70b.conf light

# 4단계: 리포트 생성
cd ../reports
python3 generate_html_report.py ../scripts/benchmark_results/[최신_디렉토리]
python3 generate_html_report.py ../scripts/accuracy_results/[최신_디렉토리]
```

## 테스트 레벨 선택

### Light (추천: PoC/빠른 검증)
- 컴파일: 3개 모델 (~1-2시간)
- Performance: 6개 테스트 (~30-40분)
- Accuracy: 2개 데이터셋 (~10-15분)
- **총 소요시간: ~2-3시간**

```bash
./compile_model.sh ../configs/llama31-70b.conf light
./run_benchmark.sh ../configs/llama31-70b.conf light
./run_accuracy.sh ../configs/llama31-70b.conf light
```

### Medium (추천: 상세 평가)
- 컴파일: 6개 모델 (~2-4시간)
- Performance: 15개 테스트 (~1-2시간)
- Accuracy: 4개 데이터셋 (~30-60분)
- **총 소요시간: ~4-7시간**

```bash
./compile_model.sh ../configs/llama31-70b.conf medium
./run_benchmark.sh ../configs/llama31-70b.conf medium
./run_accuracy.sh ../configs/llama31-70b.conf medium
```

### Heavy (추천: 프로덕션 검증)
- 컴파일: 9개 모델 (~3-6시간)
- Performance: 30개 테스트 (~2-4시간)
- Accuracy: 6개 데이터셋 (전체, ~1-2시간)
- **총 소요시간: ~6-12시간**

```bash
./compile_model.sh ../configs/llama31-70b.conf heavy
./run_benchmark.sh ../configs/llama31-70b.conf heavy
./run_accuracy.sh ../configs/llama31-70b.conf heavy
```

## 백그라운드 실행

장시간 실행이 필요한 경우:

```bash
# nohup으로 백그라운드 실행
nohup ./compile_model.sh ../configs/llama31-70b.conf light > compile.log 2>&1 &
nohup ./run_benchmark.sh ../configs/llama31-70b.conf light > benchmark.log 2>&1 &
nohup ./run_accuracy.sh ../configs/llama31-70b.conf light > accuracy.log 2>&1 &

# 진행 상황 확인
tail -f compile.log
tail -f benchmark.log
tail -f accuracy.log

# 프로세스 확인
ps aux | grep -E "compile_model|run_benchmark|run_accuracy"
```

## 결과 확인

```bash
# 결과 디렉토리 확인
ls -lht benchmark_results/
ls -lht accuracy_results/
ls -lht llmperf_results/

# 최신 결과 확인
ls -lht benchmark_results/ | head -5
ls -lht accuracy_results/ | head -5

# CSV 요약 확인
cat benchmark_results/[최신_디렉토리]/summary_light.csv
cat accuracy_results/[최신_디렉토리]/summary_light.csv
```

## 일반적인 문제 해결

### 1. 컴파일 실패
```bash
# 로그 확인
cat /home/ubuntu/compiled_models/[모델명]/compile.log

# 타임아웃 증가 (config 파일 수정)
NEURON_RT_EXEC_TIMEOUT=3600
```

### 2. 벤치마크 서버 시작 실패
```bash
# 기존 서버 종료
pkill -f "vllm.entrypoints.openai.api_server"

# 포트 확인
lsof -i :8000

# 서버 로그 확인
cat benchmark_results/[최신_디렉토리]/server_*.log
```

### 3. Accuracy 스크립트 없음
```bash
# aws-neuron-samples 확인
ls -la ~/aws-neuron-samples/inference-benchmarking/accuracy.py

# 없으면 클론
cd ~
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git

# 안전한 의존성만 설치 (기존 vLLM 환경 보호)
pip install lm-eval datasets tiktoken openai psutil botocore

# 위험한 패키지는 설치하지 않음
# torch, transformers, pydantic, pyarrow는 기존 버전 사용
```

### 4. 의존성 충돌
```bash
# 증상: accuracy 테스트 실행 시 import 에러
# 원인: requirements.txt의 패키지가 기존 환경과 충돌

# 해결: 안전한 패키지만 재설치
pip install --no-deps lm-eval
pip install datasets tiktoken openai psutil botocore

# 확인
python3 -c "import lm_eval; print('lm-eval OK')"
python3 -c "import datasets; print('datasets OK')"
```

### 4. 메모리 부족
```bash
# 메모리 사용량 확인
free -h
neuron-top

# 배치 사이즈 줄이기 (config 파일 수정)
# BS4 대신 BS2 또는 BS1 사용
```

## 다중 모델 테스트

```bash
# Llama 3.1 70B
./compile_model.sh ../configs/llama31-70b.conf light
./run_benchmark.sh ../configs/llama31-70b.conf light
./run_accuracy.sh ../configs/llama31-70b.conf light

# Qwen3 8B
./compile_model.sh ../configs/qwen3-8b.conf light
./run_benchmark.sh ../configs/qwen3-8b.conf light
./run_accuracy.sh ../configs/qwen3-8b.conf light

# 결과 비교
ls -lh benchmark_results/ | grep llama
ls -lh benchmark_results/ | grep qwen
ls -lh accuracy_results/ | grep llama
ls -lh accuracy_results/ | grep qwen
```

## 결과 정리

```bash
# 오래된 결과 삭제 (30일 이상)
find benchmark_results/ -type d -mtime +30 -exec rm -rf {} \;
find accuracy_results/ -type d -mtime +30 -exec rm -rf {} \;
find llmperf_results/ -type d -mtime +30 -exec rm -rf {} \;

# 특정 테스트 결과만 삭제
rm -rf benchmark_results/20260211_*
rm -rf accuracy_results/20260211_*

# 컴파일된 모델 정리 (재컴파일 필요)
rm -rf /home/ubuntu/compiled_models/llama31-70b-*
```

## 성능 최적화 팁

1. **컴파일 최적화**
   - Light 레벨로 시작해서 필요한 설정만 추가
   - 자주 사용하는 배치 사이즈만 컴파일

2. **테스트 순서**
   - Performance → Accuracy 순서로 실행
   - 컴파일은 한 번만, 여러 테스트 재사용

3. **리소스 관리**
   - 한 번에 하나의 테스트만 실행
   - 백그라운드 실행으로 시간 절약
   - 불필요한 로그 파일 정기 삭제

4. **결과 분석**
   - HTML 리포트로 시각화
   - CSV로 엑셀 분석
   - 여러 모델 결과 비교

## 추가 도움말

자세한 내용은 [README.md](README.md)를 참고하세요.

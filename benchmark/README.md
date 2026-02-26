# LLMPerf

```bash
# 1. llmperf 소스 코드 다운로드 (git clone)
git clone https://github.com/ray-project/llmperf.git
cd llmperf

# 2. pyproject.toml 파일에서 Python 버전 제한(<3.11)을 <3.13으로 수정 (sed 명령어 사용)
sed -i 's/<3.11/<3.13/g' pyproject.toml

# 3. 수정된 소스로 설치 (-e 옵션 사용)
pip install -e . --no-deps
```

## compile

```bash
cd benchmark/scripts
 
# 1단계: 모델 컴파일 (설정된 모든 조합에 대해 실행)
./compile_model.sh ../configs/qwen3-8b.conf
 
# 2단계: Performance 테스트 (vLLM Benchmark)
./run_vllm_bench.sh ../configs/qwen3-8b.conf
 
# 3단계: Performance 테스트 (LLMPerf)
./run_llmperf.sh ../configs/qwen3-8b.conf
 
# 4단계: Accuracy 테스트
./run_accuracy.sh ../configs/qwen3-8b.conf

```

## Runs in the background

```bash
# nohup으로 백그라운드 실행
nohup ./compile_model.sh ../configs/llama31-70b.conf > /dev/null 2>&1 &
nohup ./run_llmperf.sh ../configs/llama31-70b.conf > /dev/null 2>&1 &
nohup ./run_accuracy.sh ../configs/llama31-70b.conf > /dev/null 2>&1 &

# 프로세스 확인
ps aux | grep -E "compile_model|run_llmperf|run_accuracy"
```

## Result

```bash
# 결과 디렉토리 확인
ls -lht ~/benchmark_results/

# 최신 결과 확인
ls -lht benchmark_results/ | head -5
ls -lht accuracy_results/ | head -5
```

## Trouble shooting

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

#!/bin/bash

# LLMPerf 기반 벤치마크 실행 스크립트
# 사용법: ./run_llmperf.sh <config_file> [light|medium|heavy]
# 예시: ./run_llmperf.sh ../configs/llama31-70b.conf light

set +e  # 에러 발생해도 계속 진행

# 인자 확인
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [test_level]"
    echo "Example: $0 ../configs/llama31-70b.conf light"
    echo ""
    echo "Available configs:"
    ls -1 ../configs/*.conf 2>/dev/null || echo "  No config files found"
    exit 1
fi

CONFIG_FILE=$1
TEST_LEVEL=${2:-"light"}

# 설정 파일 로드
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
source $CONFIG_FILE

# 기본 경로 설정
BASE_COMPILED_PATH="${BASE_COMPILED_PATH:-/data/compiled_models}"
RESULTS_DIR="./llmperf_results"
COMPILED_MODEL_PREFIX=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '.' '-')

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}_${TEST_LEVEL}_${COMPILED_MODEL_PREFIX}"
mkdir -p $TEST_RUN_DIR

# Neuron 런타임 설정
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

# LLMPerf 설치 확인 및 설치 (소스 코드 기반)
echo ""
LLMPERF_DIR="${HOME}/llmperf"
LLMPERF_SCRIPT="${LLMPERF_DIR}/token_benchmark_ray.py"

if [ ! -f "$LLMPERF_SCRIPT" ]; then
    echo "❌ LLMPerf script not found at $LLMPERF_SCRIPT"
    echo ""
    read -p "Do you want to clone and install llmperf from source now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Cloning llmperf repository to $LLMPERF_DIR..."
        git clone https://github.com/ray-project/llmperf.git "$LLMPERF_DIR"
        
        echo "🔧 Patching token_benchmark_ray.py for import compatibility and syntax warning..."
        # Patch for llmperf version mismatch: common_metrics.py was renamed to common.py
        sed -i 's/from llmperf import common_metrics/from llmperf import common/' "${LLMPERF_DIR}/token_benchmark_ray.py"
        sed -i 's/common_metrics\./common\./g' "${LLMPERF_DIR}/token_benchmark_ray.py"

        sed -i 's/print(f"\\Results/print(f"Results/' "${LLMPERF_DIR}/token_benchmark_ray.py" # Fix SyntaxWarning: invalid escape sequence '\R'
        
        echo "🔧 Modifying pyproject.toml for Python compatibility..."
        # pyproject.toml의 python 버전 제약을 완화합니다.
        sed -i 's/<3.11/<3.13/g' "${LLMPERF_DIR}/pyproject.toml"
        
        echo "📦 Installing llmperf in editable mode with no dependencies..."
        # Execute pip install from within the llmperf directory to mimic manual steps
        if ! (cd "$LLMPERF_DIR" && pip install -e . --no-deps); then
            echo "❌ llmperf installation failed. Please check the pip error message above."
            exit 1
        fi
        
        if [ ! -f "$LLMPERF_SCRIPT" ]; then
            echo "❌ Installation seems to have failed. Script not found at $LLMPERF_SCRIPT"
            exit 1
        fi
        echo "✅ llmperf installed successfully."
    else
        echo "⏭️  LLMPerf not installed. Exiting."
        exit 1
    fi
fi

# 테스트 메타데이터 파일 생성
METADATA_FILE="${TEST_RUN_DIR}/test_metadata.json"
cat > $METADATA_FILE << EOF
{
  "test_run_id": "${TIMESTAMP}_${TEST_LEVEL}_${COMPILED_MODEL_PREFIX}",
  "timestamp": "$TIMESTAMP",
  "test_level": "$TEST_LEVEL",
  "tool": "llmperf",
  "model": "$MODEL_NAME",
  "model_path": "$MODEL_PATH",
  "tp_degree": $TP_DEGREE,
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tests": []
}
EOF

echo "=========================================="
echo "LLMPerf Benchmark"
echo "=========================================="
echo "Test Level: $TEST_LEVEL"
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL_NAME"
echo "TP Degree: $TP_DEGREE"
echo "Results Directory: $TEST_RUN_DIR"
echo "=========================================="
echo ""

# 테스트 레벨에 따른 설정 선택
case $TEST_LEVEL in
    "light")
        echo "📋 Light Test: 기본 성능 프로파일"
        CONCURRENCY_LEVELS=("${LLMPERF_LIGHT_CONCURRENCY[@]}")
        VARIATIONS=("${LLMPERF_LIGHT_VARIATIONS[@]}")
        ;;
    "medium")
        echo "📋 Medium Test: 상세 매트릭스"
        CONCURRENCY_LEVELS=("${LLMPERF_MEDIUM_CONCURRENCY[@]}")
        VARIATIONS=("${LLMPERF_MEDIUM_VARIATIONS[@]}")
        ;;
    "heavy")
        echo "📋 Heavy Test: 전체 스펙트럼"
        CONCURRENCY_LEVELS=("${LLMPERF_HEAVY_CONCURRENCY[@]}")
        VARIATIONS=("${LLMPERF_HEAVY_VARIATIONS[@]}")
        ;;
    *)
        echo "❌ Invalid test level. Use: light, medium, or heavy"
        exit 1
        ;;
esac

# 테스트 케이스 생성
declare -a TEST_CASES
for variation in "${VARIATIONS[@]}"; do
    IFS=' ' read -r mean_in std_in mean_out std_out <<< "$variation"
    for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
        TEST_CASES+=("$mean_in $std_in $mean_out $std_out $concurrency")
    done
done

echo "Total tests: ${#TEST_CASES[@]}"
echo ""

# 실패 로그 파일
FAILURE_LOG="${TEST_RUN_DIR}/failures.log"
echo "Test Failure Log - $(date)" > $FAILURE_LOG
echo "Test Level: $TEST_LEVEL" >> $FAILURE_LOG
echo "Model: $MODEL_NAME" >> $FAILURE_LOG
echo "Tool: LLMPerf" >> $FAILURE_LOG
echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
echo "========================================" >> $FAILURE_LOG
echo "" >> $FAILURE_LOG

# 성공/실패 카운터
SUCCESS_COUNT=0
FAILURE_COUNT=0

# 컴파일된 모델 찾기 (가장 큰 컨텍스트 길이 사용)
find_compiled_model() {
    local batch_size=$1
    local models=$(ls -d ${BASE_COMPILED_PATH}/${COMPILED_MODEL_PREFIX}-tp${TP_DEGREE}-bs${batch_size}-ctx* 2>/dev/null | sort -V -r)
    if [ -n "$models" ]; then
        echo "$models" | head -1
    else
        echo ""
    fi
}

# 테스트 실행 함수
run_llmperf_test() {
    local test_num=$1
    local mean_input=$2
    local std_input=$3
    local mean_output=$4
    local std_output=$5
    local concurrency=$6
    
    local test_name="${TIMESTAMP}_in${mean_input}_out${mean_output}_conc${concurrency}"
    local test_label="in${mean_input}±${std_input}_out${mean_output}±${std_output}_conc${concurrency}"
    
    echo ""
    echo "=========================================="
    echo "Test ${test_num}/${#TEST_CASES[@]}: $test_label"
    echo "=========================================="
    echo "Mean Input Tokens: $mean_input (±$std_input)"
    echo "Mean Output Tokens: $mean_output (±$std_output)"
    echo "Concurrency: $concurrency"
    echo "Max Requests: ${LLMPERF_MAX_REQUESTS}"
    echo "Timestamp: $TIMESTAMP"
    echo "=========================================="
    
    # 동시성에 따라 적절한 배치 사이즈 선택
    local batch_size=1
    if [ $concurrency -ge 16 ]; then
        batch_size=4
    elif [ $concurrency -ge 4 ]; then
        batch_size=2
    fi
    
    local compiled_path=$(find_compiled_model $batch_size)
    
    # Step 1: 기존 서버 종료
    echo "1. Stopping existing servers..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    
    # Step 2: 컴파일 확인
    if [ -z "$compiled_path" ]; then
        echo "2. ⚠️  No compiled model found for BS${batch_size}"
        echo "   Skipping this test"
        echo "" >> $FAILURE_LOG
        echo "Test: $test_label" >> $FAILURE_LOG
        echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
        echo "Reason: No compiled model found for BS${batch_size}" >> $FAILURE_LOG
        echo "Action: Run compile_model.sh first" >> $FAILURE_LOG
        echo "---" >> $FAILURE_LOG
        ((FAILURE_COUNT++))
        return 1
    fi
    
    echo "2. ✅ Using compiled model: $compiled_path"
    
    # 컨텍스트 길이 추출
    local context_length=$(basename $compiled_path | grep -oP 'ctx\K[0-9]+')
    
    # 컴파일된 모델의 설정에서 seq_len을 직접 읽어옴
    local seq_len
    if [ -f "${compiled_path}/compile_config.json" ]; then
        seq_len=$(python3 -c "import json; print(json.load(open('${compiled_path}/compile_config.json'))['seq_len'])")
    else
        seq_len=$((context_length + 512)) # Fallback
    fi
    
    # Step 3: 서버 시작
    echo "3. Starting vLLM server..."
    export NEURON_COMPILED_ARTIFACTS=$compiled_path
    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    
    VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-100000} python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs $batch_size \
        --max-model-len $seq_len \
        --tensor-parallel-size $TP_DEGREE \
        --block-size ${VLLM_BLOCK_SIZE:-16} \
        --port 8000 \
        $VLLM_EXTRA_ARGS \
        > ${TEST_RUN_DIR}/server_${test_name}.log 2>&1 &
    
    local server_pid=$!
    
    # Step 4: 서버 준비 대기
    echo "4. Waiting for server..."
    local server_ready=false
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ Server ready!"
            server_ready=true
            break
        fi
        sleep 5
    done
    
    if [ "$server_ready" = false ]; then
        echo "   ❌ Server failed to start"
        kill $server_pid 2>/dev/null || true
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        
        echo "" >> $FAILURE_LOG
        echo "Test: $test_label" >> $FAILURE_LOG
        echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
        echo "Reason: Server failed to start" >> $FAILURE_LOG
        echo "Check log: ${TEST_RUN_DIR}/server_${test_name}.log" >> $FAILURE_LOG
        echo "---" >> $FAILURE_LOG
        ((FAILURE_COUNT++))
        return 1
    fi
    
    # Step 5: 모델 이름 확인
    echo "5. Checking model name..."
    local model_name=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "$MODEL_PATH")
    echo "   Model: $model_name"
    
    # Step 6: LLMPerf 실행
    echo "6. Running LLMPerf benchmark..."
    
    local benchmark_success=false
    local result_dir="${TEST_RUN_DIR}/llmperf_${test_name}"
    mkdir -p $result_dir
    
    # 파이프라인의 첫 번째 명령어의 실패를 감지하기 위해 pipefail 활성화
    set -o pipefail
    
    # llmperf의 ray client가 환경 변수를 읽을 수 있도록 export
    export OPENAI_API_BASE=${LLMPERF_API_BASE}
    export OPENAI_API_KEY="dummy"

    if PYTHONPATH="$LLMPERF_DIR:$PYTHONPATH" python3 "$LLMPERF_SCRIPT" \
        --model "$model_name" \
        --tokenizer "$MODEL_PATH" \
        --mean-input-tokens $mean_input \
        --stddev-input-tokens $std_input \
        --mean-output-tokens $mean_output \
        --stddev-output-tokens $std_output \
        --max-num-completed-requests ${LLMPERF_MAX_REQUESTS} \
        --timeout ${LLMPERF_TIMEOUT} \
        --num-concurrent-requests $concurrency \
        --results-dir "$result_dir" \
        --llm-api ${LLMPERF_API_TYPE} \
        --additional-sampling-params "${LLMPERF_SAMPLING_PARAMS}" 2>&1 | tee ${TEST_RUN_DIR}/llmperf_${test_name}.log; then
        
        echo "   ✅ Benchmark complete!"
        benchmark_success=true
        ((SUCCESS_COUNT++))
        
        # 결과 파일 찾기
        local result_file=$(find $result_dir -name "*.json" -type f | head -1)
        
        if [ -f "$result_file" ]; then
            # 결과 파싱
            local throughput=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results_output_token_throughput_per_s', 0))" 2>/dev/null || echo "null")
            local ttft_mean=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results_ttft_s_mean', 0)*1000)" 2>/dev/null || echo "null")
            local ttft_p50=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results_ttft_s_quantiles_p50', 0)*1000)" 2>/dev/null || echo "null")
            local ttft_p99=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results_ttft_s_quantiles_p99', 0)*1000)" 2>/dev/null || echo "null")
            local tpot_mean=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results_inter_token_latency_s_mean', 0)*1000)" 2>/dev/null || echo "null")
            
            # 개별 테스트 결과 JSON 생성
            cat > ${TEST_RUN_DIR}/result_${test_name}.json << EOF
{
  "test_name": "$test_label",
  "timestamp": "$TIMESTAMP",
  "status": "SUCCESS",
  "config": {
    "batch_size": $batch_size,
    "context_length": $context_length,
    "mean_input_tokens": $mean_input,
    "stddev_input_tokens": $std_input,
    "mean_output_tokens": $mean_output,
    "stddev_output_tokens": $std_output,
    "concurrency": $concurrency,
    "max_requests": ${LLMPERF_MAX_REQUESTS},
    "compiled_model_path": "$compiled_path"
  },
  "results": {
    "throughput_tokens_per_sec": $throughput,
    "mean_ttft_ms": $ttft_mean,
    "median_ttft_ms": $ttft_p50,
    "p99_ttft_ms": $ttft_p99,
    "mean_tpot_ms": $tpot_mean
  },
  "files": {
    "llmperf_result": "$(basename $result_file)",
    "llmperf_log": "llmperf_${test_name}.log",
    "server_log": "server_${test_name}.log"
  }
}
EOF
        fi
        
    else
        echo "   ❌ Benchmark failed!"
        
        cat > ${TEST_RUN_DIR}/result_${test_name}.json << EOF
{
  "test_name": "$test_label",
  "timestamp": "$TIMESTAMP",
  "status": "FAILED",
  "config": {
    "batch_size": $batch_size,
    "mean_input_tokens": $mean_input,
    "stddev_input_tokens": $std_input,
    "mean_output_tokens": $mean_output,
    "stddev_output_tokens": $std_output,
    "concurrency": $concurrency,
    "compiled_model_path": "$compiled_path"
  },
  "error": "LLMPerf execution failed",
  "files": {
    "llmperf_log": "llmperf_${test_name}.log",
    "server_log": "server_${test_name}.log"
  }
}
EOF
        
        echo "" >> $FAILURE_LOG
        echo "Test: $test_label" >> $FAILURE_LOG
        echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
        echo "Reason: LLMPerf execution failed" >> $FAILURE_LOG
        echo "Check log: ${TEST_RUN_DIR}/llmperf_${test_name}.log" >> $FAILURE_LOG
        echo "---" >> $FAILURE_LOG
        ((FAILURE_COUNT++))
    fi
    
    # pipefail을 다시 비활성화하여 다른 스크립트 부분에 영향을 주지 않도록 함
    set +o pipefail
    
    # Export된 환경 변수 unset
    unset OPENAI_API_BASE
    unset OPENAI_API_KEY

    # Step 7: 서버 종료
    echo "7. Stopping server..."
    kill $server_pid 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    
    if [ "$benchmark_success" = true ]; then
        echo "✅ Test $test_label PASSED"
        return 0
    else
        echo "❌ Test $test_label FAILED"
        return 1
    fi
}

# 모든 테스트 실행
test_num=0
for test_case in "${TEST_CASES[@]}"; do
    ((test_num++))
    IFS=' ' read -r mean_input std_input mean_output std_output concurrency <<< "$test_case"
    
    run_llmperf_test "$test_num" "$mean_input" "$std_input" "$mean_output" "$std_output" "$concurrency"
    
    echo ""
    echo "Progress: ${test_num}/${#TEST_CASES[@]} tests completed"
    echo "Success: $SUCCESS_COUNT | Failed: $FAILURE_COUNT"
    echo ""
done

# 최종 결과 요약
echo ""
echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
echo "Total: ${#TEST_CASES[@]} tests"
echo "✅ Success: $SUCCESS_COUNT"
echo "❌ Failed: $FAILURE_COUNT"
echo "=========================================="
echo ""

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "⚠️  Some tests failed. Check failure log:"
    echo "   $FAILURE_LOG"
    echo ""
fi

# CSV 요약 생성
CSV_FILE="${TEST_RUN_DIR}/summary_${TEST_LEVEL}.csv"
echo "Test,Mean_Input,Stddev_Input,Mean_Output,Stddev_Output,Concurrency,Throughput,Mean_TTFT,Median_TTFT,P99_TTFT,Mean_TPOT,Status" > $CSV_FILE

for test_case in "${TEST_CASES[@]}"; do
    IFS=' ' read -r mean_input std_input mean_output std_output concurrency <<< "$test_case"
    test_name="${TIMESTAMP}_in${mean_input}_out${mean_output}_conc${concurrency}"
    test_label="in${mean_input}±${std_input}_out${mean_output}±${std_output}_conc${concurrency}"
    
    if [ -f "${TEST_RUN_DIR}/result_${test_name}.json" ]; then
        throughput=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data['results'].get('throughput_tokens_per_sec', 'N/A'))" 2>/dev/null || echo "N/A")
        ttft_mean=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data['results'].get('mean_ttft_ms', 'N/A'))" 2>/dev/null || echo "N/A")
        ttft_p50=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data['results'].get('median_ttft_ms', 'N/A'))" 2>/dev/null || echo "N/A")
        ttft_p99=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data['results'].get('p99_ttft_ms', 'N/A'))" 2>/dev/null || echo "N/A")
        tpot_mean=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data['results'].get('mean_tpot_ms', 'N/A'))" 2>/dev/null || echo "N/A")
        status=$(python3 -c "import json; data=json.load(open('${TEST_RUN_DIR}/result_${test_name}.json')); print(data.get('status', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        
        echo "$test_label,$mean_input,$std_input,$mean_output,$std_output,$concurrency,$throughput,$ttft_mean,$ttft_p50,$ttft_p99,$tpot_mean,$status" >> $CSV_FILE
    else
        echo "$test_label,$mean_input,$std_input,$mean_output,$std_output,$concurrency,N/A,N/A,N/A,N/A,N/A,FAILED" >> $CSV_FILE
    fi
done

echo ""
echo "📊 Summary CSV created: $CSV_FILE"
echo ""

# 최종 메타데이터 업데이트
export TEST_RUN_DIR
python3 << PYTHON_SCRIPT
import json
import glob
import os
from datetime import datetime

test_run_dir = os.environ.get('TEST_RUN_DIR')
if not test_run_dir:
    print("❌ Error: TEST_RUN_DIR not set")
    exit(1)

metadata_file = os.path.join(test_run_dir, 'test_metadata.json')

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

result_files = glob.glob(os.path.join(test_run_dir, 'result_*.json'))
for result_file in sorted(result_files):
    with open(result_file, 'r') as f:
        test_result = json.load(f)
        metadata['tests'].append(test_result)

metadata['end_time'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
metadata['total_tests'] = len(metadata['tests'])
metadata['successful_tests'] = sum(1 for t in metadata['tests'] if t['status'] == 'SUCCESS')
metadata['failed_tests'] = sum(1 for t in metadata['tests'] if t['status'] == 'FAILED')

with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata updated: {metadata_file}")
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "📊 All results ready!"
echo "=========================================="
echo "Results directory: $TEST_RUN_DIR"
echo ""
echo "Generate reports:"
echo "  Text:  python3 ../reports/generate_report.py $TEST_RUN_DIR"
echo "  HTML:  python3 ../reports/generate_html_report.py $TEST_RUN_DIR"
echo "=========================================="
echo ""

exit 0

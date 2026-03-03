#!/bin/bash

# 설정 파일 기반 벤치마크 실행 스크립트
# 사용법: ./run_benchmark.sh <config_file> [light|medium|heavy]
# 예시: ./run_benchmark.sh ../configs/llama31-70b.conf light

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
RESULTS_DIR="./benchmark_results"
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

# 테스트 메타데이터 파일 생성
METADATA_FILE="${TEST_RUN_DIR}/test_metadata.json"
cat > $METADATA_FILE << EOF
{
  "test_run_id": "${TIMESTAMP}_${TEST_LEVEL}_${COMPILED_MODEL_PREFIX}",
  "timestamp": "$TIMESTAMP",
  "test_level": "$TEST_LEVEL",
  "model": "$MODEL_NAME",
  "model_path": "$MODEL_PATH",
  "tp_degree": $TP_DEGREE,
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tests": []
}
EOF

echo "=========================================="
echo "Comprehensive Performance Benchmark"
echo "=========================================="
echo "Test Level: $TEST_LEVEL"
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL_NAME"
echo "TP Degree: $TP_DEGREE"
echo "Results Directory: $TEST_RUN_DIR"
echo "=========================================="
echo ""

# 테스트 케이스 정의
declare -a TEST_CASES

# 설정에서 컨텍스트 길이 추출
IFS=' ' read -r bs1_ctx bs1_seq <<< "$BS1_CONFIG"
IFS=' ' read -r bs2_ctx bs2_seq <<< "$BS2_CONFIG"
IFS=' ' read -r bs4_ctx bs4_seq <<< "$BS4_CONFIG"

if [ "$TEST_LEVEL" == "light" ]; then
    echo "📋 Light Test: 기본 성능 프로파일 (6개 테스트)"
    TEST_CASES=(
        "1 $bs1_ctx 512 128 1"
        "1 $bs1_ctx 2048 512 1"
        "2 $bs2_ctx 1024 256 2"
        "2 $bs2_ctx 2048 512 4"
        "4 $bs4_ctx 512 128 4"
        "4 $bs4_ctx 1024 256 8"
    )
    
elif [ "$TEST_LEVEL" == "medium" ]; then
    echo "📋 Medium Test: 상세 매트릭스 (15개 테스트)"
    TEST_CASES=(
        "1 $bs1_ctx 512 128 1"
        "1 $bs1_ctx 1024 256 1"
        "1 $bs1_ctx 2048 512 1"
        "1 $bs1_ctx 4096 1024 2"
        "1 $bs1_ctx $((bs1_ctx/2)) 1024 4"
        
        "2 $bs2_ctx 512 128 2"
        "2 $bs2_ctx 1024 256 2"
        "2 $bs2_ctx 2048 512 4"
        "2 $bs2_ctx 4096 1024 8"
        
        "4 $bs4_ctx 512 128 4"
        "4 $bs4_ctx 1024 256 4"
        "4 $bs4_ctx 2048 512 8"
        "4 $bs4_ctx 2048 512 12"
        "4 $bs4_ctx 2048 512 16"
        
        "1 $bs1_ctx $((bs1_ctx-1024)) 2048 1"
    )
    
elif [ "$TEST_LEVEL" == "heavy" ]; then
    echo "📋 Heavy Test: 전체 매트릭스 (30개 테스트)"
    TEST_CASES=(
        "1 $bs1_ctx 256 64 1"
        "1 $bs1_ctx 512 128 1"
        "1 $bs1_ctx 1024 256 1"
        "1 $bs1_ctx 2048 512 1"
        "1 $bs1_ctx 4096 1024 1"
        "1 $bs1_ctx $((bs1_ctx/2)) 1024 2"
        "1 $bs1_ctx $((bs1_ctx*2/3)) 2048 2"
        "1 $bs1_ctx $((bs1_ctx-1024)) 2048 4"
        
        "2 $bs2_ctx 256 64 2"
        "2 $bs2_ctx 512 128 2"
        "2 $bs2_ctx 1024 256 2"
        "2 $bs2_ctx 2048 512 2"
        "2 $bs2_ctx 4096 1024 4"
        "2 $bs2_ctx $((bs2_ctx/2)) 1024 8"
        "2 $bs2_ctx $((bs2_ctx/2)) 1024 12"
        
        "4 $bs4_ctx 256 64 4"
        "4 $bs4_ctx 512 128 4"
        "4 $bs4_ctx 1024 256 4"
        "4 $bs4_ctx 2048 512 4"
        "4 $bs4_ctx 2048 512 8"
        "4 $bs4_ctx 2048 512 12"
        "4 $bs4_ctx 2048 512 16"
        "4 $bs4_ctx 3072 512 8"
        "4 $bs4_ctx 3072 512 16"
        
        "1 $bs1_ctx $((bs1_ctx-512)) 512 1"
        "2 $bs2_ctx $((bs2_ctx-512)) 512 2"
        "4 $bs4_ctx $((bs4_ctx-256)) 256 4"
        
        "1 $bs1_ctx 4096 1024 8"
        "2 $bs2_ctx 2048 512 16"
        "4 $bs4_ctx 1024 256 16"
    )
else
    echo "❌ Invalid test level. Use: light, medium, or heavy"
    exit 1
fi

echo "Total tests: ${#TEST_CASES[@]}"
echo ""

# 실패 로그 파일
FAILURE_LOG="${TEST_RUN_DIR}/failures.log"
echo "Test Failure Log - $(date)" > $FAILURE_LOG
echo "Test Level: $TEST_LEVEL" >> $FAILURE_LOG
echo "Model: $MODEL_NAME" >> $FAILURE_LOG
echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
echo "========================================" >> $FAILURE_LOG
echo "" >> $FAILURE_LOG

# 성공/실패 카운터
SUCCESS_COUNT=0
FAILURE_COUNT=0

# 테스트 실행 함수
run_test() {
    local test_num=$1
    local batch_size=$2
    local context_length=$3
    local input_len=$4
    local output_len=$5
    local concurrency=$6
    
    local test_name="${TIMESTAMP}_bs${batch_size}_ctx${context_length}_in${input_len}_out${output_len}_conc${concurrency}"
    local test_label="bs${batch_size}_ctx${context_length}_in${input_len}_out${output_len}_conc${concurrency}"
    
    echo ""
    echo "=========================================="
    echo "Test ${test_num}/${#TEST_CASES[@]}: $test_label"
    echo "=========================================="
    echo "Batch Size: $batch_size"
    echo "Context Length: $context_length"
    echo "Input Length: $input_len"
    echo "Output Length: $output_len"
    echo "Concurrency: $concurrency"
    echo "Timestamp: $TIMESTAMP"
    echo "=========================================="
    
    # 컴파일 시 사용된 seq_len을 정확히 찾아서 사용
    local seq_len
    if [ "$batch_size" -eq 1 ] && [ "$context_length" -eq "$bs1_ctx" ]; then
        seq_len=$bs1_seq
    elif [ "$batch_size" -eq 2 ] && [ "$context_length" -eq "$bs2_ctx" ]; then
        seq_len=$bs2_seq
    elif [ "$batch_size" -eq 4 ] && [ "$context_length" -eq "$bs4_ctx" ]; then
        seq_len=$bs4_seq
    else
        # light/medium/heavy의 파생 테스트 케이스 (예: context_length/2)
        # 이 경우는 compile_model.sh의 계산 방식과 동일하게 유지
        seq_len=$((context_length + 512))
    fi

    local compiled_path="${BASE_COMPILED_PATH}/${COMPILED_MODEL_PREFIX}-tp${TP_DEGREE}-bs${batch_size}-ctx${context_length}"
    
    # Step 1: 기존 서버 종료
    echo "1. Stopping existing servers..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    
    # Step 2: 컴파일 확인
    if [ -d "$compiled_path" ]; then
        echo "2. ✅ Using existing compiled model: $compiled_path"
    else
        echo "2. ⚠️  Compiled model not found: $compiled_path"
        echo "   Skipping this test"
        echo "" >> $FAILURE_LOG
        echo "Test: $test_label" >> $FAILURE_LOG
        echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
        echo "Reason: Compiled model not found at $compiled_path" >> $FAILURE_LOG
        echo "Action: Run compile_model.sh first" >> $FAILURE_LOG
        echo "---" >> $FAILURE_LOG
        ((FAILURE_COUNT++))
        return 1
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
    
    # Step 6: 벤치마크 실행
    echo "6. Running benchmark..."
    
    local benchmark_success=false
    if vllm bench serve \
        --backend vllm \
        --base-url http://localhost:8000 \
        --endpoint /v1/completions \
        --model "$model_name" \
        --dataset-name random \
        --num-prompts 128 \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --max-concurrency $concurrency \
        --save-result \
        --result-dir ${TEST_RUN_DIR} \
        --result-filename benchmark_${test_name}.json 2>&1 | tee ${TEST_RUN_DIR}/benchmark_${test_name}.log; then
        
        echo "   ✅ Benchmark complete!"
        benchmark_success=true
        ((SUCCESS_COUNT++))
        
        # 벤치마크 결과 파싱
        local throughput=$(grep "Output token throughput" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $5}' | head -1)
        local ttft=$(grep "Mean TTFT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        local tpot=$(grep "Mean TPOT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        local p50_ttft=$(grep "Median TTFT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        local p99_ttft=$(grep "P99 TTFT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        
        # 개별 테스트 결과 JSON 생성
        cat > ${TEST_RUN_DIR}/result_${test_name}.json << EOF
{
  "test_name": "$test_label",
  "timestamp": "$TIMESTAMP",
  "status": "SUCCESS",
  "config": {
    "batch_size": $batch_size,
    "context_length": $context_length,
    "input_length": $input_len,
    "output_length": $output_len,
    "concurrency": $concurrency,
    "compiled_model_path": "$compiled_path"
  },
  "results": {
    "throughput_tokens_per_sec": ${throughput:-null},
    "mean_ttft_ms": ${ttft:-null},
    "mean_tpot_ms": ${tpot:-null},
    "median_ttft_ms": ${p50_ttft:-null},
    "p99_ttft_ms": ${p99_ttft:-null}
  },
  "files": {
    "benchmark_json": "benchmark_${test_name}.json",
    "benchmark_log": "benchmark_${test_name}.log",
    "server_log": "server_${test_name}.log"
  }
}
EOF
        
    else
        echo "   ❌ Benchmark failed!"
        
        cat > ${TEST_RUN_DIR}/result_${test_name}.json << EOF
{
  "test_name": "$test_label",
  "timestamp": "$TIMESTAMP",
  "status": "FAILED",
  "config": {
    "batch_size": $batch_size,
    "context_length": $context_length,
    "input_length": $input_len,
    "output_length": $output_len,
    "concurrency": $concurrency,
    "compiled_model_path": "$compiled_path"
  },
  "error": "Benchmark execution failed",
  "files": {
    "benchmark_log": "benchmark_${test_name}.log",
    "server_log": "server_${test_name}.log"
  }
}
EOF
        
        echo "" >> $FAILURE_LOG
        echo "Test: $test_label" >> $FAILURE_LOG
        echo "Timestamp: $TIMESTAMP" >> $FAILURE_LOG
        echo "Reason: Benchmark execution failed" >> $FAILURE_LOG
        echo "Check log: ${TEST_RUN_DIR}/benchmark_${test_name}.log" >> $FAILURE_LOG
        echo "---" >> $FAILURE_LOG
        ((FAILURE_COUNT++))
    fi
    
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
    IFS=' ' read -r batch_size context_length input_len output_len concurrency <<< "$test_case"
    
    run_test "$test_num" "$batch_size" "$context_length" "$input_len" "$output_len" "$concurrency"
    
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
echo "Test,Batch_Size,Context_Length,Input_Len,Output_Len,Concurrency,Throughput,TTFT,TPOT,Status" > $CSV_FILE

for test_case in "${TEST_CASES[@]}"; do
    IFS=' ' read -r batch_size context_length input_len output_len concurrency <<< "$test_case"
    test_name="${TIMESTAMP}_bs${batch_size}_ctx${context_length}_in${input_len}_out${output_len}_conc${concurrency}"
    test_label="bs${batch_size}_ctx${context_length}_in${input_len}_out${output_len}_conc${concurrency}"
    
    if [ -f "${TEST_RUN_DIR}/benchmark_${test_name}.log" ] && grep -q "Serving Benchmark Result" "${TEST_RUN_DIR}/benchmark_${test_name}.log"; then
        throughput=$(grep "Output token throughput" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $5}' | head -1)
        ttft=$(grep "Mean TTFT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        tpot=$(grep "Mean TPOT" ${TEST_RUN_DIR}/benchmark_${test_name}.log | awk '{print $4}' | head -1)
        echo "$test_label,$batch_size,$context_length,$input_len,$output_len,$concurrency,$throughput,$ttft,$tpot,SUCCESS" >> $CSV_FILE
    else
        echo "$test_label,$batch_size,$context_length,$input_len,$output_len,$concurrency,N/A,N/A,N/A,FAILED" >> $CSV_FILE
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

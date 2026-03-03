#!/bin/bash

# 설정 파일 기반 정확도 테스트 스크립트
# 사용법: ./run_accuracy.sh <config_file> [light|medium|heavy]
# 예시: ./run_accuracy.sh ../configs/llama31-70b.conf light

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
BASE_COMPILED_PATH="${BASE_COMPILED_PATH:-/data/compiled_models}" # 컴파일된 모델 경로 (결과 저장 경로 아님)
RESULTS_DIR="$(pwd)/accuracy_results" # RESULTS_DIR을 절대 경로로 설정
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

echo "=========================================="
echo "Accuracy Evaluation Suite"
echo "=========================================="
echo "Test Level: $TEST_LEVEL"
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL_NAME"
echo "TP Degree: $TP_DEGREE"
echo "Results Directory: $TEST_RUN_DIR"
echo "=========================================="
echo ""

# AWS Neuron Samples 리포지토리 확인 및 클론
NEURON_SAMPLES_DIR="${HOME}/aws-neuron-samples"
ACCURACY_SCRIPT="${NEURON_SAMPLES_DIR}/inference-benchmarking/accuracy.py"

if [ ! -f "$ACCURACY_SCRIPT" ]; then
    echo "📦 AWS Neuron Samples not found. Cloning repository..."
    
    if [ -d "$NEURON_SAMPLES_DIR" ]; then
        echo "   Directory exists but accuracy.py not found. Re-cloning..."
        rm -rf "$NEURON_SAMPLES_DIR"
    fi
    
    git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git "$NEURON_SAMPLES_DIR"
    
    if [ ! -f "$ACCURACY_SCRIPT" ]; then
        echo "❌ Failed to clone aws-neuron-samples or accuracy.py not found"
        echo "   Expected location: $ACCURACY_SCRIPT"
        exit 1
    fi
    
    echo "✅ AWS Neuron Samples cloned successfully"
    
    # 선택적 의존성 설치 (기존 환경 보호)
    echo ""
    echo "⚠️  Dependency Installation"
    echo "   The following packages may be needed:"
    echo "   - lm-eval (for accuracy evaluation)"
    echo "   - datasets, tiktoken (for data loading)"
    echo ""
    echo "   WARNING: requirements.txt includes torch, transformers, pydantic"
    echo "   which may conflict with your existing vLLM environment!"
    echo ""
    
    read -p "Install only safe dependencies (lm-eval, datasets, tiktoken)? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "📦 Installing safe dependencies..."
        pip install -q lm-eval datasets tiktoken openai psutil botocore 2>/dev/null || true
        echo "✅ Safe dependencies installed"
        echo ""
        echo "   Skipped: torch, transformers, pydantic, pyarrow"
        echo "   (using existing versions from vLLM environment)"
    else
        echo "⏭️  Skipping dependency installation"
        echo "   You can manually install later if needed:"
        echo "   pip install lm-eval datasets tiktoken"
    fi
else
    echo "✅ AWS Neuron Samples found: $NEURON_SAMPLES_DIR"
fi

echo ""

# 테스트 레벨에 따라 테스트 목록 선택
declare -a TEST_LIST

if [ "$TEST_LEVEL" == "light" ]; then
    TEST_LIST=("${ACCURACY_LIGHT_TESTS[@]}")
    echo "📋 Light Test: Quick validation (${#TEST_LIST[@]} datasets)"
elif [ "$TEST_LEVEL" == "medium" ]; then
    TEST_LIST=("${ACCURACY_MEDIUM_TESTS[@]}")
    echo "📋 Medium Test: Standard evaluation (${#TEST_LIST[@]} datasets)"
elif [ "$TEST_LEVEL" == "heavy" ]; then
    TEST_LIST=("${ACCURACY_HEAVY_TESTS[@]}")
    echo "📋 Heavy Test: Full evaluation (${#TEST_LIST[@]} datasets)"
else
    echo "❌ Invalid test level. Use: light, medium, or heavy"
    exit 1
fi

declare -a COMPILE_CONFIGS

# 설정에서 컨텍스트 길이 추출
IFS=' ' read -r bs1_ctx bs1_seq <<< "$BS1_CONFIG"
IFS=' ' read -r bs2_ctx bs2_seq <<< "$BS2_CONFIG"
IFS=' ' read -r bs4_ctx bs4_seq <<< "$BS4_CONFIG"

if [ "$TEST_LEVEL" == "light" ]; then
    COMPILE_CONFIGS=( "1 $BS1_CONFIG" "2 $BS2_CONFIG" "4 $BS4_CONFIG" )
elif [ "$TEST_LEVEL" == "medium" ]; then
    COMPILE_CONFIGS=(
        "1 $BS1_CONFIG" "1 $((bs1_ctx/2)) $((bs1_ctx/2 + 512))"
        "2 $BS2_CONFIG" "2 $((bs2_ctx/2)) $((bs2_ctx/2 + 512))"
        "4 $BS4_CONFIG" "4 $((bs4_ctx/2)) $((bs4_ctx/2 + 512))"
    )
elif [ "$TEST_LEVEL" == "heavy" ]; then
    COMPILE_CONFIGS=(
        "1 $BS1_CONFIG" "1 $((bs1_ctx/2)) $((bs1_ctx/2 + 512))" "1 $((bs1_ctx/4)) $((bs1_ctx/4 + 512))"
        "2 $BS2_CONFIG" "2 $((bs2_ctx/2)) $((bs2_ctx/2 + 512))" "2 $((bs2_ctx/4)) $((bs2_ctx/4 + 512))"
        "4 $BS4_CONFIG" "4 $((bs4_ctx/2)) $((bs4_ctx/2 + 512))" "4 $((bs4_ctx/4)) $((bs4_ctx/4 + 512))"
    )
fi

echo "   Will test against ${#COMPILE_CONFIGS[@]} compiled model(s)."
TOTAL_TESTS=$((${#COMPILE_CONFIGS[@]} * ${#TEST_LIST[@]}))
echo "Total tests to run: $TOTAL_TESTS"
echo ""

# Working directory를 accuracy.py가 있는 곳으로 변경
ORIGINAL_DIR=$(pwd)
cd "${NEURON_SAMPLES_DIR}/inference-benchmarking"

echo "Working directory: $(pwd)"
echo ""

# 성공/실패 카운터
SUCCESS_COUNT=0
FAILURE_COUNT=0

# 개별 테스트 실행 함수
run_accuracy_test() {
    local test_num=$1
    local dataset_spec=$2
    
    # dataset:limit 형식 파싱
    IFS=':' read -r dataset limit <<< "$dataset_spec"
    
    # BATCH_SIZE, CONTEXT_LENGTH는 외부 루프에서 설정된 전역 변수 사용
    local test_name="${TIMESTAMP}_bs${BATCH_SIZE}_ctx${CONTEXT_LENGTH}_${dataset}"
    
    echo ""
    echo "=========================================="
    echo "Test ${test_num}/${TOTAL_TESTS}: BS${BATCH_SIZE} CTX${CONTEXT_LENGTH} - $dataset"
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "Sample Limit: ${limit:-full dataset}"
    echo "Timestamp: $TIMESTAMP"
    echo "=========================================="
    
    # YAML 설정 파일 생성
    local config_file="${TEST_RUN_DIR}/config_${test_name}.yaml"
    
    cat > $config_file << EOF
server:
  name: "${MODEL_NAME}-accuracy-test"
  model_path: "$MODEL_PATH"
  model_s3_path: "not_used" # Provide a non-empty placeholder to avoid parsing errors
  max_seq_len: $SEQ_LEN
  context_encoding_len: $CONTEXT_LENGTH
  tp_degree: $TP_DEGREE
  n_vllm_threads: ${ACCURACY_N_VLLM_THREADS:-16}
  server_port: ${ACCURACY_SERVER_PORT:-8000}
  continuous_batch_size: $BATCH_SIZE
  compiled_model_path: "$COMPILED_MODEL_PATH"

test:
  accuracy:
    ${dataset}_test:
      client: "lm_eval"
      datasets: ["$dataset"]
      max_concurrent_requests: ${ACCURACY_MAX_CONCURRENT_REQUESTS:-1}
      timeout: ${ACCURACY_TIMEOUT:-3600}
      client_params:
        batch_size: ${ACCURACY_CLIENT_PARAMS_BATCH_SIZE:-1}
        num_fewshot: ${ACCURACY_CLIENT_PARAMS_NUM_FEW_SHOT:-5}
EOF

    # limit이 0이 아니면 추가
    if [ "$limit" != "0" ] && [ -n "$limit" ]; then
        echo "        limit: $limit" >> $config_file
    fi
    
    echo ""
    echo "1. Generated config: $config_file"
    # --- Start Debug ---
    # 생성된 설정 파일의 내용을 직접 출력하여 확인합니다.
    echo "--- Verifying generated config content ---"
    cat $config_file
    echo "----------------------------------------"
    # --- End Debug ---
    echo "2. Running accuracy test..."
    echo ""
    
    # 정확도 테스트 실행
    if python3 $ACCURACY_SCRIPT --config $config_file > ${TEST_RUN_DIR}/accuracy_${test_name}.log 2>&1; then
        echo "   ✅ Test complete!"
        
        # 결과 파싱 (lm-eval 출력에서 추출)
        local accuracy=$(grep -A 5 "$dataset" ${TEST_RUN_DIR}/accuracy_${test_name}.log | grep -E "acc|exact_match" | head -1 | awk '{print $NF}')
        
        # 개별 테스트 결과 JSON 생성
        cat > ${TEST_RUN_DIR}/result_${test_name}.json << RESULT_EOF
{
  "test_name": "$dataset",
  "timestamp": "$TIMESTAMP",
  "status": "SUCCESS",
  "config": {
    "dataset": "$dataset",
    "sample_limit": ${limit:-null},
    "batch_size": $BATCH_SIZE,
    "context_length": $CONTEXT_LENGTH,
    "tp_degree": $TP_DEGREE
  },
  "results": {
    "accuracy": ${accuracy:-null}
  },
  "files": {
    "config": "config_${test_name}.yaml",
    "log": "accuracy_${test_name}.log"
  }
}
RESULT_EOF
        
        ((SUCCESS_COUNT++))
        return 0
    else
        echo "   ❌ Test failed!"
        
        cat > ${TEST_RUN_DIR}/result_${test_name}.json << RESULT_EOF
{
  "test_name": "$dataset",
  "timestamp": "$TIMESTAMP",
  "status": "FAILED",
  "config": {
    "dataset": "$dataset",
    "sample_limit": ${limit:-null},
    "batch_size": $BATCH_SIZE,
    "context_length": $CONTEXT_LENGTH,
    "tp_degree": $TP_DEGREE
  },
  "error": "Test execution failed",
  "files": {
    "config": "config_${test_name}.yaml",
    "log": "accuracy_${test_name}.log"
  }
}
RESULT_EOF
        
        ((FAILURE_COUNT++))
        return 1
    fi
}

# 메타데이터 파일 초기화
METADATA_FILE="${TEST_RUN_DIR}/test_metadata.json"
cat > $METADATA_FILE << EOF
{
  "test_run_id": "${TIMESTAMP}_${TEST_LEVEL}_${COMPILED_MODEL_PREFIX}",
  "timestamp": "$TIMESTAMP",
  "test_level": "$TEST_LEVEL",
  "test_type": "accuracy",
  "model": "$MODEL_NAME",
  "model_path": "$MODEL_PATH",
  "tp_degree": $TP_DEGREE,
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tests": []
}
EOF

test_num=0
for compile_config in "${COMPILE_CONFIGS[@]}"; do
    IFS=' ' read -r BATCH_SIZE CONTEXT_LENGTH SEQ_LEN <<< "$compile_config"

    COMPILED_MODEL_PATH="${BASE_COMPILED_PATH}/${COMPILED_MODEL_PREFIX}-tp${TP_DEGREE}-bs${BATCH_SIZE}-ctx${CONTEXT_LENGTH}"

    if [ ! -d "$COMPILED_MODEL_PATH" ]; then
        echo ""
        echo "=========================================="
        echo "SKIPPING: Compiled model not found for BS${BATCH_SIZE} CTX${CONTEXT_LENGTH}"
        echo "Path: $COMPILED_MODEL_PATH"
        echo "=========================================="
        echo ""
        continue
    fi

    for test_spec in "${TEST_LIST[@]}"; do
        ((test_num++))
        run_accuracy_test "$test_num" "$test_spec"
        
        echo ""
        echo "Progress: ${test_num}/${TOTAL_TESTS} tests completed"
        echo "Success: $SUCCESS_COUNT | Failed: $FAILURE_COUNT"
        echo ""
    done
done

# 원래 디렉토리로 복귀
cd "$ORIGINAL_DIR"

# 최종 결과 요약
echo ""
echo "=========================================="
echo "All Accuracy Tests Complete!"
echo "=========================================="
echo "Total: $TOTAL_TESTS tests"
echo "✅ Success: $SUCCESS_COUNT"
echo "❌ Failed: $FAILURE_COUNT"
echo "=========================================="
echo ""

# CSV 요약 생성
CSV_FILE="${TEST_RUN_DIR}/summary_${TEST_LEVEL}.csv"
echo "Batch_Size,Context_Length,Dataset,Sample_Limit,Accuracy,Status" > $CSV_FILE

python3 << PYTHON_SCRIPT
import json
import os

test_run_dir = os.environ.get('TEST_RUN_DIR')
csv_file = os.path.join(test_run_dir, 'summary_${TEST_LEVEL}.csv')

with open(os.path.join(test_run_dir, 'test_metadata.json'), 'r') as f:
    metadata = json.load(f)

with open(csv_file, 'a') as f:
    for test in sorted(metadata['tests'], key=lambda x: (x['config']['batch_size'], x['config']['context_length'], x['test_name'])):
        bs = test['config']['batch_size']
        ctx = test['config']['context_length']
        dataset = test['config']['dataset']
        limit = test['config'].get('sample_limit', 'full')
        accuracy = test.get('results', {}).get('accuracy', 'N/A')
        status = test['status']
        f.write(f"{bs},{ctx},{dataset},{limit},{accuracy},{status}\n")
PYTHON_SCRIPT

echo "📊 Summary CSV created: $CSV_FILE"
echo ""

# 최종 메타데이터 업데이트
export TEST_RUN_DIR
python3 << 'PYTHON_SCRIPT'
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

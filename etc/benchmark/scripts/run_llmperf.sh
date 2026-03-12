#!/bin/bash

# LLMPerf를 사용하여 컴파일된 모든 모델 버전에 대한 성능을 측정하는 스크립트
# 사용법: ./run_llmperf.sh <config_file>
# 예시: ./run_llmperf.sh ../configs/qwen3-8b.conf

set -e

# 1. --- Argument and Config Validation ---
if [ -z "$1" ]; then
    echo "Error: Configuration file not provided."
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

source $CONFIG_FILE
echo "✅ Loaded configuration from $CONFIG_FILE"
echo

# 2. --- Setup Result Directory ---
RESULTS_BASE_DIR="/home/ubuntu/benchmark_result"

# 컨피그에 EXPERIMENT_NAME이 없으면 임시로 타임스탬프를 사용
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME=$(date +"%Y%m%d_%H%M%S")
    echo "⚠️ EXPERIMENT_NAME not set in config. Using timestamp: $EXPERIMENT_NAME"
fi

TEST_RUN_DIR="${RESULTS_BASE_DIR}/llmperf/${MODEL_NAME}/${EXPERIMENT_NAME}"
# NVIDIA 스크립트라면: TEST_RUN_DIR="${RESULTS_BASE_DIR}/llmperf_nvidia/${MODEL_NAME}/${EXPERIMENT_NAME}"

mkdir -p "$TEST_RUN_DIR"
echo "📊 Saving all results to: $TEST_RUN_DIR"
echo

# 3. --- Find Compiled Models ---
COMPILED_MODELS_BASE_PATH="/data/compiled_models/${MODEL_NAME}"
COMPILED_MODEL_PATHS=$(find "$COMPILED_MODELS_BASE_PATH" -mindepth 1 -maxdepth 1 -type d -name "*-tp*-bs*-ctx*" | sort -V)

if [ -z "$COMPILED_MODEL_PATHS" ]; then
    echo "❌ No compiled models found in $COMPILED_MODELS_BASE_PATH"
    echo "   Please run compile_model.sh first."
    exit 1
fi

# --- DEBUG: Print the discovered model paths ---
echo "--- DEBUG: Found compiled model paths ---"
echo "$COMPILED_MODEL_PATHS"
echo "-----------------------------------------"

# 4. --- Check for LLMPerf script ---
LLMPERF_DIR="llmperf"
LLMPERF_SCRIPT="${LLMPERF_DIR}/token_benchmark_ray.py"
if [ ! -f "$LLMPERF_SCRIPT" ]; then
    echo "❌ Error: llmperf script not found at '$LLMPERF_SCRIPT'"
    echo "   Please run 'setup.sh' or install 'llmperf' manually."
    exit 1
fi
echo "✅ Found llmperf script: $LLMPERF_SCRIPT"

# 5. --- Set Neuron Runtime Environment ---
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

# 6. --- Main Loop ---
TOTAL_MODELS=$(echo "$COMPILED_MODEL_PATHS" | wc -l)
TOTAL_CONCURRENCIES=$(echo "$LLMPERF_CONCURRENCIES" | wc -w)
TOTAL_VARIATIONS=${#LLMPERF_VARIATIONS[@]}
TOTAL_JOBS=$((TOTAL_MODELS * TOTAL_CONCURRENCIES * TOTAL_VARIATIONS))
CURRENT_JOB=0

for compiled_path in $COMPILED_MODEL_PATHS; do
    
    # --- Parse parameters from directory name ---
    dir_name=$(basename "$compiled_path")
    tp=$(echo "$dir_name" | grep -oP 'tp\K[0-9]+')
    bs=$(echo "$dir_name" | grep -oP 'bs\K[0-9]+')
    ctx=$(echo "$dir_name" | grep -oP 'ctx\K[0-9]+')

    # --- Check for successful compilation marker ---
    SUCCESS_MARKER="${compiled_path}/.compile_success"
    if [ ! -f "$SUCCESS_MARKER" ]; then
        echo "================================================================="
        echo "⏭️  SKIPPING Model: $dir_name (Compilation failed or is incomplete)"
        echo "================================================================="
        echo
        continue
    fi

    echo "================================================================="
    echo "🚀 Benchmarking Model: $dir_name (TP=${tp}, BS=${bs}, CTX=${ctx})"
    echo "================================================================="

    # --- Pre-check if all tests for this model are already complete before starting the server ---
    all_tests_done=true
    # This loop must exactly mirror the main benchmark loop's structure
    for conc_check in $LLMPERF_CONCURRENCIES; do
        # Don't check for concurrencies higher than the batch size
        if [ "$conc_check" -gt "$bs" ]; then
            continue
        fi

        for variation_check in "${LLMPERF_VARIATIONS[@]}"; do
            IFS=' ' read -r mean_in_check std_in_check mean_out_check std_out_check <<< "$variation_check"
            
            test_name_check="conc${conc_check}_in${mean_in_check}_out${mean_out_check}"
            result_dir_check="${TEST_RUN_DIR}/${dir_name}/llmperf_${test_name_check}"

            if ! ls "${result_dir_check}"/*_summary.json 1> /dev/null 2>&1; then
                all_tests_done=false
                break 2 # Break out of both inner and outer loops, a test is missing
            fi
        done
    done

    if [ "$all_tests_done" = true ]; then
        echo "✅ All tests for model configuration '$dir_name' are already complete. Skipping server startup."
        echo
        continue
    fi

    # Create a subdirectory for this specific compiled model's results
    MODEL_RESULTS_DIR="${TEST_RUN_DIR}/${dir_name}"
    mkdir -p "$MODEL_RESULTS_DIR"

    # --- Start Server ---
    echo "1. Stopping any existing server..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3

    echo "2. Starting vLLM server for $dir_name..."
    export NEURON_COMPILED_ARTIFACTS=$compiled_path
    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    
    SERVER_LOG_FILE="${MODEL_RESULTS_DIR}/server.log"

    VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-100000} python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --trust-remote-code \
        --max-num-seqs "$bs" \
        --max-model-len "$SEQ_LEN" \
        --tensor-parallel-size "$tp" \
        --block-size 16 \
        --port 8000 \
        > "$SERVER_LOG_FILE" 2>&1 &
    
    server_pid=$!

    # --- Wait for Server ---
    echo "3. Waiting for server to be ready..."
    server_ready=false
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ Server ready!"
            server_ready=true
            break
        fi
        sleep 5
    done

    if [ "$server_ready" = false ]; then
        echo "   ❌ Server failed to start for $dir_name. Check log: $SERVER_LOG_FILE"
        kill $server_pid 2>/dev/null || true
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        continue # Skip to the next compiled model
    fi

    # =================================================================
    # --- [추가됨] 3.5. WARM-UP (Cold-Start & Lazy Loading 방지) ---
    # =================================================================
    echo "3.5 Warming up the model to initialize Neuron execution graphs..."
    
    # 컨피그의 첫 번째 VARIATION 설정을 가져와서 웜업에 사용 (현재 1024 0 256 0)
    first_variation="${LLMPERF_VARIATIONS[0]}"
    IFS=' ' read -r warm_in warm_std_in warm_out warm_std_out <<< "$first_variation"

    # Python을 이용해 설정된 토큰 길이에 맞는 더미 요청을 vLLM에 전송
    python3 -c "
import urllib.request
import urllib.error
import json
import time

# 단어 하나가 대략 1~1.5 토큰임을 감안하여 입력 길이(warm_in)만큼 텍스트 생성
# (정확한 1024 토큰이 아니어도, Neuron이 가장 가까운 1024 버킷을 로드하도록 유도함)
prompt_text = 'hello world ' * (int($warm_in) // 2)

data = json.dumps({
    'model': '$MODEL_NAME',
    'prompt': prompt_text,
    'max_tokens': int($warm_out),
    'temperature': 0.0
}).encode('utf-8')

req = urllib.request.Request(
    'http://localhost:8000/v1/completions',
    data=data,
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer dummy'}
)

try:
    print(f'   - Sending dummy request (Target Input: {$warm_in}, Target Output: {$warm_out})...')
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=300) as response:
        response.read()
    elapsed = time.time() - start_time
    print(f'   ✅ Warm-up complete! (Took {elapsed:.2f} seconds)')
except urllib.error.URLError as e:
    print(f'   ⚠️ Warm-up request failed, but continuing benchmark. Error: {e}')
"
    # =================================================================

    # --- Run Benchmarks ---
    for conc in $LLMPERF_CONCURRENCIES; do
        for variation in "${LLMPERF_VARIATIONS[@]}"; do
            CURRENT_JOB=$((CURRENT_JOB + 1))
            IFS=' ' read -r mean_in std_in mean_out std_out <<< "$variation"

            # Don't test concurrency higher than the compiled batch size
            if [ "$conc" -gt "$bs" ]; then
                echo "   ⏭️  Skipping concurrency $conc (greater than batch size $bs). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
                continue
            fi

            echo "-----------------------------------------------------------------"
            echo "   ▶️  Running test ${CURRENT_JOB}/${TOTAL_JOBS}: Concurrency=${conc}, Input=${mean_in}, Output=${mean_out}"
            echo "-----------------------------------------------------------------"

            test_name="conc${conc}_in${mean_in}_out${mean_out}"
            bench_log_file="${MODEL_RESULTS_DIR}/bench_${test_name}.log"
            result_dir="${MODEL_RESULTS_DIR}/llmperf_${test_name}"
            mkdir -p "$result_dir"

            # --- Check for existing successful benchmark result ---
            # Check for any file ending with _summary.json to mark completion.
            result_file=$(ls "${result_dir}"/*_summary.json 2>/dev/null | head -n 1)
            if [ -n "$result_file" ]; then
                echo "   ⏭️  SKIPPING test ${test_name} (result found: $(basename "$result_file")). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
                continue
            fi

            export OPENAI_API_BASE=${LLMPERF_API_BASE}
            export OPENAI_API_KEY="dummy"

            if (
                set -o pipefail
                PYTHONPATH="$LLMPERF_DIR:$PYTHONPATH" python3 "$LLMPERF_SCRIPT" \
                    --model "$MODEL_NAME" \
                    --mean-input-tokens "$mean_in" \
                    --stddev-input-tokens "$std_in" \
                    --mean-output-tokens "$mean_out" \
                    --stddev-output-tokens "$std_out" \
                    --max-num-completed-requests "${LLMPERF_MAX_REQUESTS}" \
                    --timeout "${LLMPERF_TIMEOUT}" \
                    --num-concurrent-requests "$conc" \
                    --results-dir "$result_dir" \
                    --llm-api "${LLMPERF_API_TYPE}" \
                    --additional-sampling-params "${LLMPERF_SAMPLING_PARAMS}" 2>&1 | tee "$bench_log_file"
            ); then
                # No need to rename, the visualization script can handle the default name.
                # The existence of the *_summary.json file is the success marker.
                echo "   ✅ SUCCESS: Concurrency=${conc}. Result saved in $result_dir"
            else
                echo "   ❌ FAILED: Concurrency=${conc}. See log." >&2
            fi

            unset OPENAI_API_BASE
            unset OPENAI_API_KEY
        done
    done

    # --- Stop Server ---
    echo "4. Stopping server..."
    kill $server_pid 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    echo
done

echo "================================================================="
echo "🎉 All llmperf jobs finished."
echo "   Results are in: $TEST_RUN_DIR"
echo "================================================================="

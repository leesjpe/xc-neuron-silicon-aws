#!/bin/bash

# vLLM 내장 벤치마크 도구를 사용하여 컴파일된 모든 모델 버전에 대한 성능을 측정하는 스크립트
# 사용법: ./run_vllm_bench.sh <config_file>
# 예시: ./run_vllm_bench.sh ../configs/qwen3-8b.conf

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
TEST_RUN_DIR="${RESULTS_BASE_DIR}/vllm/${MODEL_NAME}"
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

TOTAL_MODELS=$(echo "$COMPILED_MODEL_PATHS" | wc -l)
TOTAL_CONCURRENCIES=$(echo "$VLLM_BENCH_CONCURRENCIES" | wc -w)
TOTAL_JOBS=$((TOTAL_MODELS * TOTAL_CONCURRENCIES))
CURRENT_JOB=0

# 4. --- Set Neuron Runtime Environment ---
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

# 5. --- Main Loop ---
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
            echo "   ✅ Server is ready!"
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

    # --- Run Benchmarks for different concurrencies ---
    for conc in $VLLM_BENCH_CONCURRENCIES; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        # Don't test concurrency higher than the compiled batch size
        if [ "$conc" -gt "$bs" ]; then
            echo "   ⏭️  Skipping concurrency $conc (greater than batch size $bs). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
            continue
        fi

        echo "-----------------------------------------------------------------"
        echo "   ▶️  Running test ${CURRENT_JOB}/${TOTAL_JOBS}: Concurrency=${conc}"
        echo "-----------------------------------------------------------------"

        test_name="conc${conc}"
        bench_log_file="${MODEL_RESULTS_DIR}/bench_${test_name}.log"
        result_json_file="${MODEL_RESULTS_DIR}/bench_${test_name}.json"

        # --- Check for existing successful benchmark result ---
        if [ -f "$result_json_file" ]; then
            echo "   ⏭️  SKIPPING concurrency $conc (result file already exists). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
            continue
        fi
        
        if (
            set -o pipefail
            python -m vllm.bench.entrypoints.openai.api_server_throughput \
                --model "$MODEL_PATH" \
                --tokenizer "$MODEL_PATH" \
                --dataset-name random \
                --num-prompts "$VLLM_BENCH_NUM_PROMPTS" \
                --input-len "$VLLM_BENCH_INPUT_LEN" \
                --output-len "$VLLM_BENCH_OUTPUT_LEN" \
                --concurrency "$conc" \
                --save-result \
                --result-dir "$MODEL_RESULTS_DIR" \
                --result-filename "bench_${test_name}.json" 2>&1 | tee "$bench_log_file"
        ); then
            echo "   ✅ SUCCESS: Concurrency=${conc}. Parsing results..."
            # --- Result Summary ---
            throughput=$(grep "Output token throughput:" "$bench_log_file" | awk '{print $4}')
            req_per_sec=$(grep "Request throughput:" "$bench_log_file" | awk '{print $3}')
            mean_ttft=$(grep "Mean TTFT:" "$bench_log_file" | awk '{print $3}')
            mean_tpot=$(grep "Mean TPOT:" "$bench_log_file" | awk '{print $3}')
            
            echo "      ----------------------------------------"
            echo "      📊 Summary (Concurrency: ${conc})"
            echo "      ----------------------------------------"
            echo "      Request Throughput : ${req_per_sec} req/s"
            echo "      Output Token T-put : ${throughput} tokens/s"
            echo "      Mean TTFT          : ${mean_ttft} s"
            echo "      Mean TPOT          : ${mean_tpot} s"
            echo "      ----------------------------------------"
        else
            echo "   ❌ FAILED: Concurrency=${conc}. See log." >&2
        fi
    done

    # --- Stop Server ---
    echo "4. Stopping server..."
    kill $server_pid 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    echo
done

echo "================================================================="
echo "🎉 All benchmark jobs finished."
echo "   Results are in: $TEST_RUN_DIR"
echo "================================================================="
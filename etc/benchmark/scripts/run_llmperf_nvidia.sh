#!/bin/bash

# LLMPerf를 사용하여 NVIDIA GPU 환경에서 모델 성능을 측정하는 스크립트
# 사용법: ./run_llmperf_nvidia.sh <config_file>
# 예시: ./run_llmperf_nvidia.sh ../configs/qwen3-8b_nvidia.conf

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
TEST_RUN_DIR="${RESULTS_BASE_DIR}/llmperf_nvidia/${MODEL_NAME}"
mkdir -p "$TEST_RUN_DIR"
echo "📊 Saving all results to: $TEST_RUN_DIR"
echo

# 3. --- Check for LLMPerf script ---
LLMPERF_DIR="llmperf"
LLMPERF_SCRIPT="${LLMPERF_DIR}/token_benchmark_ray.py"
if [ ! -f "$LLMPERF_SCRIPT" ]; then
    echo "❌ Error: llmperf script not found at '$LLMPERF_SCRIPT'"
    echo "   Please run 'setup.sh' or install 'llmperf' manually."
    exit 1
fi
echo "✅ Found llmperf script: $LLMPERF_SCRIPT"

# 4. --- Main Loop ---
TOTAL_CONCURRENCIES=$(echo "$LLMPERF_CONCURRENCIES" | wc -w)
TOTAL_VARIATIONS=$(echo "$LLMPERF_VARIATIONS" | wc -l)
TOTAL_JOBS=$(( $(echo "$TP_DEGREES" | wc -w) * $(echo "$BATCH_SIZES" | wc -w) * TOTAL_CONCURRENCIES * TOTAL_VARIATIONS ))
CURRENT_JOB=0

for tp in $TP_DEGREES; do
    for bs in $BATCH_SIZES; do

        dir_name="${MODEL_NAME}-tp${tp}-bs${bs}-ctx${SEQ_LEN}"

        echo "================================================================="
        echo "🚀 Benchmarking Model: $dir_name (TP=${tp}, BS=${bs}, CTX=${SEQ_LEN})"
        echo "================================================================="

        # Create a subdirectory for this specific model configuration's results
        MODEL_RESULTS_DIR="${TEST_RUN_DIR}/${dir_name}"
        mkdir -p "$MODEL_RESULTS_DIR"

        # --- Start Server ---
        echo "1. Stopping any existing server..."
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        sleep 3

        echo "2. Starting vLLM server for $dir_name..."
        SERVER_LOG_FILE="${MODEL_RESULTS_DIR}/server.log"

        # Standard vLLM server startup for NVIDIA GPUs
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --trust-remote-code \
            --max-num-seqs "$bs" \
            --max-model-len "$SEQ_LEN" \
            --tensor-parallel-size "$tp" \
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
            continue # Skip to the next configuration
        fi

        # --- Run Benchmarks ---
        for conc in $LLMPERF_CONCURRENCIES; do
            for variation in "${LLMPERF_VARIATIONS[@]}"; do
                CURRENT_JOB=$((CURRENT_JOB + 1))
                IFS=' ' read -r mean_in std_in mean_out std_out <<< "$variation"

                # Don't test concurrency higher than the max sequences
                if [ "$conc" -gt "$bs" ]; then
                    echo "   ⏭️  Skipping concurrency $conc (greater than max_num_seqs $bs). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
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
                result_file=$(ls "${result_dir}"/*_summary.json 2>/dev/null | head -n 1)
                if [ -n "$result_file" ]; then
                    echo "   ⏭️  SKIPPING test ${test_name} (result found: $(basename "$result_file")). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
                    continue
                fi

                export OPENAI_API_BASE=${LLMPERF_API_BASE:-"http://localhost:8000/v1"}
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
done

echo "================================================================="
echo "🎉 All llmperf jobs for NVIDIA finished."
echo "   Results are in: $TEST_RUN_DIR"
echo "================================================================="
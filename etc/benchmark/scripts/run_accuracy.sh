#!/bin/bash

# 컴파일된 모든 모델 버전에 대해 lm-eval 기반 정확도를 측정하는 스크립트
# 사용법: ./run_accuracy.sh <config_file>
# 예시: ./run_accuracy.sh ../configs/qwen3-8b.conf

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
TEST_RUN_DIR="${RESULTS_BASE_DIR}/accuracy/${MODEL_NAME}"
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

# 4. --- Check for lm-eval command ---
if ! command -v lm_eval &> /dev/null
then
    echo "❌ Error: 'lm_eval' command not found. Please make sure 'lm-eval-harness' is installed in your environment."
    exit 1
fi
echo "✅ Found lm-eval command"

# 5. --- Set Neuron Runtime Environment ---
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

# 6. --- Main Loop ---
TOTAL_MODELS=$(echo "$COMPILED_MODEL_PATHS" | wc -l)
TOTAL_DATASETS=$(echo "$ACCURACY_DATASETS" | wc -w)
TOTAL_JOBS=$((TOTAL_MODELS * TOTAL_DATASETS))
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
    echo "🎯 Testing Model: $dir_name (TP=${tp}, BS=${bs}, CTX=${ctx})"
    echo "================================================================="

    # Create a subdirectory for this specific compiled model's results
    MODEL_RESULTS_DIR="${TEST_RUN_DIR}/${dir_name}"
    mkdir -p "$MODEL_RESULTS_DIR"

    # --- Start Server ---
    echo "1. Stopping any existing server..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3

    SERVER_PORT=${ACCURACY_SERVER_PORT:-8000}
    echo "2. Starting vLLM server for $dir_name on port $SERVER_PORT..."
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
        --port "$SERVER_PORT" \
        > "$SERVER_LOG_FILE" 2>&1 &
    
    server_pid=$!

    # --- Wait for Server ---
    echo "3. Waiting for server to be ready..."
    server_ready=false
    for i in {1..60}; do
        if curl -s http://localhost:${SERVER_PORT}/health > /dev/null 2>&1; then
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

    for dataset_spec in $ACCURACY_DATASETS; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        # Parse "dataset:limit" format
        IFS=':' read -r dataset limit <<< "$dataset_spec"
        limit=${limit:-0} # Default limit to 0 if not specified

        echo "-----------------------------------------------------------------"
        echo "   ▶️  Running test ${CURRENT_JOB}/${TOTAL_JOBS}: Dataset=${dataset}, Limit=${limit:-no limit}"
        echo "-----------------------------------------------------------------"

        test_name="acc_${dataset}"
        test_log_file="${MODEL_RESULTS_DIR}/${test_name}.log"
        result_json_file="${MODEL_RESULTS_DIR}/result_${test_name}.json"
        detailed_result_json_file="${MODEL_RESULTS_DIR}/detailed_result_${test_name}.json"

        # --- Check for existing successful benchmark result ---
        if [ -f "$result_json_file" ]; then
            echo "   ⏭️  SKIPPING dataset $dataset (result file already exists). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
            continue
        fi
        
        API_BASE="http://localhost:${SERVER_PORT}/v1"
        MODEL_ARGS="model=${MODEL_NAME},base_url=${API_BASE},api_key=dummy"
        
        limit_arg=""
        if [ "$limit" -gt 0 ]; then
            limit_arg="--limit $limit"
        fi

        # --- Run Accuracy Test ---
        echo "   1. Running lm-eval (log: ${test_log_file})..."
        if (
            set -o pipefail
            lm_eval --model openai-completions \
                --model_args "$MODEL_ARGS" \
                --tasks "$dataset" \
                --batch_size "${ACCURACY_CLIENT_PARAMS_BATCH_SIZE:-1}" \
                --num_fewshot "${ACCURACY_CLIENT_PARAMS_NUM_FEW_SHOT:-0}" \
                $limit_arg \
                --output_path "$detailed_result_json_file" 2>&1 | tee "$test_log_file"
        ); then
            # Parse result from detailed JSON output
            accuracy_val=$(jq -r ".results[\"$dataset\"].acc // .results[\"$dataset\"].acc_norm // null" "$detailed_result_json_file")
            
            echo "   ✅ SUCCESS: Dataset=${dataset}. Accuracy = ${accuracy_val:-N/A}"
            
            # Write simplified summary result file
            cat > "$result_json_file" << EOF
{
  "model_config": "$dir_name",
  "dataset": "$dataset",
  "limit": ${limit:-null},
  "status": "SUCCESS",
  "accuracy": ${accuracy_val:-null}
}
EOF
        else
            echo "   ❌ FAILED: Dataset=${dataset}. See log: $test_log_file" >&2
            # Create a failure record
            cat > "$result_json_file" << EOF
{
  "model_config": "$dir_name",
  "dataset": "$dataset",
  "limit": ${limit:-null},
  "status": "FAILED",
  "accuracy": null
}
EOF
        fi
        echo
    done

    # --- Stop Server ---
    echo "4. Stopping server..."
    kill $server_pid 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    echo
done

echo "================================================================="
echo "🎉 All accuracy jobs finished."
echo "   Results are in: $TEST_RUN_DIR"
echo "================================================================="

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

# 4. --- Check for accuracy.py script ---
ACCURACY_SCRIPT_PATH="accuracy.py"
if [ ! -f "$ACCURACY_SCRIPT_PATH" ]; then
    echo "❌ Error: accuracy.py not found at '$ACCURACY_SCRIPT_PATH'"
    echo "   Please run 'setup.sh' or clone 'aws-neuron-samples' repository manually."
    exit 1
fi
echo "✅ Found accuracy script: $ACCURACY_SCRIPT_PATH"

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

    for dataset_spec in $ACCURACY_DATASETS; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        # Parse "dataset:limit" format
        IFS=':' read -r dataset limit <<< "$dataset_spec"

        echo "-----------------------------------------------------------------"
        echo "   ▶️  Running test ${CURRENT_JOB}/${TOTAL_JOBS}: Dataset=${dataset}, Limit=${limit}"
        echo "-----------------------------------------------------------------"

        # --- Generate YAML config dynamically ---
        test_name="acc_${dataset}"
        yaml_config_file="${MODEL_RESULTS_DIR}/${test_name}.yaml"
        test_log_file="${MODEL_RESULTS_DIR}/${test_name}.log"
        result_json_file="${MODEL_RESULTS_DIR}/result_${test_name}.json"

        # --- Check for existing successful benchmark result ---
        if [ -f "$result_json_file" ]; then
            echo "   ⏭️  SKIPPING dataset $dataset (result file already exists). Job ${CURRENT_JOB}/${TOTAL_JOBS}."
            continue
        fi

        cat > "$yaml_config_file" << EOF
server:
  name: "${dir_name}-accuracy-test"
  model_path: "$MODEL_PATH"
  model_s3_path: "not_used"
  max_seq_len: $SEQ_LEN
  context_encoding_len: $ctx
  tp_degree: $tp
  n_vllm_threads: ${ACCURACY_N_VLLM_THREADS:-16}
  server_port: ${ACCURACY_SERVER_PORT:-8000}
  continuous_batch_size: $bs
  compiled_model_path: "$compiled_path"

test:
  accuracy:
    ${dataset}_test:
      client: "lm_eval"
      datasets: ["$dataset"]
      max_concurrent_requests: ${ACCURACY_MAX_CONCURRENT_REQUESTS:-1}
      timeout: ${ACCURACY_TIMEOUT:-3600}
      client_params:
        batch_size: ${ACCURACY_CLIENT_PARAMS_BATCH_SIZE:-1}
        num_fewshot: ${ACCURACY_CLIENT_PARAMS_NUM_FEW_SHOT:-0}
EOF

        # Add limit to YAML only if it's greater than 0
        if [ "$limit" -gt 0 ]; then
            echo "        limit: $limit" >> "$yaml_config_file"
        fi

        echo "   1. Generated YAML config: $yaml_config_file"
        echo "   2. Running accuracy test (log: ${test_log_file})..."

        # --- Run Accuracy Test ---
        if (
            set -o pipefail
            # Change to the script's directory to ensure it finds its modules
            cd "$(dirname "$ACCURACY_SCRIPT_PATH")"
            python3 "$(basename "$ACCURACY_SCRIPT_PATH")" --config "$yaml_config_file" 2>&1 | tee "$test_log_file"
        ); then
            # Parse result from log
            accuracy_line=$(grep -A 5 "\"task\": \"$dataset\"" "$test_log_file" | grep -E '"acc"|"acc_norm"' | head -1)
            accuracy_val=$(echo "$accuracy_line" | awk -F': ' '{print $2}' | sed 's/,//')

            echo "   ✅ SUCCESS: Dataset=${dataset}. Accuracy = ${accuracy_val:-N/A}"
            
            # Write individual result file
            cat > "$result_json_file" << EOF
{
  "model_config": "$dir_name",
  "dataset": "$dataset",
  "limit": $limit,
  "status": "SUCCESS",
  "accuracy": ${accuracy_val:-null}
}
EOF
        else
            echo "   ❌ FAILED: Dataset=${dataset}. See log." >&2
        fi
        echo
    done
done

echo "================================================================="
echo "🎉 All accuracy jobs finished."
echo "   Results are in: $TEST_RUN_DIR"
echo "================================================================="

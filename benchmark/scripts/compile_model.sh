#!/bin/bash

# Exit on first error
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

# --- Setup Log Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_BASE_DIR="/home/ubuntu/benchmark_result/compilation"
TEST_RUN_LOG_DIR="${LOG_BASE_DIR}/${MODEL_NAME}/${TIMESTAMP}"
mkdir -p "$TEST_RUN_LOG_DIR"
SUMMARY_LOG_FILE="${TEST_RUN_LOG_DIR}/summary.log"
echo "📝 Saving all compilation logs to: $TEST_RUN_LOG_DIR"
echo "   Summary log: $SUMMARY_LOG_FILE"

# 2. --- Set Neuron Runtime Environment ---
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

# 3. --- Compilation Loop ---
TOTAL_JOBS=$(($(echo "$TP_DEGREES" | wc -w) * $(echo "$BATCH_SIZES" | wc -w)))
CURRENT_JOB=0

for tp in $TP_DEGREES; do
    for bs in $BATCH_SIZES; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        dir_name="${MODEL_NAME}-tp${tp}-bs${bs}-ctx${MAX_CONTEXT_LENGTH}"
        COMPILED_MODEL_DIR="/data/compiled_models/${MODEL_NAME}/${dir_name}"
        LOG_FILE="${TEST_RUN_LOG_DIR}/${dir_name}.log"
        SUCCESS_MARKER="${COMPILED_MODEL_DIR}/.compile_success"

        # --- Check for existing successful compilation ---
        # DEBUG: Print the exact path being checked for the success marker.
        echo "DEBUG: Checking for success marker at: [${SUCCESS_MARKER}]"
        if [ -f "$SUCCESS_MARKER" ]; then
            echo "================================================================="
            echo "✅ SKIPPING Job ${CURRENT_JOB}/${TOTAL_JOBS}: TP=${tp}, BS=${bs} (already compiled)"
            echo "[$(date)] SKIPPED: ${dir_name} (already compiled)" >> "$SUMMARY_LOG_FILE"
            echo "================================================================="
            echo
            continue
        fi
        echo "================================================================="
        echo "➡️  Starting Compilation Job ${CURRENT_JOB}/${TOTAL_JOBS}: TP=${tp}, BS=${bs}, CTX=${MAX_CONTEXT_LENGTH}"
        echo "================================================================="
        echo "Model: $MODEL_NAME"
        echo "Saving to: $COMPILED_MODEL_DIR"
        echo "Log file: $LOG_FILE"
        echo "-----------------------------------------------------------------"

        mkdir -p "$COMPILED_MODEL_DIR"

        if (
            set -o pipefail
            inference_demo \
                --model-type "$MODEL_TYPE" --task-type "$TASK_TYPE" run \
                --model-path "$MODEL_PATH" --compiled-model-path "$COMPILED_MODEL_DIR" \
                --torch-dtype "$TORCH_DTYPE" --tp-degree "$tp" --batch-size "$bs" \
                --max-context-length "$MAX_CONTEXT_LENGTH" --seq-len "$SEQ_LEN" \
                --pad-token-id "$PAD_TOKEN_ID" \
                --context-encoding-buckets $CONTEXT_BUCKETS \
                --token-generation-buckets $TOKEN_BUCKETS \
                $COMPILE_OPTS \
                --prompt "What is AWS Trainium?" 2>&1 | tee "$LOG_FILE"
        ); then
            echo "✅ SUCCESS: TP=${tp}, BS=${bs}"
            echo "[$(date)] SUCCESS: ${dir_name}" >> "$SUMMARY_LOG_FILE"
            touch "$SUCCESS_MARKER"
        else
            echo "❌ FAILED: TP=${tp}, BS=${bs}. See log: $LOG_FILE" >&2
            echo "[$(date)] FAILED:  ${dir_name} - See log at ${LOG_FILE}" >> "$SUMMARY_LOG_FILE"
            rm -f "$SUCCESS_MARKER" # Ensure marker is not present on failure
        fi
        echo
    done
done

echo "================================================================="
echo "🎉 All compilation jobs finished."

# Move the global metric store to the log directory for this run
if [ -f "global_metric_store.json" ]; then
    mv global_metric_store.json "$TEST_RUN_LOG_DIR/"
fi

echo "   Check summary log at: $SUMMARY_LOG_FILE"
echo "================================================================="
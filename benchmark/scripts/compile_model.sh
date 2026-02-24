#!/bin/bash

# 설정 파일 기반 모델 컴파일 스크립트
# 사용법: ./compile_model.sh <config_file> [light|medium|heavy]
# 예시: ./compile_model.sh ../configs/llama31-70b.conf light

set +e  # 에러 발생해도 계속 진행

# 인자 확인
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [compile_level]"
    echo "Example: $0 ../configs/llama31-70b.conf light"
    echo ""
    echo "Available configs:"
    ls -1 ../configs/*.conf 2>/dev/null || echo "  No config files found"
    exit 1
fi

CONFIG_FILE=$1
COMPILE_LEVEL=${2:-"light"}

# 설정 파일 로드
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
source $CONFIG_FILE

# 기본 경로 설정
BASE_COMPILED_PATH="${BASE_COMPILED_PATH:-/data/compiled_models}"
COMPILED_MODEL_PREFIX=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '.' '-')

# Neuron 런타임 설정
export NEURON_RT_VIRTUAL_CORE_SIZE=${NEURON_RT_VIRTUAL_CORE_SIZE:-2}
export NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-64}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1800}
export XLA_DENSE_GATHER_FACTOR=${XLA_DENSE_GATHER_FACTOR:-0}
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}

echo "=========================================="
echo "Model Compilation Script"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Model Type: $MODEL_TYPE"
echo "Compile Level: $COMPILE_LEVEL"
echo "TP Degree: $TP_DEGREE"
echo "Output: $BASE_COMPILED_PATH"
echo "=========================================="
echo ""

# 컴파일할 모델 정의
declare -a COMPILE_CONFIGS

if [ "$COMPILE_LEVEL" == "light" ]; then
    echo "📋 Light: 기본 3개 모델 컴파일"
    COMPILE_CONFIGS=(
        "1 $BS1_CONFIG"
        "2 $BS2_CONFIG"
        "4 $BS4_CONFIG"
    )
elif [ "$COMPILE_LEVEL" == "medium" ]; then
    echo "📋 Medium: 6개 모델 컴파일"
    # BS1의 컨텍스트 길이 추출
    IFS=' ' read -r bs1_ctx bs1_seq <<< "$BS1_CONFIG"
    IFS=' ' read -r bs2_ctx bs2_seq <<< "$BS2_CONFIG"
    IFS=' ' read -r bs4_ctx bs4_seq <<< "$BS4_CONFIG"
    
    COMPILE_CONFIGS=(
        "1 $BS1_CONFIG"
        "1 $((bs1_ctx/2)) $((bs1_ctx/2 + 512))"
        "2 $BS2_CONFIG"
        "2 $((bs2_ctx/2)) $((bs2_ctx/2 + 512))"
        "4 $BS4_CONFIG"
        "4 $((bs4_ctx/2)) $((bs4_ctx/2 + 512))"
    )
elif [ "$COMPILE_LEVEL" == "heavy" ]; then
    echo "📋 Heavy: 9개 모델 컴파일"
    IFS=' ' read -r bs1_ctx bs1_seq <<< "$BS1_CONFIG"
    IFS=' ' read -r bs2_ctx bs2_seq <<< "$BS2_CONFIG"
    IFS=' ' read -r bs4_ctx bs4_seq <<< "$BS4_CONFIG"
    
    COMPILE_CONFIGS=(
        "1 $BS1_CONFIG"
        "1 $((bs1_ctx/2)) $((bs1_ctx/2 + 512))"
        "1 $((bs1_ctx/4)) $((bs1_ctx/4 + 512))"
        "2 $BS2_CONFIG"
        "2 $((bs2_ctx/2)) $((bs2_ctx/2 + 512))"
        "2 $((bs2_ctx/4)) $((bs2_ctx/4 + 512))"
        "4 $BS4_CONFIG"
        "4 $((bs4_ctx/2)) $((bs4_ctx/2 + 512))"
        "4 $((bs4_ctx/4)) $((bs4_ctx/4 + 512))"
    )
else
    echo "❌ Invalid compile level. Use: light, medium, or heavy"
    exit 1
fi

echo "Total models to compile: ${#COMPILE_CONFIGS[@]}"
echo ""

# 컴파일 함수
compile_model() {
    local batch_size=$1
    local context_length=$2
    local seq_len=$3
    local compiled_path="${BASE_COMPILED_PATH}/${COMPILED_MODEL_PREFIX}-tp${TP_DEGREE}-bs${batch_size}-ctx${context_length}"
    
    echo ""
    echo "=========================================="
    echo "Compiling: TP${TP_DEGREE} BS${batch_size} CTX${context_length}"
    echo "=========================================="
    
    # 이미 존재하면 스킵
    if [ -d "$compiled_path" ]; then
        echo "✅ Already compiled: $compiled_path"
        echo "   Skipping..."
        return 0
    fi
    
    echo "🔧 Starting compilation..."
    echo "   Batch Size: $batch_size"
    echo "   Context Length: $context_length"
    echo "   Seq Length: $seq_len"
    echo "   Output Path: $compiled_path"
    echo ""
    echo "⏱️  This will take 20-60 minutes..."
    echo ""
    
    mkdir -p $compiled_path
    
    # 배치 크기별 버킷 선택
    if [ "$batch_size" -eq 4 ]; then
        CONTEXT_BUCKETS=$BS4_CONTEXT_BUCKETS
        TOKEN_BUCKETS=$BS4_TOKEN_BUCKETS
    elif [ "$batch_size" -eq 2 ]; then
        CONTEXT_BUCKETS=$BS2_CONTEXT_BUCKETS
        TOKEN_BUCKETS=$BS2_TOKEN_BUCKETS
    else
        CONTEXT_BUCKETS=$BS1_CONTEXT_BUCKETS
        TOKEN_BUCKETS=$BS1_TOKEN_BUCKETS
    fi
    
    echo "   Context Buckets: $CONTEXT_BUCKETS"
    echo "   Token Buckets: $TOKEN_BUCKETS"
    echo ""
    
    # 컴파일 실행
    if inference_demo \
        --model-type $MODEL_TYPE \
        --task-type $TASK_TYPE \
            run \
            --model-path $MODEL_PATH \
            --compiled-model-path $compiled_path \
            --torch-dtype $TORCH_DTYPE \
            --start_rank_id 0 \
            --local_ranks_size $TP_DEGREE \
            --tp-degree $TP_DEGREE \
            --batch-size $batch_size \
            --max-context-length $context_length \
            --seq-len $seq_len \
            --pad-token-id $PAD_TOKEN_ID \
            --context-encoding-buckets $CONTEXT_BUCKETS \
            --token-generation-buckets $TOKEN_BUCKETS \
            $COMPILE_OPTS \
            --prompt "What is AWS Trainium?" 2>&1 | tee ${compiled_path}/compile.log; then
        
        echo ""
        echo "✅ Compilation successful!"
        
        # 컴파일 설정 저장
        cat > ${compiled_path}/compile_config.json << EOF
{
    "model": "$MODEL_NAME",
    "model_path": "$MODEL_PATH",
    "model_type": "$MODEL_TYPE",
    "compiled_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "tp_degree": $TP_DEGREE,
    "batch_size": $batch_size,
    "max_context_length": $context_length,
    "seq_len": $seq_len,
    "torch_dtype": "$TORCH_DTYPE",
    "pad_token_id": $PAD_TOKEN_ID,
    "context_buckets": "$CONTEXT_BUCKETS",
    "token_buckets": "$TOKEN_BUCKETS"
}
EOF
        return 0
    else
        echo ""
        echo "❌ Compilation failed!"
        echo "   Check log: ${compiled_path}/compile.log"
        return 1
    fi
}

# 모든 모델 컴파일
SUCCESS_COUNT=0
FAILURE_COUNT=0
SKIPPED_COUNT=0

for config in "${COMPILE_CONFIGS[@]}"; do
    IFS=' ' read -r batch_size context_length seq_len <<< "$config"
    
    compiled_path="${BASE_COMPILED_PATH}/${COMPILED_MODEL_PREFIX}-tp${TP_DEGREE}-bs${batch_size}-ctx${context_length}"
    
    if [ -d "$compiled_path" ]; then
        echo ""
        echo "=========================================="
        echo "TP${TP_DEGREE} BS${batch_size} CTX${context_length}: Already exists"
        echo "=========================================="
        echo "✅ Skipping: $compiled_path"
        ((SKIPPED_COUNT++))
        continue
    fi
    
    if compile_model "$batch_size" "$context_length" "$seq_len"; then
        ((SUCCESS_COUNT++))
    else
        ((FAILURE_COUNT++))
    fi
done

# 최종 요약
echo ""
echo "=========================================="
echo "Compilation Complete!"
echo "=========================================="
echo "✅ Successful: $SUCCESS_COUNT"
echo "⏭️  Skipped: $SKIPPED_COUNT"
echo "❌ Failed: $FAILURE_COUNT"
echo "=========================================="
echo ""

# 컴파일된 모델 목록
echo "📁 Compiled Models:"
ls -lh $BASE_COMPILED_PATH/ | grep "$COMPILED_MODEL_PREFIX"
echo ""

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "⚠️  Some compilations failed. Check logs in compiled model directories."
    exit 1
fi

echo "✅ All compilations successful!"
echo "   You can now run: ./run_benchmark.sh $CONFIG_FILE light"
echo ""

exit 0

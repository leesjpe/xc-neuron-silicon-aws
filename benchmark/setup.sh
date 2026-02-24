#!/bin/bash

# Benchmark Suite 설정 스크립트
# 사용법: ./setup.sh

set -e

echo "=========================================="
echo "Benchmark Suite Setup"
echo "=========================================="
echo ""

# 1. 가상환경 확인
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Please activate Neuron virtual environment first:"
    echo "   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Virtual environment: $VIRTUAL_ENV"
fi

echo ""

# 2. AWS Neuron Samples 클론
NEURON_SAMPLES_DIR="${HOME}/aws-neuron-samples"

if [ -d "$NEURON_SAMPLES_DIR" ]; then
    echo "✅ AWS Neuron Samples already exists: $NEURON_SAMPLES_DIR"
    
    # accuracy.py 확인
    if [ ! -f "${NEURON_SAMPLES_DIR}/inference-benchmarking/accuracy.py" ]; then
        echo "⚠️  accuracy.py not found. Re-cloning..."
        rm -rf "$NEURON_SAMPLES_DIR"
    else
        echo "✅ accuracy.py found"
        
        # 업데이트 확인
        read -p "Update aws-neuron-samples? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "📦 Updating aws-neuron-samples..."
            cd "$NEURON_SAMPLES_DIR"
            git pull
            cd -
        fi
    fi
fi

if [ ! -d "$NEURON_SAMPLES_DIR" ]; then
    echo "📦 Cloning aws-neuron-samples..."
    git clone --depth 1 https://github.com/aws-neuron/aws-neuron-samples.git "$NEURON_SAMPLES_DIR"
    
    if [ ! -f "${NEURON_SAMPLES_DIR}/inference-benchmarking/accuracy.py" ]; then
        echo "❌ Failed to clone or accuracy.py not found"
        exit 1
    fi
    
    echo "✅ AWS Neuron Samples cloned successfully"
fi

echo ""

# 3. 의존성 설치
REQUIREMENTS_FILE="${NEURON_SAMPLES_DIR}/inference-benchmarking/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "=========================================="
    echo "Dependency Installation"
    echo "=========================================="
    echo ""
    echo "⚠️  WARNING: Potential Version Conflicts"
    echo ""
    echo "The requirements.txt includes:"
    echo "  - torch (may conflict with Neuron torch)"
    echo "  - transformers (may conflict with existing version)"
    echo "  - pydantic>2.10 (may conflict with vLLM)"
    echo "  - pyarrow==20.0.0 (specific version)"
    echo ""
    echo "Recommended: Install only safe dependencies"
    echo "  - lm-eval (required for accuracy tests)"
    echo "  - datasets, tiktoken (data loading)"
    echo "  - openai, psutil, botocore (utilities)"
    echo ""
    
    echo "Options:"
    echo "  1) Install safe dependencies only (recommended)"
    echo "  2) Install all from requirements.txt (may break vLLM)"
    echo "  3) Skip installation (manual install later)"
    echo ""
    
    read -p "Choose option (1-3) [1]: " -n 1 -r
    echo
    REPLY=${REPLY:-1}
    
    if [ "$REPLY" = "1" ]; then
        echo "📦 Installing safe dependencies..."
        pip install -q lm-eval datasets tiktoken openai psutil botocore
        echo "✅ Safe dependencies installed"
        echo ""
        echo "Skipped packages (using existing versions):"
        echo "  - torch (using Neuron torch)"
        echo "  - transformers (using existing)"
        echo "  - pydantic (using existing)"
        echo "  - pyarrow (using existing)"
    elif [ "$REPLY" = "2" ]; then
        echo "📦 Installing all dependencies from requirements.txt..."
        echo "⚠️  This may cause version conflicts!"
        pip install -r "$REQUIREMENTS_FILE"
        echo "✅ All dependencies installed"
    else
        echo "⏭️  Skipping dependency installation"
        echo ""
        echo "To install manually later:"
        echo "  pip install lm-eval datasets tiktoken"
    fi
else
    echo "⚠️  requirements.txt not found: $REQUIREMENTS_FILE"
fi

echo ""

# 4. 스크립트 실행 권한 설정
echo "🔧 Setting execute permissions on scripts..."
chmod +x scripts/*.sh
echo "✅ Execute permissions set"

echo ""

# 5. 디렉토리 생성
echo "📁 Creating result directories..."
mkdir -p scripts/benchmark_results
mkdir -p scripts/accuracy_results
mkdir -p scripts/llmperf_results
echo "✅ Directories created"

echo ""

# 6. 환경 확인
echo "=========================================="
echo "Environment Check"
echo "=========================================="

# Python 버전
echo "Python: $(python3 --version)"

# Neuron 도구 확인
if command -v neuron-top &> /dev/null; then
    echo "✅ neuron-top: $(neuron-top --version 2>&1 | head -1)"
else
    echo "⚠️  neuron-top not found"
fi

# vLLM 확인
if python3 -c "import vllm" 2>/dev/null; then
    echo "✅ vLLM installed"
else
    echo "⚠️  vLLM not found"
fi

# lm-eval 확인
if python3 -c "import lm_eval" 2>/dev/null; then
    echo "✅ lm-eval installed"
else
    echo "⚠️  lm-eval not found (will be installed from requirements.txt)"
fi

# Git 확인
if command -v git &> /dev/null; then
    echo "✅ Git: $(git --version)"
else
    echo "❌ Git not found (required)"
fi

echo ""

# 7. 설정 파일 확인
echo "=========================================="
echo "Configuration Files"
echo "=========================================="

CONFIG_COUNT=$(ls -1 configs/*.conf 2>/dev/null | wc -l)
echo "Found $CONFIG_COUNT config file(s):"
ls -1 configs/*.conf 2>/dev/null || echo "  No config files found"

echo ""

# 8. 완료
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review config files in configs/"
echo "  2. Compile models: cd scripts && ./compile_model.sh ../configs/llama31-70b.conf light"
echo "  3. Run benchmarks: ./run_benchmark.sh ../configs/llama31-70b.conf light"
echo "  4. Run accuracy tests: ./run_accuracy.sh ../configs/llama31-70b.conf light"
echo ""
echo "For more information, see:"
echo "  - README.md: Detailed documentation"
echo "  - QUICKSTART.md: Quick start guide"
echo ""

exit 0

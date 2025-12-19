# Fine-tuning Llama 3 8B on AWS Trainium2 & Serving with vLLM

This repository provides a step-by-step guide to fine-tuning **Llama 3 8B** using **LoRA (Low-Rank Adaptation)** and **Tensor Parallelism** on **AWS Trainium2 (Trn2)** instances. It also covers how to serve the fine-tuned model using **vLLM** on Neuron.

## Prerequisites

* **Instance:** AWS EC2 `trn2.48xlarge` (or similar Trn2 instance)
* **AMI:** AWS Deep Learning AMI (Neuron)
* **Software:** Neuron SDK installed (PyTorch environment)

---

## 1. Environment Setup

### 1.1. Activate Neuron Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/activate

```

### 1.2. Download Training Scripts

Download the necessary scripts from `aws-neuron/neuronx-distributed`.

```bash
# Create a directory for experiments
mkdir -p ~/examples/tp_llama3_8b_lora_finetune
cd ~/examples/tp_llama3_8b_lora_finetune

# Download scripts
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/data_module.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/module_llama.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/tp_llama_hf_finetune_ptl.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lr.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/modeling_llama_nxd.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements.txt
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements_ptl.txt
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/training_utils.py
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/test/integration/modules/lora/test_llama_lora_finetune.sh

```

### 1.3. Install Requirements

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_ptl.txt
python3 -m pip install nltk blobfile tiktoken "huggingface_hub<1.0"

# Grant execution permission
chmod +x test_llama_lora_finetune.sh

# Prepare NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');"

```

---

## 2. Prepare Model & Dataset

### 2.1. Download Llama-3-8B Checkpoint (Hugging Face)

Download the base model to a local directory.

```bash
huggingface-cli login
# Download to ./models/llama3-8b
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir /home/ubuntu/models/llama3-8b

```

### 2.2. Convert Checkpoint to Neuron Format (NXD)

To optimize memory usage during conversion, we use a custom conversion script that loads the model directly using `AutoModelForCausalLM`.

1. **Create/Update `convert_checkpoints.py`:**
Overwrite the default script with the following optimized code to handle memory efficiently.
```python
import argparse
import torch
from transformers import AutoModelForCausalLM
from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase

class CheckpointConverterLlama(CheckpointConverterBase):
    def load_full_state(self, args):
        print(f"Loading model directly from {args.input_dir} using Transformers...")
        # Load model efficiently
        model = AutoModelForCausalLM.from_pretrained(
            args.input_dir, 
            torch_dtype="auto", 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        return model.state_dict()

if __name__ == "__main__":
    checkpoint_converter = CheckpointConverterLlama()
    parser = checkpoint_converter.get_arg_parser()
    args, _ = parser.parse_known_args()
    checkpoint_converter.run(args)

```


2. **Run Conversion:**
Convert the Hugging Face format to Neuron Distributed (Megatron-like) format.
```bash
python3 convert_checkpoints.py \
  --hw_backend trn2 \
  --tp_size 32 \
  --qkv_linear 1 \
  --kv_size_multiplier 4 \
  --convert_from_full_state \
  --config config.json \
  --input_dir /home/ubuntu/models/llama3-8b \
  --output_dir /home/ubuntu/models/llama3-8b-nxdt-tp32/pretrained_weight/

```


* `--tp_size 32`: Target Tensor Parallelism degree (matches Trn2 node).
* `--hw_backend trn2`: Target hardware.



---

## 3. Fine-tuning with LoRA

### 3.1. Configure Training Script

Edit `test_llama_lora_finetune.sh` to set paths and training parameters.

```bash
# Set Paths
PRETRAINED_PATH=/home/ubuntu/models/llama3-8b-nxdt-tp32/
BASE_MODEL=/home/ubuntu/models/llama3-8b
HF_TOKEN='your_token_here'

# Training Parameters (Full Fine-tuning)
# Use -1 for steps to run full epochs
TOTAL_STEPS=-1 
TOTAL_EPOCHS=3

```

> **Note:** Ensure `max_train_samples` is removed or commented out to train on the full dataset.

### 3.2. Start Training

```bash
./test_llama_lora_finetune.sh

```

**Completion Indicators:**

* Log message: `Training finished!`
* Log message: `synced saving of checkpoint lora completed`

### 3.3. Verify Output

The LoRA adapter will be saved in the output directory. Note that NXD saves checkpoints as sharded `.pt` files.

```text
lora_adapter/
├── adapter_config.json
└── lora/model/
    ├── dp_rank_00_tp_rank_00_pp_rank_00.pt
    ... (32 files for TP=32)

```

---

## 4. Inference with vLLM (Multi-LoRA)

### 4.1. Run vLLM Docker Container

Launch the vLLM container with Neuron device access.

```bash
docker run -d -it --privileged --shm-size=32g \
  -v /home/ubuntu/:/home/ubuntu/ \
  -v /dev:/dev \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK \
  -p 8000:8000 \
  --name vllm_llama8 \
  vllm-neuron:latest  # Replace with your image ID

```

### 4.2. Prepare LoRA Adapter for vLLM

vLLM on Neuron expects the sharded `.pt` files to be in the root of the adapter directory, not nested deep in `lora/model`.

1. **Move Checkpoint Files:**
```bash
cd /home/ubuntu/lora_adapters/llama3-8b-dolly-lora/
mv lora/model/*.pt .
rm -rf lora

```


*Now `adapter_config.json` and `*.pt` files should be in the same directory.*
2. **Fix `adapter_config.json` (Important):**
Neuron vLLM uses separate layers for Q, K, V. You must update `target_modules` in `adapter_config.json`.
**Before:**
```json
"target_modules": ["qkv_proj"],

```


**After:**
```json
"target_modules": ["q_proj", "k_proj", "v_proj"],

```



### 4.3. Run Inference Script (`neuron_multi_lora.py`)

Create a Python script to serve the model with the LoRA adapter.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Paths
MODEL_PATH = "/home/ubuntu/models/llama3-8b/"
LORA_PATH_1 = "/home/ubuntu/lora_adapters/llama3-8b-dolly-lora/" # Path containing .pt files & config

# Prompts (Use Instruction Format)
prompts = ["""### Instruction:
Write a romantic love letter about a potato.

### Response:
"""]

# Sampling Params
sampling_params = SamplingParams(top_k=1, max_tokens=1024)

# Initialize LLM
llm = LLM(
    model=MODEL_PATH,
    max_num_seqs=2,
    max_model_len=4096,
    tensor_parallel_size=32,
    device="neuron",
    override_neuron_config={
        "sequence_parallel_enabled": False,
        "lora_modules": {"lora_id_1": LORA_PATH_1}, # Static loading
    },
    enable_lora=True,
    max_loras=2,
)

# Generate
lora_req_1 = LoRARequest("lora_id_1", 1, LORA_PATH_1)
outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1])

# Print Output
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

```

### 4.4. Troubleshooting Check

* **Response truncated?** Check `max_tokens` in `SamplingParams`.
* **Result same as base model?** Ensure you are pointing to the correct `checkpoint-xxx` directory if multiple checkpoints exist, and verify that `target_modules` in config matches the Neuron model structure (`q_proj`, `k_proj`, `v_proj`).

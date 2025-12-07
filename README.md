# XC AWS Silicon Neuron 🚀

A comprehensive guide and resource hub for **AWS Accelerated Computing (XC)**, focusing exclusively on AWS Silicon (**Trainium & Inferentia**) and the **AWS Neuron SDK**.

## 🎯 Objectives
* **Neuron SDK Mastery:** Setup guides for `torch-neuronx` and `neuronx-distributed`.
* **Inference:** Serving LLMs using **vLLM (NxD)** and **Optimum Neuron**.
* **Training:** Large-scale distributed training on Trn1/Trn2 instances.
* **Performance:** Tips for compilation, caching, and NeuronCore utilization.

## 📂 Contents
* `/vllm-serving`: Guides for deploying Qwen, Llama, and other models using vLLM on NxD.
* `/training-distributed`: Examples for pre-training and fine-tuning with Megatron/NeMo.
* `/benchmarks`: Performance testing scripts and results.

## 🛠️ Getting Started
```bash
# Example: Setting up Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
pip install neuronx-distributed
```

## 🔗 Related Repositories
XC Common Infra
ParallelCluster on AWS

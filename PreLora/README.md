# PreLora: Fine-tuning with Low-Rank Matrix Decomposition and Prefix Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-yellow.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **PreLora**, a novel fine-tuning approach that combines prefix tuning with low-rank matrix decomposition for efficient adaptation of large language models to domain-specific tasks.

> 📖 **Related Paper**: [PreLora: A Fine-tuning Approach with Low-Rank Matrix Decomposition and Prefix Tuning for Pre-hospital Emergency Text Classification](https://arxiv.org/abs/xxxx.xxxxx)

## ✨ Features

- **🚀 Parameter-efficient Fine-tuning**: Combines prefix tuning and LoRA for optimal performance with minimal trainable parameters
- **🏥 Domain Adaptation**: Specifically designed for medical text classification tasks
- **🔄 Model Agnostic**: Compatible with various transformer architectures (BERT, GPT, LLaMA, ChatGLM, etc.)
- **⚡ Easy Integration**: Simple API for quick adaptation to existing pipelines
- **📊 Comprehensive Evaluation**: Includes performance comparison with other fine-tuning methods


## 🛠 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+

### Install from Source

```bash
git clone https://github.com/your-username/prelora.git
cd prelora
pip install -r requirements.txt





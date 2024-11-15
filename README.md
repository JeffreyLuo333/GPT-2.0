# GPT-2.0 Implementation

This repository contains my self-coded implementation of GPT-2, a transformer-based language model for generating human-like text. The project is an educational and experimental endeavor to understand the architecture and working of transformer models.

## Overview

GPT-2 (Generative Pre-trained Transformer 2) is a state-of-the-art language model that uses the transformer architecture and attention mechanisms to process and generate text. This implementation builds the model from scratch, exploring core components such as:

- Multi-head self-attention
- Positional encodings
- Tokenization
- Layer normalization
- Residual connections

The main implementation resides in `train_gpt2.py`, which includes both the model definition and training loop.

---

## Features

- Implementation of the full transformer architecture, including multi-head self-attention and feedforward layers.
- Tokenization and dataset preprocessing for text generation tasks.
- Training on small text datasets with a single GPU setup.

---

## Limitations

Due to hardware constraints while working on a macOS environment, this implementation does **not** include functionality for **multi-GPU parallelization**. The code is optimized for MPS systems but can be adapted for multi-GPU setups using libraries like `torch.nn.DataParallel` or `torch.distributed`.

---

## Requirements

Make sure you have the following installed:

- Python 3.8+
- PyTorch 2.0+
- NumPy

Install the dependencies manually or through pip:

```bash
pip install torch numpy

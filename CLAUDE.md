# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GETA (Generic, Efficient Training framework that Automates joint structured pruning and mixed precision quantization) is a PyTorch-based neural network compression framework. It automates structured pruning and quantization across CNNs, transformers, vision transformers, and small language models.

## Core Architecture

The framework follows a graph-based approach to neural network compression:

- **OTO (Only Train Once)**: Main entry point class that orchestrates the entire compression pipeline
- **Graph Analysis**: Constructs pruning search space via quantization-aware dependency graph (QADG)
- **GETA Optimizer**: White-box optimizer for joint sparsity and quantization constraints (QASSO)
- **Subnet Construction**: Extracts compressed subnetworks after training

Key modules in `only_train_once/`:
- `graph/`: Computational graph construction and node grouping
- `optimizer/`: GETA, HESSO, and other sparse optimizers
- `quantization/`: Quantization-aware layers and model conversion
- `transform/`: Graph and tensor transformations
- `subnet_construction/`: Pruned model extraction

## Development Commands

### Running Tests
```bash
# Run all sanity checks
cd sanity_check
python sanity_check.py

# Run individual test
python -m pytest sanity_check/test_resnet18.py -v

# Run specific test class
python -m pytest sanity_check/test_resnet18.py::TestResNet18::test_sanity -v
```

### Installation
```bash
git clone https://github.com/microsoft/geta.git
cd geta
pip install -e .
```

Requires PyTorch >= 2.0

## Usage Pattern

```python
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model

# 1. Prepare quantized model
model = model_to_quantize_model(model, quant_mode="weight_and_activation")

# 2. Initialize OTO with dummy input
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# 3. Create optimizer with compression targets
optimizer = oto.geta(
    variant="adam",
    lr=1e-3,
    target_group_sparsity=0.5,
    bit_reduction=2,
    min_bit_wt=4,
    max_bit_wt=16,
    # ... other parameters
)

# 4. Train normally
for epoch in range(max_epoch):
    for X, y in trainloader:
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. Extract compressed subnet
oto.construct_subnet(out_dir='./')
```

## Key Concepts

- **Node Groups**: Parameters grouped for joint pruning decisions
- **Group Sparsity**: Fraction of node groups zeroed out
- **Bit Width Range**: Target quantization precision bounds
- **Projection/Pruning Steps**: Training steps for different compression phases

## File Structure

- `sanity_check/`: Test suite with model-specific test cases
- `tutorials/`: Jupyter notebooks with usage examples
- `test_scripts/`: Extended test scripts for various architectures
- `tools/`: Visualization utilities

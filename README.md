# cMamba Model Documentation

## Overview

The cMamba model is an unofficial implementation based on the state-of-the-art methodologies proposed in recent research. This implementation integrates key advancements from two significant papers:

1. [C-Mamba: Efficient Context Modeling with Mamba](https://arxiv.org/abs/2406.05316v1) (2023)
2. [Linear Time-Sequence Modeling with Mamba](https://arxiv.org/abs/2312.00752v2) (2023)

## Features

- **Channel Mixup**: A data augmentation technique to enhance the robustness of the model.
- **Channel Attention**: Mechanism to dynamically weigh different channels based on their importance.
- **Patch-based Input Processing**: Efficiently handles long sequences by splitting them into patches.
- **Selective State-Space Model (SSM)**: Implements advanced state-space modeling for capturing long-range dependencies.

## Installation

To use this model, you need to have the following libraries installed:
- `torch`
- `einops`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `wandb`

You can install them using pip:

```bash
pip install torch einops numpy pandas scikit-learn matplotlib wandb

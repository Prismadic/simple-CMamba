# c(hannel)Mamba Model Documentation

## Overview

The cMamba code outlined is for an unofficial implementation based on the SoTA methodologies proposed in recent state-space model research. This implementation integrates key advancements from two significant papers:

1. [C-Mamba: Efficient Context Modeling with Mamba](https://arxiv.org/abs/2406.05316v1) (2024)
2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2) (2023)

with help from [johnma2006](https://github.com/johnma2006/mamba-minimal/) for the original Mamba implementation in pure pytorch

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
```

## Data Preparation

Before training the cMamba model, you'll need to prepare your data. We provide a data preparation script that preprocesses your time series data, normalizes the features, and splits it into training and test sets.

### Data Preparation Script

The data preparation script performs the following steps:
1. Loads the input JSON file containing the time series data.
2. Converts timestamps to datetime format, removes duplicates, and sorts the data.
3. Selects numerical columns as features.
4. Normalizes the features using `StandardScaler`.
5. Creates input-output sequences based on specified input and forecast lengths.
6. Splits the data into training and test sets.
7. Saves the processed data to an NPZ file.

### Usage

To use the data preparation script, save it as `prepare_data.py` and run it from the command line with the appropriate arguments. Here are the arguments you need to provide:

- `--input_file`: Path to the input JSON file containing the time series data.
- `--output_file`: Path to the output file (NPZ format) where the processed data will be saved.
- `--input_length`: Length of the input sequences.
- `--forecast_length`: Length of the forecast sequences.
- `--test_size`: Proportion of the dataset to include in the test split (default is 0.2).

### Example Command

```python
python prepare_data.py --input_file exported_data_transformed.json --output_file prepared_data.npz --input_length 96 --forecast_length 96 --test_size 0.2
```

This command will load the data from exported_data_transformed.json, process it, and save the training and test sets to prepared_data.npz.

## Training

After preparing your data, you can train the cMamba model using the training script. The training script initializes the model, sets up the loss function and optimizer, and handles the training loop with logging and model saving.

### Training Script

The training script performs the following steps:

1. Initializes the WandB run for tracking.
2. Initializes the model, loss function, and optimizer.
3. Loads the training and validation data.
4. Runs the training loop with gradient clipping and loss calculation.
5. Logs various metrics, including batch loss, epoch loss, gradient norms, and weight statistics.
6. Saves the model checkpoints and final model.

#### Usage

To use the training script, save it as train_c_mamba.py and run it from the command line with the appropriate arguments. Here are the arguments you need to provide:

- `--train_dataset`: Path to the training dataset (NPZ format).
- `--test_dataset`: Path to the test dataset (NPZ format).
- `--project_name`: WandB project name.
- `--learning_rate`: Learning rate for the optimizer (default is 0.- 001).
- `--num_epochs`: Number of epochs for training (default is 10).
- `--batch_size`: Batch size for training (default is 32).
- `--seq_len`: Length of the input sequences.
- `--forecast_len`: Length of the forecast sequences.
- `--input_dim`: Input dimension for the model.
- `--hidden_dim`: Hidden dimension for the model.
- `--num_layers`: Number of layers in the model.


#### Example Command

```bash
python train_c_mamba.py --train_dataset prepared_data.npz --test_dataset prepared_data.npz --seq_len 96 --forecast_len 96 --input_dim 17 --hidden_dim 128 --num_layers 4 --project_name my_c_mamba_project --learning_rate 0.001 --num_epochs 10 --batch_size 32
```

This command will train the cMamba model using the provided parameters and log the training process to WandB.

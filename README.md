# GNN-Based Graph Classification with Noisy Label Robustness

## Overview
This project implements a **Graph Neural Network (GNN)** model to classify graph datasets. 
It supports multiple GNN architectures, handles noisy labels in naive settings, and provides
predictions with checkpoint saving and visualization of training progress. The solution is 
for graph datasets provided in competition settings.

---

## Features
- **Multiple GNN Architectures**: Supports GIN, GCN, and their virtual node variants, here i have implemented for GIN.
- **Checkpointing**: Saves the model at defined intervals to track progress.
- **Visualization**: Logs and visualizes training loss and accuracy across epochs.
- **Submission Output**: Generates a CSV file with predictions for test datasets.
- **Best models**: models that gave highest training accuracies

---

## How to Run

### Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
```

For training
```python
python main.py --train_path <train_dataset_path> --test_path <test_dataset_path>
```
For Infrence 

```python
python main.py --test_path <test_dataset_path>
```

**Outputs:**
- Training Logs: Saved in the logs/ directory.
- Checkpoints: Saved in checkpoints/.
- Predictions: Generated as submission/testset_<test_folder_name>.csv.

## Key Arguments

| Argument       | Description                                                | Default       |
|----------------|------------------------------------------------------------|---------------|
| `--train_path` | Path to the training dataset (optional for inference).      | `None`        |
| `--test_path`  | Path to the test dataset (required).                        | `None`        |
| `--gnn`        | Type of GNN: `gin`, `gin-virtual`, `gcn`, or `gcn-virtual`. | `gin`         |
| `--epochs`     | Number of training epochs.                                  | `5`           |
| `--batch_size` | Batch size for training/testing.                            | `32`          |
| `--emb_dim`    | Embedding dimensions for hidden layers.                     | `300`         |
| `--num_layer`  | Number of GNN layers.                                       | `5`           |
| `--device`     | GPU device to use (if available).                           | `0` (first GPU) |



## Project Structure

```bash
.
├── src/
│   ├── __init__.py 
│   ├── loadData.py     # Graph dataset loader
│   ├── models.py       # GNN model definitions
│   ├── utils.py        # Utility functions (e.g., seeding)
│   ├── conv.py         # different model layers and convolutions
│   ├── Zipthefolder.py # This one specifically used to creted the tar file for submission
├── main.py             # Main execution script
├── requirements.txt    # Required Python dependencies
└── README.md           # Project documentation
```
## Output Example

Predictions are saved as a CSV file in the `submission/` directory:

```csv
id,pred
0,1
1,2
2,0
...
```

# PhyGHT: Physics-Guided Hypergraph Transformer for Pileup Mitigation

This repository contains the official PyTorch implementation of PhyGHT (Physics-Guided Hypergraph Transformer). The paper can be found here: [arXiv](link here).

## Environment & Dependencies

This code was developed and tested using the following environment configurations:

* Python: 3.10.18
* PyTorch: 2.6.0+cu124
* CUDA: 12.4
* PyTorch Geometric: 2.6.1
* Awkward Array: 2.8.5
* NumPy: 1.26.4
* Scikit-Learn: 1.7.1
* Matplotlib: 3.10.3
* Seaborn: 0.13.2
* tqdm: 4.67.1

## Dataset Preparation

The dataset contains simulated particle collision events mimicking the HL-LHC environment, available for two pileup conditions: standard LHC pileup (mu=60) and extreme HL-LHC pileup (mu=200). Details on the dataset can be found in the [paper](link here) and the associated [Zenodo repository](link here).

### 1. Download Raw Data

Download the raw `.pkl` files from our Zenodo repository:
* [Zenodo](link here)

Place the downloaded files into the `data/raw_data/` directory so the project structure looks like this:

```text
PhyGHT/
├── data/
│   ├── raw_data/
│   │   ├── mu60_10k_events_data.pkl
│   │   └── mu200_10k_events_data.pkl
```

### 2. Preprocess and Split Data

Before training, we need to shuffle and split the raw events into training, validation, and test sets.

Navigate to the `data/` directory and run the preprocessing script. We can process either dataset by modifying the `MU_VERSION` variable inside `preprocess_events.py` to `"mu60"` or `"mu200"`.

```python
cd data
python preprocess_events.py
cd ..
```

This will generate the processed splits (train, validation, test) inside the `data/processed_data/` directory.

## Configuration

All global variables, model hyperparameters, and paths are managed centrally in `config.py`. There are no command-line arguments required for training; simply adjust the values in `config.py` before running the experiment.

Key parameters that can be adjusted include:
* `DATASET_NAME`: Switch between `"mu60_10k_events"` and `"mu200_10k_events"`.
* `BATCH_SIZE`: Number of events per batch (default: `16`).
* `K_NEIGHBORS`: Number of edges for the spatial k-NN graph (default: `8`).
* `AUX_WEIGHT`: Controls the contribution of the auxiliary classification loss (default: `0.1`).
* `LEARNING_RATE`: Set the learning rate for the optimizer (default: `3e-4`).
* `OPTIMIZER`: The optimizer to use among sgd, adam, and adamw (default: `adamw`).
* `NUM_EPOCHS`: Total number of training epochs (default: `200`).
* `NUM_HEADS`: Number of attention heads (default: `4`).
* `NUM_LAYERS`: Number of layers (default: `3`).
* `HIDDEN_DIM`: Hidden dimensions for the PhyGHT model (default: `128`).
* `DROPOUT`: Dropout rate (default: `0.1`).
* `EVAL_EVERY_N_EPOCHS`: Evaluate the model every N epochs (default: `1`).
* `DEBUG`: Set to `True` for quick debugging (default: `False`).

## Training PhyGHT

Once the data is preprocessed and configuration is set in `config.py`, we can train the PhyGHT model by running:

```python
python train.py
```

## Output Directory Structure

During training, the pipeline automatically creates the following directories to save the results:

* `checkpoints/PhyGHT/`: Saves the best model weights based on Validation Loss
* `metrics/PhyGHT/`: Saves train, val, and test evaluation metrics.
* `plots/PhyGHT/`: Saves Loss Curves, Actual vs. Predicted heatmaps, and Residual distributions.

## Citation

If you find this code or dataset useful in your research, please consider citing our paper:

```bibtex

```
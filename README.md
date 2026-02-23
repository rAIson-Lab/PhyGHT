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

The datasets contain simulated particle collision events mimicking the HL-LHC environment, available for two pileup conditions: standard LHC pileup (mu=60) and extreme HL-LHC pileup (mu=200).

### 1. Download Raw Data

Download the raw .pkl files from our Zenodo repository:
* [Zenodo](link here)

Place the downloaded files into the data/raw_data/ directory so your project structure looks like this:

```
PhyGHT/
├── data/
│   ├── raw_data/
│   │   ├── mu60_10k_events_data.pkl
│   │   └── mu200_10k_events_data.pkl
```

### 2. Preprocess and Split Data

Before training, you need to shuffle and split the raw events into training, validation, and test sets.

Navigate to the data directory and run the preprocessing script. You can process either dataset by modifying the MU_VERSION variable inside data/preprocess_events.py to "mu60" or "mu200".

```python
python preprocess_events.py
```

This will generate the processed splits (train, validation, test) inside the data/processed_data/ directory.

## Configuration

All global variables, model hyperparameters, and paths are managed centrally in config.py. There are no command-line arguments required for training; simply adjust the values in config.py before running your experiment.

Key parameters you might want to adjust include:
* DATASET_NAME: Switch between "mu60_10k_events" and "mu200_10k_events".
* BATCH_SIZE: Number of events per batch (default: 16).
* K_NEIGHBORS: Number of edges for the spatial k-NN graph (default: 8).
* AUX_WEIGHT: Controls the contribution of the auxiliary classification loss (default: 1.0).
* LEARNING_RATE: Set the learning rate for the optimizer (default: 3e-4).
* NUM_EPOCHS: Total number of training epochs (default: 200).

## Training PhyGHT

Once your data is preprocessed and your configuration is set in config.py, you can train the PhyGHT model by running:

```python
python train.py
```

## Output Directory Structure

During training, the code automatically creates the following directories to save your results:

* checkpoints/phyght/: Saves the best model weights based on Validation Loss
* metrics/phyght/: Saves train, val, and test evaluation metrics.
* plots/phyght/: Saves Loss Curves, Actual vs. Predicted heatmaps, and Residual distributions.

## Citation

If you find this code or dataset useful in your research, please consider citing our paper:

```bibtex

```
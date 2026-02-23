import os
import torch

# --- Global Flags ---
DEBUG = False
# DEBUG = True
MULTI_GPU = False
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")

if torch.cuda.is_available():
    print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")
    if MULTI_GPU:
        print(f"--- Multi-GPU Enabled: Using {torch.cuda.device_count()} GPUs ---")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

for d in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Dataset & Architecture ---
DATASET_NAME = "mu60_10k_events"
# DATASET_NAME = "mu60_200k_events"
MODEL_ARCHITECTURE = "PhyGHT"

# --- Hyperparameters ---
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
K_NEIGHBORS = 8
DROPOUT = 0.1

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 200
OPTIMIZER = 'adamw'
AUX_WEIGHT = 1.0

EVAL_EVERY_N_EPOCHS = 1

if DEBUG:
    print("!!! RUNNING IN DEBUG MODE !!!")
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    EVAL_EVERY_N_EPOCHS = 1

# --- Configuration Dictionary ---
CONFIG = {
    'BASE_DIR': BASE_DIR,
    'PROCESSED_DATA_DIR': PROCESSED_DATA_DIR,
    'CHECKPOINT_DIR': CHECKPOINT_DIR,
    'METRICS_DIR': METRICS_DIR,
    'PLOTS_DIR': PLOTS_DIR,
    
    'DEBUG': DEBUG,
    'MULTI_GPU': MULTI_GPU,
    'DEVICE': DEVICE,
    'SEED': SEED,
    
    'DATASET_NAME': DATASET_NAME,
    'ARCHITECTURE': MODEL_ARCHITECTURE,
    
    'HIDDEN_DIM': HIDDEN_DIM,
    'NUM_HEADS': NUM_HEADS,
    'NUM_LAYERS': NUM_LAYERS,
    'K_NEIGHBORS': K_NEIGHBORS,
    'DROPOUT': DROPOUT,
    
    'BATCH_SIZE': BATCH_SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'NUM_EPOCHS': NUM_EPOCHS,
    'OPTIMIZER': OPTIMIZER,
    'EVAL_FREQ': EVAL_EVERY_N_EPOCHS,
    'AUX_WEIGHT': AUX_WEIGHT,
}

# --- Automatic Run Name Generation ---
def get_run_name(cfg):
    parts = []
    
    if cfg['DEBUG']:
        parts.append("DEBUG")
        
    parts.extend([
        cfg['ARCHITECTURE'],
        cfg['DATASET_NAME'],
        f"BS{cfg['BATCH_SIZE']}",
        f"LR{cfg['LEARNING_RATE']}",
        f"EP{cfg['NUM_EPOCHS']}",
        f"OPT{cfg['OPTIMIZER']}",
        f"DIM{cfg['HIDDEN_DIM']}",
        f"HDS{cfg['NUM_HEADS']}",
        f"LAY{cfg['NUM_LAYERS']}",
        f"KN{cfg['K_NEIGHBORS']}",
        f"DR{cfg['DROPOUT']}",
        f"AUX{cfg['AUX_WEIGHT']}",
        f"SEED{cfg['SEED']}"
    ])
        
    return "_".join(parts)

CONFIG['RUN_NAME'] = get_run_name(CONFIG)

if __name__ == "__main__":
    import pprint
    print("\n--- FINAL CONFIGURATION ---")
    pprint.pprint(CONFIG)
    print(f"\nGenerated Run Name: {CONFIG['RUN_NAME']}")
    print("-" * 30)
import os
import pickle
import numpy as np
from typing import Tuple, Optional

def preprocess_and_split_events(
    pickle_path: str,
    output_dir: str,
    dataset_name: str,
    seed: int = 42,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    subset_size: Optional[int] = None
):
    """
    Loads raw event data, shuffles all event indices,
    and splits that set into train/validation/test files.
    """
    print(f"Loading raw event data from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    jet_events_ak = data_dict["jets"]
    jet_track_events_ak = data_dict["jet_trk_IDs"]
    all_track_events_ak = data_dict["trks"]

    num_events_total = len(jet_events_ak)
    print(f"Found {num_events_total} total events in file.")

    np.random.seed(seed)
    all_shuffled_indices = np.random.permutation(num_events_total)

    if subset_size is not None:
        subset_size = min(subset_size, num_events_total)
        print(f"Selecting a random subset of {subset_size} events.")
        event_indices_to_split = all_shuffled_indices[:subset_size]
    else:
        print(f"Processing all {num_events_total} events.")
        event_indices_to_split = all_shuffled_indices
    
    num_events_to_split = len(event_indices_to_split)
    train_end = int(num_events_to_split * split_ratios[0])
    val_end = train_end + int(num_events_to_split * split_ratios[1])

    train_indices = event_indices_to_split[:train_end]
    val_indices = event_indices_to_split[train_end:val_end]
    test_indices = event_indices_to_split[val_end:]
    
    os.makedirs(output_dir, exist_ok=True)

    def save_split(name, indices):
        split_jets = jet_events_ak[indices]
        split_jet_tracks = jet_track_events_ak[indices]
        split_all_tracks = all_track_events_ak[indices]
        
        output_path = os.path.join(output_dir, f"{dataset_name}_{name}_seed_{seed}.pkl")
        print(f"Saving {len(split_jets)} events for '{name}' set to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump((split_jets, split_jet_tracks, split_all_tracks), f)

    save_split('train', train_indices)
    save_split('validation', val_indices)
    save_split('test', test_indices)

    print("\nEvent-based preprocessing and splitting complete.")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
    
    RANDOM_SEED = 42
    SPLIT_RATIOS = (0.8, 0.1, 0.1)
    
    # Choose which dataset version to process: "mu60" or "mu200"
    # MU_VERSION = "mu60" 
    MU_VERSION = "mu200" 
    
    dataset_name = f"{MU_VERSION}_10k_events"
    subset_size_to_process = None
    
    RAW_PICKLE_FILE = os.path.join(RAW_DATA_DIR, f"{dataset_name}_data.pkl")
    
    if not os.path.exists(RAW_PICKLE_FILE):
        print(f"Error: Raw data file not found at {RAW_PICKLE_FILE}")
        exit(1)
        
    preprocess_and_split_events(
        pickle_path=RAW_PICKLE_FILE,
        output_dir=PROCESSED_DATA_DIR,
        dataset_name=dataset_name,
        seed=RANDOM_SEED,
        split_ratios=SPLIT_RATIOS,
        subset_size=subset_size_to_process
    )
import os
import pickle
import torch
import numpy as np
import awkward as ak
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from config import CONFIG

class PhyGHTDataset(Dataset):
    """
    PyTorch Geometric Dataset for PhyGHT.
    
    Constructs a heterogeneous graph per event:
    - Nodes: 'track' (6 features, 1 label) and 'jet' (4 features, 2 targets)
    - Edges: ('jet', 'contains', 'track') representing constituency.
    """
    def __init__(self, split='train'):
        self.split = split
        self.data_dir = CONFIG['PROCESSED_DATA_DIR']
        self.dataset_name = CONFIG['DATASET_NAME']
        self.seed = CONFIG['SEED']
        
        file_name = f"{self.dataset_name}_{self.split}_seed_{self.seed}.pkl"
        file_path = os.path.join(self.data_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data file not found at: {file_path}")
            
        print(f"Loading {split} dataset from {file_path}...")
        with open(file_path, "rb") as f:
            self.jets_ak, self.jet_trk_ids_ak, self.tracks_ak = pickle.load(f)
            
        self.num_events = len(self.jets_ak)
        print(f"Successfully loaded {self.num_events} events.")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        ev_jets = self.jets_ak[idx]
        ev_jet_trk_ids = self.jet_trk_ids_ak[idx]
        ev_tracks = self.tracks_ak[idx]

        data = HeteroData()

        # Process Tracks
        trk_np = ak.to_numpy(ev_tracks)
        id_to_idx_map = {}
        
        if len(trk_np) == 0:
            data['track'].x = torch.zeros((0, 6), dtype=torch.float32)
            data['track'].label = torch.zeros((0,), dtype=torch.float32)
        else:
            data['track'].x = torch.from_numpy(trk_np[:, :6].astype(np.float32))
            
            raw_labels = trk_np[:, -1]
            data['track'].label = torch.from_numpy((raw_labels == -1).astype(np.float32))
            
            track_ids_raw = trk_np[:, 6].astype(int)
            id_to_idx_map = {tid: i for i, tid in enumerate(track_ids_raw)}

        # Process Jets
        jet_np = ak.to_numpy(ev_jets)
        
        if len(jet_np) == 0:
            data['jet'].x = torch.zeros((0, 4), dtype=torch.float32)
            data['jet'].y = torch.zeros((0, 2), dtype=torch.float32)
        else:
            data['jet'].x = torch.from_numpy(jet_np[:, :4].astype(np.float32))
            data['jet'].y = torch.from_numpy(jet_np[:, -2:].astype(np.float32))

        # Build Bipartite Edges
        sources, targets = [], []
        
        if len(jet_np) > 0 and len(trk_np) > 0:
            jet_constituents_list = ev_jet_trk_ids.tolist()
            
            for j_idx, constituents in enumerate(jet_constituents_list):
                for trk_id in constituents:
                    if trk_id in id_to_idx_map:
                        t_idx = id_to_idx_map[trk_id]
                        sources.append(j_idx)
                        targets.append(t_idx)

        if len(sources) > 0:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        data['jet', 'contains', 'track'].edge_index = edge_index

        return data

if __name__ == "__main__":
    
    print("--- PhyGHT Dataset Test ---")
    
    try:
        dataset = PhyGHTDataset(split='train')
    except Exception as e:
        print(f"Failed to initialize: {e}")
        exit()

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset Size: {len(dataset)}")
    
    for batch in loader:
        print("\nBatch Loaded (HeteroData Batch):")
        print(batch)
        print(f" Track Features: {batch['track'].x.shape}")
        print(f" Jet Features:   {batch['jet'].x.shape}")
        print(f" Track Labels:   {batch['track'].label.shape}")
        print(f" Edge Index:     {batch['jet', 'contains', 'track'].edge_index.shape}")
        
        if hasattr(batch['track'], 'batch'):
            print(f" Track Batch Vec: {batch['track'].batch.shape}")
            print(f" Max Event Idx:   {batch['track'].batch.max().item()} (Should be 3 for BS=4)")
        break
        
    print("\nTest passed successfully.")
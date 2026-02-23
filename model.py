import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, knn_graph
from torch_geometric.utils import to_dense_batch

from config import CONFIG

class GaussianGAT(nn.Module):
    """
    Phase I: Local Geometric Context
    A Graph Attention Layer (GATv2) that encodes the local jet structure using
    spatial distance (dR^2) to bias attention weights based on spatial proximity.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        # add_self_loops=False is critical because self-loops are handled in knn_graph
        self.gat = GATv2Conv(
            in_channels, out_channels, heads=heads, 
            concat=False, add_self_loops=False, edge_dim=1
        ) 
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, dist):
        identity = x
        out = self.gat(x, edge_index, edge_attr=dist)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + identity

class GlobalTransformerBlock(nn.Module):
    """
    Phase II: Global Context
    A Transformer Encoder that captures long-range correlations across the event,
    such as momentum balance and event-wide pileup density.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch)
        padding_mask = ~mask
        x_global_dense = self.transformer(x_dense, src_key_padding_mask=padding_mask)
        x_global_flat = x_global_dense[mask]
        return x_global_flat

class JetAttention(nn.Module):
    """
    Phase IV: Hypergraph Aggregation
    A Bipartite GAT where the Jet Node dynamically queries its constituent Tracks.
    """
    def __init__(self, track_dim, jet_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.track_proj = nn.Linear(track_dim, out_dim)
        self.jet_proj = nn.Linear(jet_dim, out_dim)

        self.gat = GATv2Conv(
            (out_dim, out_dim), out_dim, heads=heads, 
            concat=False, add_self_loops=False
        )

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_track, x_jet, edge_index):
        h_track = F.gelu(self.track_proj(x_track))
        h_jet = F.gelu(self.jet_proj(x_jet))

        h_jet_updated = self.gat((h_track, h_jet), edge_index)
        h_jet_updated = self.norm(h_jet_updated)
        h_jet_updated = self.dropout(h_jet_updated)

        return h_jet_updated

class PhyGHT(nn.Module):
    """
    PhyGHT: Physics-Guided Hypergraph Transformer
    A hierarchical architecture combining local geometric graphs with global transformers
    for robust pileup mitigation at the HL-LHC.
    """
    def __init__(self, hidden_dim=None, num_heads=None, num_layers=None, k_neighbors=None):
        super().__init__()

        hidden_dim = hidden_dim or CONFIG['HIDDEN_DIM']
        num_heads = num_heads or CONFIG['NUM_HEADS']
        transformer_layers = num_layers or CONFIG['NUM_LAYERS']
        self.k_neighbors = k_neighbors or CONFIG['K_NEIGHBORS']
        dropout = CONFIG['DROPOUT']

        input_track_dim = 6 # [pT, eta, phi, m, d0, z0]
        input_jet_dim = 4   # [pT, eta, phi, m]

        # Encoders
        self.track_lin = nn.Sequential(
            nn.Linear(input_track_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.jet_lin = nn.Sequential(
            nn.Linear(input_jet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Phase I: Local Geometry
        self.local_gat = GaussianGAT(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)

        # Phase II: Global Context
        self.global_transformer = GlobalTransformerBlock(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Phase III: Pileup Suppression Gate (PSG)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Phase IV: Hypergraph Aggregation
        self.jet_attention = JetAttention(hidden_dim, hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)

        # Fusion & Prediction Heads
        self.jet_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_efrac = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_mfrac = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, data):
        x = data['track'].x
        batch = data['track'].batch
        
        # 1. Spatial Geometry Encoding
        eta, phi = x[:, 1], x[:, 2]
        pos = torch.stack([eta, phi], dim=1)
        
        edge_index_knn = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=True)
        
        row, col = edge_index_knn
        d_eta = eta[row] - eta[col]
        d_phi = phi[row] - phi[col]
        dist_sq = (d_eta**2 + d_phi**2).view(-1, 1)

        # 2. Hybrid Processing
        h = self.track_lin(x)
        h_local = self.local_gat(h, edge_index_knn, dist_sq)
        h_global = self.global_transformer(h_local, batch)
        h_combined = h_local + h_global

        # 3. Gating & Aggregation
        signal_prob = self.gate_net(h_combined)
        h_weighted = h_combined * signal_prob
        
        bipartite_edge_index = data['jet', 'contains', 'track'].edge_index
        track_to_jet_index = torch.stack([bipartite_edge_index[1], bipartite_edge_index[0]], dim=0)

        h_jet_raw = self.jet_lin(data['jet'].x)
        h_jet_aggr = self.jet_attention(h_weighted, h_jet_raw, track_to_jet_index)
        
        # 4. Prediction
        h_jet_final = self.jet_fusion(torch.cat([h_jet_raw, h_jet_aggr], dim=-1))

        pred_e = self.head_efrac(h_jet_final).squeeze(-1)
        pred_m = self.head_mfrac(h_jet_final).squeeze(-1)

        return pred_e, pred_m, signal_prob


if __name__ == "__main__":
    from torch_geometric.data import HeteroData
    
    print("--- Testing PhyGHT Architecture ---")
    
    data = HeteroData()
    data['jet'].x = torch.randn(5, 4)       
    data['track'].x = torch.randn(100, 6)   
    data['track'].batch = torch.zeros(100).long() 

    jet_idxs = torch.arange(5).repeat_interleave(10)
    trk_idxs = torch.randint(0, 100, (50,))
    data['jet', 'contains', 'track'].edge_index = torch.stack([jet_idxs, trk_idxs], dim=0)

    model = PhyGHT(hidden_dim=32, num_heads=2, num_layers=2, k_neighbors=5)

    try:
        e, m, p = model(data)
        print("\nForward Pass Successful!")
        print(f"Energy Pred Shape: {e.shape}")
        print(f"Mass Pred Shape:   {m.shape}")
        print(f"Signal Prob Shape: {p.shape}")
    except Exception as e:
        print(f"\nError: {e}")
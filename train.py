import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from config import CONFIG
from dataset import PhyGHTDataset
from model import PhyGHT
from eval import calculate_regression_metrics, save_metrics_to_file
from plot import plot_loss_curves, plot_actual_vs_predicted, plot_residuals

# --- Configuration & Seeding ---
DEVICE = CONFIG['DEVICE']
RUN_NAME = CONFIG['RUN_NAME']
ARCH_DIR = CONFIG['ARCHITECTURE'].lower()
SEED = CONFIG['SEED']

torch.manual_seed(SEED)
np.random.seed(SEED)

def get_dataloaders():
    """Initializes PyG DataLoaders for all splits."""
    print("\n--- Initializing DataLoaders ---")
    loaders = {}
    for split in ['train', 'validation', 'test']:
        ds = PhyGHTDataset(split=split)
        loaders[split] = DataLoader(
            ds, 
            batch_size=CONFIG['BATCH_SIZE'], 
            shuffle=(split == 'train'),
            num_workers=0 
        )
    return loaders

def get_optimizer(model):
    """Selects the optimizer based on configuration."""
    opt_name = CONFIG['OPTIMIZER'].lower()
    lr = CONFIG['LEARNING_RATE']
    
    if opt_name == 'adamw': return optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == 'adam': return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'sgd': return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else: raise ValueError(f"Unsupported Optimizer: {opt_name}")

class HybridLoss(nn.Module):
    """
    Hybrid Physics Loss for PhyGHT.
    1. Regression (MSE): Energy Fraction & Mass Fraction (Per Jet)
    2. Classification (BCE): Signal vs Pileup (Per Track)
    """
    def __init__(self, aux_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss() 
        self.bce = nn.BCELoss()
        self.aux_weight = aux_weight

    def forward(self, pred_e, pred_m, pred_signal, batch):
        target_e = batch['jet'].y[:, 0].to(DEVICE)
        target_m = batch['jet'].y[:, 1].to(DEVICE)
        
        loss_e = self.mse(pred_e, target_e)
        loss_m = self.mse(pred_m, target_m)
        reg_loss = loss_e + loss_m

        target_signal = batch['track'].label.to(DEVICE)
        aux_loss = self.bce(pred_signal.view(-1), target_signal)

        total_loss = reg_loss + (self.aux_weight * aux_loss)
        return total_loss, reg_loss, aux_loss

def run_inference(model, loader, desc="Evaluating"):
    """Runs inference to collect regression predictions."""
    model.eval()
    all_true_e, all_pred_e = [], []
    all_true_m, all_pred_m = [], []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=desc, leave=False):
            batch = batch.to(DEVICE)
            pred_e, pred_m, _ = model(batch)
            
            all_true_e.append(batch['jet'].y[:, 0].cpu().numpy())
            all_pred_e.append(pred_e.cpu().numpy())
            all_true_m.append(batch['jet'].y[:, 1].cpu().numpy())
            all_pred_m.append(pred_m.cpu().numpy())
            
    return (np.concatenate(all_true_e), np.concatenate(all_pred_e),
            np.concatenate(all_true_m), np.concatenate(all_pred_m))

def train_one_epoch(model, loader, optimizer, criterion):
    """Executes a single training epoch."""
    model.train()
    total_loss_avg, reg_loss_avg, aux_loss_avg = 0.0, 0.0, 0.0
    
    pbar = tqdm.tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        pred_e, pred_m, pred_signal = model(batch)
        loss, reg, aux = criterion(pred_e, pred_m, pred_signal, batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss_avg += loss.item()
        reg_loss_avg += reg.item()
        aux_loss_avg += aux.item()
        
        pbar.set_postfix({
            'T': f"{loss.item():.4f}", 
            'R': f"{reg.item():.4f}", 
            'A': f"{aux.item():.4f}"
        })
        
    return total_loss_avg / len(loader)

def validate(model, loader, criterion):
    """Runs validation and computes loss and R2 metrics."""
    model.eval()
    running_loss = 0.0
    all_true_e, all_pred_e = [], []
    all_true_m, all_pred_m = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred_e, pred_m, pred_signal = model(batch)
            
            loss, _, _ = criterion(pred_e, pred_m, pred_signal, batch)
            running_loss += loss.item()
            
            all_true_e.append(batch['jet'].y[:, 0].cpu().numpy())
            all_pred_e.append(pred_e.cpu().numpy())
            all_true_m.append(batch['jet'].y[:, 1].cpu().numpy())
            all_pred_m.append(pred_m.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    
    metrics_e = calculate_regression_metrics(np.concatenate(all_true_e), np.concatenate(all_pred_e))
    metrics_m = calculate_regression_metrics(np.concatenate(all_true_m), np.concatenate(all_pred_m))
    
    return avg_loss, metrics_e['R2'], metrics_m['R2']

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- STARTING PhyGHT TRAINING: {RUN_NAME} ---")
    
    loaders = get_dataloaders()
    model = PhyGHT(k_neighbors=CONFIG['K_NEIGHBORS']).to(DEVICE)
    
    if CONFIG['MULTI_GPU']: 
        model = nn.DataParallel(model)
        
    optimizer = get_optimizer(model)
    criterion = HybridLoss(aux_weight=CONFIG['AUX_WEIGHT'])
    
    ckpt_dir = os.path.join(CONFIG['CHECKPOINT_DIR'], ARCH_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    last_saved_loss_path = ""
    
    train_losses, val_losses = [], []
    EPOCHS = CONFIG['NUM_EPOCHS']
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        t_loss = train_one_epoch(model, loaders['train'], optimizer, criterion)
        train_losses.append(t_loss)
        
        if (epoch + 1) % CONFIG['EVAL_FREQ'] == 0:
            v_loss, v_r2_e, v_r2_m = validate(model, loaders['validation'], criterion)
            val_losses.append(v_loss)
            
            print(f"Loss: {t_loss:.4f} (Train) | {v_loss:.4f} (Val) || R2_E: {v_r2_e:.4f} | R2_M: {v_r2_m:.4f}")
            
            def get_ckpt_name(tag, ep, re, rm):
                return os.path.join(ckpt_dir, f"{RUN_NAME}_{tag}_ep{ep}_R2E{re:.4f}_R2M{rm:.4f}.pt")

            # Save Strategy: Best Overall Hybrid Loss
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                new_path = get_ckpt_name("BestLoss", epoch+1, v_r2_e, v_r2_m)
                if last_saved_loss_path and os.path.exists(last_saved_loss_path): os.remove(last_saved_loss_path)
                print(f"--> Best Loss! Saving to {os.path.basename(new_path)}")
                torch.save(model.state_dict(), new_path)
                last_saved_loss_path = new_path
            
        else:
            val_losses.append(val_losses[-1] if len(val_losses) > 0 else t_loss)

    plot_loss_curves(train_losses, val_losses, RUN_NAME, ARCH_DIR)
    
    print("\n--- Running Final Evaluation on Best Models ---")
    full_report = {
        'Best Loss Model': {},
    }
    
    def eval_and_record(ckpt_path, label, plot_energy=True, plot_mass=True):
        """Loads best checkpoints, records metrics, and generates plots."""
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"Skipping {label} (No checkpoint found)")
            return

        print(f"\nProcessing {label}...")
        
        state_dict = torch.load(ckpt_path)
        if CONFIG['MULTI_GPU'] and not isinstance(model, nn.DataParallel):
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in state_dict.items(): new_state_dict[k[7:]] = v
             state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        for split in ['train', 'validation', 'test']:
            print(f"  Eval on {split}...")
            y_true_e, y_pred_e, y_true_m, y_pred_m = run_inference(model, loaders[split])
            
            m_e = calculate_regression_metrics(y_true_e, y_pred_e)
            m_m = calculate_regression_metrics(y_true_m, y_pred_m)
            
            full_report[label][split] = {'Energy': m_e, 'Mass': m_m}
            
            if plot_energy:
                plot_actual_vs_predicted(y_true_e, y_pred_e, f"{split}_Energy", RUN_NAME, ARCH_DIR)
                plot_residuals(y_true_e, y_pred_e, f"{split}_Energy", RUN_NAME, ARCH_DIR)
            if plot_mass:
                plot_actual_vs_predicted(y_true_m, y_pred_m, f"{split}_Mass", RUN_NAME, ARCH_DIR)
                plot_residuals(y_true_m, y_pred_m, f"{split}_Mass", RUN_NAME, ARCH_DIR)

    eval_and_record(last_saved_loss_path, 'Best Loss Model')

    save_metrics_to_file(full_report, RUN_NAME, ARCH_DIR)
    print(f"\n--- EXPERIMENT COMPLETE: {RUN_NAME} ---")
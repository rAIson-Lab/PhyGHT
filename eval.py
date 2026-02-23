import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import CONFIG

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates MSE, RMSE, MAE, and R2.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def save_metrics_to_file(full_report, run_name, architecture_dir="PhyGHT"):
    """
    Saves a consolidated metrics report to a single text file.
    """
    base_path = os.path.join(CONFIG['METRICS_DIR'], architecture_dir)
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f"{run_name}_metrics.txt")
    print(f"Saving combined metrics to {file_path}...")
    
    with open(file_path, "w") as f:
        f.write(f"FULL EXPERIMENT REPORT: {run_name}\n")
        f.write("=" * 60 + "\n")
        
        for model_ver, splits in full_report.items():
            f.write(f"\n[[ {model_ver} ]]\n")
            f.write("*" * 30 + "\n")
            
            for split_name, metrics in splits.items():
                f.write(f"\n  >>> {split_name} Set <<<\n")
                
                if 'Energy' in metrics or 'Mass' in metrics:
                    for task, values in metrics.items():
                        f.write(f"    [{task}]\n")
                        for m_name, m_val in values.items():
                            f.write(f"      {m_name}: {m_val:.6f}\n")
                else:
                    for m_name, m_val in metrics.items():
                        f.write(f"    {m_name}: {m_val:.6f}\n")
            f.write("-" * 60 + "\n")
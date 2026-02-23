import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def save_plot(fig, filename, subfolder):
    """
    Saves a matplotlib figure to the specified subfolder within the plots directory.
    """
    base_path = os.path.join(CONFIG['PLOTS_DIR'], subfolder)
    os.makedirs(base_path, exist_ok=True)
    
    path = os.path.join(base_path, filename)
    print(f"Saving plot to {path}...")
    fig.savefig(path, dpi=400, bbox_inches='tight')
    plt.close(fig)

def plot_loss_curves(train_losses, val_losses, run_name, subfolder):
    """
    Plots Training vs Validation Loss over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_losses, label='Train Loss', linewidth=2, color='#2c3e50')
    ax.plot(val_losses, label='Val Loss', linewidth=2, color='#e74c3c', linestyle='--')
    
    ax.set_title(f"Loss Curve: {run_name}", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    
    target_dir = os.path.join(subfolder, 'loss_curves')
    save_plot(fig, f"loss_curve_{run_name}.png", target_dir)

def plot_actual_vs_predicted(y_true, y_pred, task_name, run_name, subfolder):
    """
    Plots a 2D Histogram (Heatmap) of Actual vs Predicted values.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 2D Histogram
    h = ax.hist2d(y_true, y_pred, bins=50, cmap='viridis', 
                  range=[[0, 1], [0, 1]], cmin=1)
    
    # Diagonal line representing perfect prediction
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal')
    
    fig.colorbar(h[3], ax=ax, label='Count')
    
    ax.set_title(f"{task_name}: Actual vs Predicted\n{run_name}", fontsize=14)
    ax.set_xlabel(f"True {task_name}")
    ax.set_ylabel(f"Predicted {task_name}")
    ax.legend()
    
    target_dir = os.path.join(subfolder, 'actual_vs_predicted')
    save_plot(fig, f"pred_vs_actual_{task_name}_{run_name}.png", target_dir)

def plot_residuals(y_true, y_pred, task_name, run_name, subfolder):
    """
    Plots the distribution of residuals (Predicted - True).
    """
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='#3498db', ax=ax)
    
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_title(f"{task_name} Residuals Distribution", fontsize=16)
    ax.set_xlabel(f"Residual ({task_name})")
    ax.set_ylabel("Count")
    
    # Add statistics box
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    textstr = r'$\mu={:.4f}$' '\n' r'$\sigma={:.4f}$'.format(mu, sigma)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    target_dir = os.path.join(subfolder, 'residuals')
    save_plot(fig, f"residuals_{task_name}_{run_name}.png", target_dir)
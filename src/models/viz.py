# src/visualize_gnn_results.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set these to your cross-validation results folder
cv_results_dir = 'models/cv_results'
num_folds = 3  # Update as needed based on the number of CV folds you ran

test_mse_scores = []

# Load test MSE scores from each fold
for fold_idx in range(1, num_folds + 1):
    fold_dir = os.path.join(cv_results_dir, f'fold_{fold_idx}')
    # Option: Store fold metrics in a CSV or text file in each fold directory.
    mse_path = os.path.join(fold_dir, 'test_mse.txt')
    if os.path.exists(mse_path):
        with open(mse_path, 'r') as f:
            test_mse = float(f.read().strip())
        test_mse_scores.append(test_mse)
    else:
        print(f"Warning: {mse_path} not found, skipping fold {fold_idx}.")

# ---- 1. Cross-Validation Fold Test MSE Plot ----
plt.figure(figsize=(8, 5))
folds = np.arange(1, len(test_mse_scores) + 1)
plt.bar(folds, test_mse_scores, color='teal')
plt.xlabel('Fold')
plt.ylabel('Test MSE')
plt.title('Test MSE per Cross-Validation Fold')
for i, v in enumerate(test_mse_scores):
    plt.text(i + 1, v + 0.002, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('cv_test_mse_bar.png')
plt.show()

# ---- 2. Training and Validation Loss Curves ----
# Look for train_val_losses.csv saved per fold: columns are 'epoch','train_loss','val_loss'
for fold_idx in range(1, num_folds + 1):
    loss_path = os.path.join(cv_results_dir, f'fold_{fold_idx}', 'train_val_losses.csv')
    if os.path.exists(loss_path):
        losses_df = pd.read_csv(loss_path)
        plt.figure(figsize=(7, 5))
        plt.plot(losses_df['epoch'], losses_df['train_loss'], label=f'Fold {fold_idx} Train')
        plt.plot(losses_df['epoch'], losses_df['val_loss'], label=f'Fold {fold_idx} Val', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'Training & Validation Loss (Fold {fold_idx})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fold_{fold_idx}_train_val_loss.png')
        plt.show()
    else:
        print(f"Warning: {loss_path} not found for fold {fold_idx}.")

# ---- 3. Predicted vs True delay_risk: pick a fold to visualize ----
fold_to_plot = 1  # Change as needed
pred_path = os.path.join(cv_results_dir, f'fold_{fold_to_plot}', 'pred_vs_true_test.csv')
if os.path.exists(pred_path):
    pred_df = pd.read_csv(pred_path)
    plt.figure(figsize=(7, 7))
    plt.scatter(pred_df['true'], pred_df['pred'], alpha=0.5)
    min_val = min(pred_df['true'].min(), pred_df['pred'].min())
    max_val = max(pred_df['true'].max(), pred_df['pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True delay_risk')
    plt.ylabel('Predicted delay_risk')
    plt.title(f'Predicted vs True delay_risk (Fold {fold_to_plot} - Test Set)')
    plt.tight_layout()
    plt.savefig(f'fold_{fold_to_plot}_pred_vs_true_test.png')
    plt.show()
else:
    print(f"Warning: {pred_path} not found for fold {fold_to_plot}.")

print("All visualizations done (plots also saved to disk).")

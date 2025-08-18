# src/viz/gnn_viz.py
"""
Visualization script for GNN model performance in SynchroChain.
Plots training/validation loss curves, compares CV fold results,
and analyzes test set performance.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path to import custom modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(MODELS_DIR, 'saved') # Assuming saved models/results are here
CV_RESULTS_DIR = os.path.join(MODELS_DIR, 'cv_results') # Assuming CV results are saved here

# Paths to specific files (adjust if file names differ)
TRAIN_LOG_PATH = os.path.join(RESULTS_DIR, 'train_gnn_log.txt') # Hypothetical detailed log file
FINAL_MODEL_METRICS_PATH = os.path.join(RESULTS_DIR, 'supply_gnn_final_metrics.json') # If you save final metrics
FINAL_MODEL_TEST_LOSS = 0.0021 # Example from your last run, or load from a file
FINAL_MODEL_TRAIN_LOSS_HISTORY = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002] # Example, or load from a file

# --- 1. Plot Training/Validation Loss Curves ---
def plot_training_curves(train_losses, val_losses, title="GNN Training & Validation Loss", save_path=None):
    """Plots training and validation loss over epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    if val_losses:
        # Ensure val_losses matches epochs length if provided separately
        val_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(val_epochs, val_losses, label='Validation Loss', marker='s')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curve plot saved to {save_path}")
    plt.show()

# --- 2. Plot Cross-Validation Results ---
def plot_cv_results(cv_mse_scores, title="Cross-Validation Fold Results", save_path=None):
    """Plots individual CV fold MSE and summary statistics."""
    if not cv_mse_scores:
        print("Warning: No CV scores provided for plotting.")
        return

    folds = [f"Fold {i+1}" for i in range(len(cv_mse_scores))]
    avg_mse = np.mean(cv_mse_scores)
    std_mse = np.std(cv_mse_scores)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(folds, cv_mse_scores, color='skyblue', label='Fold MSE')
    plt.axhline(y=avg_mse, color='red', linestyle='--', linewidth=2, label=f'Average MSE: {avg_mse:.5f}')
    plt.axhspan(avg_mse - std_mse, avg_mse + std_mse, alpha=0.2, color='red', label=f'+/- 1 Std Dev ({std_mse:.5f})')

    # Add value labels on bars
    for bar, score in zip(bars, cv_mse_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0001, f'{score:.5f}', ha='center', va='bottom', fontsize=9)

    plt.title(title)
    plt.xlabel('CV Fold')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CV results plot saved to {save_path}")
    plt.show()

# --- 3. Compare Final Model vs. Baseline/CV Average ---
def plot_performance_comparison(final_test_mse, cv_avg_mse, cv_std_mse, title="GNN Performance Comparison", save_path=None):
    """Compares final model test MSE against CV average."""
    labels = ['Final Model Test MSE', 'CV Average MSE']
    values = [final_test_mse, cv_avg_mse]
    errors = [0, cv_std_mse] # Assume no error bar for final model for simplicity

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, yerr=errors, capsize=10, color=['gold', 'lightcoral'], edgecolor='black')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(axis='y')

    # Add value labels
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(errors)/10, f'{value:.5f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance comparison plot saved to {save_path}")
    plt.show()

# --- 4. Feature Importance Analysis (Placeholder) ---
def analyze_feature_importance(model_path, data_sample_path, save_path=None):
    """
    Placeholder for feature importance analysis.
    In a full implementation, this would load the trained model,
    use techniques like permutation feature importance or SHAP
    on a sample of the data to determine which input features
    (node attributes) are most influential for the model's predictions.
    """
    print("--- Feature Importance Analysis (Placeholder) ---")
    print("This function would analyze which input features (e.g., inventory_level, delay_risk)")
    print("are most important for the GNN's delay_risk predictions.")
    print("Implementation would involve:")
    print("1. Loading the trained SupplyGNN model.")
    print("2. Loading a sample of GNN data (nodes, edges).")
    print("3. Using a method like Permutation Feature Importance or SHAP:")
    print("   a. Get baseline model performance (e.g., MSE) on the sample.")
    print("   b. For each feature (column in node features):")
    print("       i. Randomly shuffle the values of that feature across all nodes.")
    print("       ii. Measure the model's performance on the shuffled data.")
    print("       iii. The difference (increase in error) indicates the feature's importance.")
    print("4. Plotting the resulting importance scores.")
    print("This requires the model and data to be loaded and is computationally intensive.")
    print("For now, this is a conceptual placeholder.")
    if save_path:
        # Create a dummy plot as a placeholder
        features = ['inventory_level', 'delay_risk', 'carbon_cost', 'supplier_reliability', 'price', 'category_encoded', 'prod_hist_delay', 'reg_hist_delay']
        importances = np.random.rand(len(features)) # Dummy importances
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.xlabel('Importance (Dummy Values)')
        plt.title('(Placeholder) Feature Importance for GNN')
        plt.gca().invert_yaxis() # Highest importance at the top
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Placeholder feature importance plot saved to {save_path}")
        plt.show()

# --- Main Visualization Function ---
def main():
    """Main function to run all visualizations."""
    print("--- Starting GNN Visualization ---")
    viz_output_dir = os.path.join(PROJECT_ROOT, 'reports', 'viz')
    os.makedirs(viz_output_dir, exist_ok=True)
    print(f"Figures will be saved to: {viz_output_dir}")

    # --- Example Data (Replace with actual data loading logic) ---
    # 1. Training History (from log file parsing or saved metrics)
    # For simplicity, using example data. In practice, parse `train_gnn_log.txt` or load from `train_gnn_model.py` metrics.
    example_train_losses = FINAL_MODEL_TRAIN_LOSS_HISTORY # Replace with actual parsed data
    example_val_losses = [0.08, 0.04, 0.025, 0.015, 0.01, 0.008, 0.007, 0.006, 0.0055, 0.0052, 0.0051, 0.0050, 0.0049, 0.0048, 0.0047, 0.0046, 0.0045, 0.0044, 0.0043, 0.0042] # Example
    # Ensure val_losses matches train_losses length if plotting together, or adjust logic

    # 2. Cross-Validation Results (from `cross_validate_gnn.py` output or saved results)
    # Example CV results from your last run
    example_cv_mse_scores = [0.000457, 0.061279] # Add more if you have them, or load from CV log/results
    # If you have the full CV results, load them. Example:
    # try:
    #     with open(os.path.join(CV_RESULTS_DIR, 'cv_results_summary.json'), 'r') as f:
    #         cv_summary = json.load(f)
    #     example_cv_mse_scores = cv_summary['test_mse_scores'] # Assuming this key exists
    # except FileNotFoundError:
    #     print("CV results summary not found, using example data.")
    #     example_cv_mse_scores = [0.0005, 0.02, 0.06] # Fallback example

    final_test_mse = FINAL_MODEL_TEST_LOSS # Example from your last run
    cv_avg_mse = np.mean(example_cv_mse_scores) if example_cv_mse_scores else 0.03
    cv_std_mse = np.std(example_cv_mse_scores) if len(example_cv_mse_scores) > 1 else 0.0

    # --- Generate Plots ---
    print("\nGenerating Training/Validation Loss Curve...")
    plot_training_curves(
        example_train_losses,
        example_val_losses, # Pass None or an empty list if not plotting val
        title="Final GNN Model Training & Validation Loss",
        save_path=os.path.join(viz_output_dir, 'gnn_train_val_loss.png')
    )

    print("\nGenerating Cross-Validation Results Plot...")
    plot_cv_results(
        example_cv_mse_scores,
        title="GNN Cross-Validation Test MSE per Fold",
        save_path=os.path.join(viz_output_dir, 'gnn_cv_folds.png')
    )

    print("\nGenerating Performance Comparison Plot...")
    plot_performance_comparison(
        final_test_mse,
        cv_avg_mse,
        cv_std_mse,
        title="GNN: Final Model vs. Cross-Validation Average",
        save_path=os.path.join(viz_output_dir, 'gnn_performance_comparison.png')
    )

    print("\nGenerating Feature Importance Analysis (Placeholder)...")
    analyze_feature_importance(
        model_path=os.path.join(RESULTS_DIR, 'supply_gnn_final.pth'),
        data_sample_path=os.path.join(DATA_DIR, 'processed', 'temporal_splits', 'test', 'gnn_nodes.csv'), # Example path
        save_path=os.path.join(viz_output_dir, 'gnn_feature_importance_placeholder.png')
    )

    print(f"\n--- All GNN Visualizations Generated ---")
    print(f"Check the directory: {viz_output_dir}")

if __name__ == '__main__':
    main()
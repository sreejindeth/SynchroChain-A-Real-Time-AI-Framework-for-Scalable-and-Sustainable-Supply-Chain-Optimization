# src/models/cross_validate_gnn.py
"""
Script to perform Time-Series Cross-Validation on the Supply Chain GNN model.
"""
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import sys

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary functions from your existing scripts
# Make sure these imports are correct based on your project structure
from preprocessing import preprocess_gnn_data, load_data # Adjust import path if needed
from models.train_gnn_model import load_and_prepare_gnn_data, train_gnn_model_with_validation, evaluate_gnn_model # Adjust import path if needed
from models.supply_gnn import SupplyGNN # Adjust import path if needed

def run_single_fold(fold_num, train_dates, val_dates, test_dates, df_orders_raw, df_access_processed, cv_results_dir, device):
    """
    Runs training and evaluation for a single CV fold.
    """
    print(f"\n========== Running CV Fold {fold_num} ==========")
    print(f"Train Dates:   {train_dates[0]} to {train_dates[1]}")
    print(f"Val Dates:     {val_dates[0]} to {val_dates[1]}")
    print(f"Test Dates:    {test_dates[0]} to {test_dates[1]}")
    
    fold_dir = os.path.join(cv_results_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # --- 1. Preprocess Data for this fold ---
    print("  --- Preprocessing Data for Fold ---")
    # Convert string dates to datetime for filtering
    train_start_dt = pd.to_datetime(train_dates[0]) if train_dates[0] else None
    train_end_dt = pd.to_datetime(train_dates[1]) if train_dates[1] else None
    val_start_dt = pd.to_datetime(val_dates[0]) if val_dates[0] else None
    val_end_dt = pd.to_datetime(val_dates[1]) if val_dates[1] else None
    test_start_dt = pd.to_datetime(test_dates[0]) if test_dates[0] else None
    test_end_dt = pd.to_datetime(test_dates[1]) if test_dates[1] else None

    # Use a temporary directory for this fold's data
    fold_temp_data_dir = os.path.join(fold_dir, "temp_data")
    os.makedirs(fold_temp_data_dir, exist_ok=True)

    # --- Process Train Data ---
    train_nodes_df, train_edges_df, _ = preprocess_gnn_data(
        df_orders_raw, df_access_processed, start_date=train_start_dt, end_date=train_end_dt
    )
    train_nodes_path = os.path.join(fold_temp_data_dir, "train_gnn_nodes.csv")
    train_edges_path = os.path.join(fold_temp_data_dir, "train_gnn_edges.csv")
    train_nodes_df.to_csv(train_nodes_path, index=False)
    train_edges_df.to_csv(train_edges_path, index=False)

    # --- Process Val Data ---
    val_nodes_df, val_edges_df, _ = preprocess_gnn_data(
        df_orders_raw, df_access_processed, start_date=val_start_dt, end_date=val_end_dt
    )
    val_nodes_path = os.path.join(fold_temp_data_dir, "val_gnn_nodes.csv")
    val_edges_path = os.path.join(fold_temp_data_dir, "val_gnn_edges.csv")
    val_nodes_df.to_csv(val_nodes_path, index=False)
    val_edges_df.to_csv(val_edges_path, index=False)

    # --- Process Test Data ---
    test_nodes_df, test_edges_df, _ = preprocess_gnn_data(
        df_orders_raw, df_access_processed, start_date=test_start_dt, end_date=test_end_dt
    )
    test_nodes_path = os.path.join(fold_temp_data_dir, "test_gnn_nodes.csv")
    test_edges_path = os.path.join(fold_temp_data_dir, "test_gnn_edges.csv")
    test_nodes_df.to_csv(test_nodes_path, index=False)
    test_edges_df.to_csv(test_edges_path, index=False)

    # --- 2. Load and Prepare Data ---
    print("  --- Loading and Preparing Data ---")
    train_data, train_targets, _, _, _ = load_and_prepare_gnn_data(train_nodes_path, train_edges_path)
    val_data, val_targets, _, _, _ = load_and_prepare_gnn_data(val_nodes_path, val_edges_path)
    test_data, test_targets, _, _, _ = load_and_prepare_gnn_data(test_nodes_path, test_edges_path)
    num_node_features = train_data.x.shape[1]
    print(f"  Fold {fold_num} data prepared. Num features: {num_node_features}")

    
    # --- 3. Initialize Model (WITH TUNED HYPERPARAMETERS) ---
    print("  --- Initializing Model (Using Tuned Hyperparameters) ---")
    # Use the best parameters found by Optuna
    model = SupplyGNN(
        num_node_features=num_node_features,
        hidden_dim=64,       # <-- Best from tuning
        output_dim=32,       # <-- Example: half of hidden_dim, adjust if needed
        num_heads=1,        # <-- Best from tuning
        num_layers=3,       # <-- Best from tuning
        dropout=0.0865,     # <-- Best from tuning (or round to 0.09)
        # Note: learning_rate, weight_decay, num_epochs are optimizer/training args,
        #       not model __init__ args. They are passed to train_gnn_model_with_validation.
    )
    print(f"  Fold {fold_num} model initialized with tuned hyperparameters.")
    # ------------------------

    # --- 4. Train Model with Validation ---
    print("  --- Initiating Training with Validation ---")
    # Use a reduced number of epochs for CV speed, or use early stopping effectively
    train_gnn_model_with_validation(
        model, train_data, val_data, train_targets, val_targets,
        num_epochs=20, # Reduced for CV speed, adjust if needed
        learning_rate=0.000332, # <-- Best from tuning (or 0.0003)
        weight_decay=0.000861,  # <-- Best from tuning (or 0.0009)
        device=device
    )
    print("  --- Training Completed ---")

    # --- 5. Evaluate on Test Set ---
    print("  --- Initiating Test Evaluation ---")
    model.eval()
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    criterion = nn.MSELoss()
    with torch.no_grad():
        _, pred_test, _ = model(test_data)
        test_loss = criterion(pred_test, test_targets).item()
    print(f"  --- Fold {fold_num} Test MSE: {test_loss:.6f} ---")
    
    # --- 6. Cleanup temporary data for this fold ---
    # shutil.rmtree(fold_temp_data_dir) # Uncomment if you want to delete temp data after each fold

    return test_loss

def main():
    """Main cross-validation function."""
    print("--- Starting Time-Series Cross-Validation for Supply GNN ---")
    start_time = datetime.now()
    
    # --- Configuration ---
    raw_orders_path = 'data/raw/DataCoSupplyChainDataset.csv'
    raw_access_path = 'data/raw/tokenized_access_logs.csv' # Make sure this points to the correct file with data
    
    # --- Define Temporal Folds ---
    # Example: 3 folds. Adjust dates and number of folds as needed.
    # IMPORTANT: These dates need to be within the range of your actual data.
    # Based on your previous output, data spans 2015-01-01 to 2018-01-31.
    # Let's define 3 folds, each roughly 1 year, with a 3-month gap for val/test.
    # You can refine these splits.
    folds_definition = [
        {
            "name": "Fold 1",
            "train": ("2015-01-01", "2015-12-31"),
            "val":   ("2016-01-01", "2016-03-31"),
            "test":  ("2016-04-01", "2016-06-30")
        },
        {
            "name": "Fold 2",
            "train": ("2015-07-01", "2016-06-30"),
            "val":   ("2016-07-01", "2016-09-30"),
            "test":  ("2016-10-01", "2016-12-31")
        },
        {
            "name": "Fold 3",
            "train": ("2016-01-01", "2016-12-31"),
            "val":   ("2017-01-01", "2017-03-31"),
            "test":  ("2017-04-01", "2017-06-30")
        }
        # Add more folds if desired
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv_results_dir = 'models/cv_results'
    os.makedirs(cv_results_dir, exist_ok=True)
    # -----------------------

    # --- 1. Load Raw Data Once ---
    print("Loading raw data...")
    df_raw_orders = load_data(raw_orders_path, is_access_log=False)
    df_raw_access = load_data(raw_access_path, is_access_log=True)
    # Preprocess access logs once (assuming it doesn't depend on order dates directly)
    df_processed_access = preprocess_gnn_data.__globals__['preprocess_access_logs'](df_raw_access) # Hacky way to call it, better to import directly
    # Or if preprocess_access_logs is in preprocess_data module:
    # from preprocessing.preprocess_data import preprocess_access_logs
    # df_processed_access = preprocess_access_logs(df_raw_access)
    print("Raw data loaded and access logs preprocessed.")
    # ----------------------------

    # --- 2. Run Cross-Validation Folds ---
    test_mse_scores = []
    for i, fold_def in enumerate(folds_definition):
        fold_num = i + 1
        try:
            test_mse = run_single_fold(
                fold_num,
                fold_def["train"], fold_def["val"], fold_def["test"],
                df_raw_orders, df_processed_access,
                cv_results_dir, device
            )
            test_mse_scores.append(test_mse)
            print(f"Fold {fold_num} completed. Test MSE: {test_mse:.6f}")
        except Exception as e:
            print(f"Error occurred in Fold {fold_num}: {e}")
            # Depending on requirements, you might want to stop or continue
            # For now, let's continue and mark this fold as failed
            test_mse_scores.append(np.nan) # Use NaN to indicate failure
    
    # --- 3. Analyze and Report Results ---
    print("\n========== Cross-Validation Complete ==========")
    valid_scores = [score for score in test_mse_scores if not np.isnan(score)]
    if valid_scores:
        avg_mse = np.mean(valid_scores)
        std_mse = np.std(valid_scores)
        min_mse = np.min(valid_scores)
        max_mse = np.max(valid_scores)
        print(f"Number of completed folds: {len(valid_scores)} out of {len(folds_definition)}")
        print(f"Average Test MSE: {avg_mse:.6f}")
        print(f"Standard Deviation of Test MSE: {std_mse:.6f}")
        print(f"Min Test MSE: {min_mse:.6f}")
        print(f"Max Test MSE: {max_mse:.6f}")
        print("-----------------------------------------------")
        print("Interpretation:")
        print("- A low average MSE indicates good overall performance.")
        print("- A low standard deviation indicates consistent performance across time periods.")
        print("- High variance might suggest the model's performance is sensitive to the time period.")
    else:
        print("All folds failed. Please check the errors reported above.")
    print("================================================")
    end_time = datetime.now()
    print(f"--- Cross-Validation Finished ({end_time - start_time}) ---")

if __name__ == '__main__':
    main()

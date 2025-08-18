# src/models/tune_gnn.py
"""
Script for Hyperparameter Tuning of the Supply GNN model using Optuna.
"""
import os
import sys
import torch
import torch.nn as nn
import optuna
from functools import partial

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary functions from your existing training script
# Adjust import paths as needed based on your project structure
from models.train_gnn_model import load_and_prepare_gnn_data, train_gnn_model_with_validation, evaluate_gnn_model
from models.supply_gnn import SupplyGNN

# --- Configuration ---
TRAIN_NODES_FILE = 'data/processed/temporal_splits/train/gnn_nodes.csv'
TRAIN_EDGES_FILE = 'data/processed/temporal_splits/train/gnn_edges.csv'
VAL_NODES_FILE = 'data/processed/temporal_splits/val/gnn_nodes.csv'
VAL_EDGES_FILE = 'data/processed/temporal_splits/val/gnn_edges.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = 'models/tuned'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# --- End Configuration ---

def load_data_for_tuning(train_nodes_file, train_edges_file, val_nodes_file, val_edges_file):
    """Wrapper to load and prepare training and validation data."""
    print("Loading training data for tuning...")
    train_data, train_targets, _, _, _ = load_and_prepare_gnn_data(train_nodes_file, train_edges_file)
    print("Loading validation data for tuning...")
    val_data, val_targets, _, _, _ = load_and_prepare_gnn_data(val_nodes_file, val_edges_file)
    num_node_features = train_data.x.shape[1]
    print(f"Data loaded. Num features: {num_node_features}")
    return train_data, train_targets, val_data, val_targets, num_node_features

def objective(trial, data_loaders):
    """
    Objective function for Optuna to minimize (validation MSE).
    """
    train_data, train_targets, val_data, val_targets, num_node_features = data_loaders

    # --- Suggest Hyperparameters ---
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    # Ensure num_heads divides hidden_dim
    if hidden_dim % num_heads != 0:
        # If not divisible, skip this trial
        raise optuna.TrialPruned()

    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    # -----------------------------

    # --- Create Model ---
    model = SupplyGNN(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim // 2, # Example: half of hidden_dim
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    print(f"Trial {trial.number} - Model created with params: "
          f"hd={hidden_dim}, nl={num_layers}, nh={num_heads}, do={dropout:.2f}, "
          f"lr={learning_rate:.6f}, wd={weight_decay:.2e}")

    # --- Train Model ---
    # Use a reduced number of epochs for faster trials
    num_epochs = trial.suggest_int('num_epochs', 10, 20) # Allow Optuna to choose epochs too, within a range

    try:
        train_gnn_model_with_validation(
            model, train_data, val_data, train_targets, val_targets,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=DEVICE
        )
    except Exception as e:
        print(f"Trial {trial.number} failed during training: {e}")
        # Return a large value to indicate failure
        return float('inf')

    # --- Evaluate on Validation Set (Final model state) ---
    model.eval()
    val_data = val_data.to(DEVICE)
    val_targets = val_targets.to(DEVICE)
    criterion = nn.MSELoss()
    with torch.no_grad():
        _, pred_val, _ = model(val_data)
        val_loss = criterion(pred_val, val_targets).item()
    print(f"Trial {trial.number} finished. Validation MSE: {val_loss:.6f}")
    return val_loss # Optuna will try to minimize this

def main():
    """Main tuning function."""
    print("--- Starting Hyperparameter Tuning for Supply GNN ---")

    # --- 1. Load Data Once ---
    print("Preparing data for tuning...")
    data_tuple = load_data_for_tuning(TRAIN_NODES_FILE, TRAIN_EDGES_FILE, VAL_NODES_FILE, VAL_EDGES_FILE)
    print("Data preparation complete.")

    # --- 2. Define Objective with Data ---
    objective_with_data = partial(objective, data_loaders=data_tuple)

    # --- 3. Create Optuna Study ---
    study_name = "supply_gnn_tuning"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True # Resume if study exists
    )

    # --- 4. Run Optimization ---
    n_trials = 20 # Number of different hyperparameter combinations to try
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective_with_data, n_trials=n_trials)

    # --- 5. Print Results ---
    print("\n--- Tuning Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Validation MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print(f"\n--- To use the best parameters ---")
    print("Update the parameters in `src/models/train_gnn_model.py` or retrain with these settings.")
    print(f"Best model can potentially be found in the Optuna database or retrained.")

if __name__ == '__main__':
    main()
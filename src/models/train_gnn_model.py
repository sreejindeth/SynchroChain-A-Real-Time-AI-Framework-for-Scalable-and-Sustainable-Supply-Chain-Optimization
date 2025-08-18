# src/models/train_gnn_model.py
"""
Script to train the Supply Chain GNN model with validation and final test evaluation.
"""
import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader # Kept for potential future use
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder # Added for feature scaling

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Import the GNN model
from models.supply_gnn import SupplyGNN
# Import or replicate necessary data loading/prep logic if needed
# from preprocessing.preprocess_data import load_data # Uncomment if needed and path is correct

def load_and_prepare_gnn_data(nodes_path, edges_path):
    """
    Loads processed node/edge CSVs and converts them into a PyG Data object.
    This version focuses on preparing features correctly.
    """
    print(f"Loading nodes from {nodes_path} and edges from {edges_path} ...")
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    print(f"Nodes loaded: {nodes_df.shape}, Edges loaded: {edges_df.shape}")

    # --- 1. Encode Node Types (Categorical) ---
    le_node_type = LabelEncoder()
    nodes_df['node_type_encoded'] = le_node_type.fit_transform(nodes_df['node_type'])
    print(f"Node types encoded: {le_node_type.classes_}")

    # --- 2. Handle Features ---
    # Identify numerical features for nodes (matching preprocessing)
    # Ensure this list matches the features generated in preprocessing
    feature_columns = [
        'node_type_encoded', 'Product Price', 'inventory_level', 
        'delay_risk', 'carbon_cost', 'supplier_reliability',
        'product_historical_delay_rate', 'region_historical_delay_rate'
    ]
    # Fill any remaining NaNs (should be minimal after preprocessing)
    nodes_df[feature_columns] = nodes_df[feature_columns].fillna(0)
    
    # --- Scaling (Standardization) - Important for GNNs ---
    # Note: In a production setting, you would load a scaler fitted on the training data.
    # For this script, we fit it here for the specific dataset split being loaded.
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(nodes_df[feature_columns])
    x = torch.tensor(node_features_scaled, dtype=torch.float)
    print(f"Node feature matrix (x) shape: {x.shape}")
    print(f"Node feature matrix (x) - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
    # Specifically check the 'delay_risk' column (index might vary, but should be consistent)
    delay_risk_col_idx = feature_columns.index('delay_risk')
    print(f"Target 'delay_risk' values (col {delay_risk_col_idx}) - min: {x[:, delay_risk_col_idx].min().item():.4f}, max: {x[:, delay_risk_col_idx].max().item():.4f}")

    # --- 3. Create Edge Index ---
    # Map node_id strings to integer indices based on the order in nodes_df
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'])}
    
    # Map source and target node_ids to indices
    edges_df['source_idx'] = edges_df['source_node_id'].map(node_id_to_idx)
    edges_df['target_idx'] = edges_df['target_node_id'].map(node_id_to_idx)
    
    # Check for any unmapped nodes (should be rare after good preprocessing)
    unmapped_sources = edges_df['source_idx'].isna().sum()
    unmapped_targets = edges_df['target_idx'].isna().sum()
    if unmapped_sources > 0 or unmapped_targets > 0:
        print(f"Warning: {unmapped_sources} source and {unmapped_targets} target nodes could not be mapped. Dropping edges.")
        edges_df.dropna(subset=['source_idx', 'target_idx'], inplace=True)
        edges_df.reset_index(drop=True, inplace=True) # Reset index after dropping
        
    # Create edge_index tensor (2, num_edges)
    if edges_df.empty:
        raise ValueError("No valid edges remaining after mapping. Check node_id consistency between nodes_df and edges_df.")
        
    # Convert to numpy first to avoid the warning and potential speed issue
    source_indices = edges_df['source_idx'].astype(int).values
    target_indices = edges_df['target_idx'].astype(int).values
    edge_index = torch.tensor(
        np.array([source_indices, target_indices]), # Use np.array first
        dtype=torch.long
    )
    print(f"Edge index shape: {edge_index.shape}")

    # --- 4. Create PyG Data Object ---
    data = Data(x=x, edge_index=edge_index)
    # Optionally, store scaler, encoder, node_id mapping if needed for inference/saving
    # data.node_id_to_idx = node_id_to_idx
    # data.scaler = scaler
    # data.le_node_type = le_node_type
    
    print("PyG Data object created.")
    return data, torch.tensor(nodes_df['delay_risk'].values, dtype=torch.float).unsqueeze(1), scaler, le_node_type, node_id_to_idx

def train_gnn_model_with_validation(model, train_data, val_data, target_train, target_val,
                                    num_epochs=20, learning_rate=0.01, weight_decay=0.0, device='cpu'):
    """
    Trains the GNN model with validation.
    Args:
        model (torch.nn.Module): The GNN model to train.
        train_data (torch_geometric.data.Data): Training PyG data object.
        val_data (torch_geometric.data.Data): Validation PyG data object.
        target_train (torch.Tensor): Training targets (e.g., delay_risk).
        target_val (torch.Tensor): Validation targets (e.g., delay_risk).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer. Defaults to 0.0.
        device (str): Device to train on ('cpu' or 'cuda').
    """
    model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    target_train = target_train.to(device)
    target_val = target_val.to(device)

    criterion = nn.MSELoss()
    # --- FIX 1: Include weight_decay in the optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # -----------------------------------------------------

    # --- Training Loop ---
    model.train()
    # --- FIX 2: Initialize variables for early stopping ---
    best_val_loss = float('inf')
    patience = 5 # Example patience, adjust as needed or make it a parameter
    patience_counter = 0
    # -------------------
    
    print("--- Starting Training Loop with Validation ---")
    for epoch in range(num_epochs):
        # --- Training Step ---
        optimizer.zero_grad()
        _, pred_train, _ = model(train_data)
        loss_train = criterion(pred_train, target_train)
        loss_train.backward()
        optimizer.step()
        train_loss_value = loss_train.item() # Store training loss value

        # --- Validation Step ---
        model.eval()
        with torch.no_grad():
            _, pred_val, _ = model(val_data)
            loss_val = criterion(pred_val, target_val)
            # --- FIX 3: Correctly define the validation loss value variable ---
            val_loss_value = loss_val.item() # Use a consistent name like _value
            # ---------------------------------------------------------------
        model.train() # Set back to train mode

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss_value:.4f} Val Loss: {val_loss_value:.4f}")

        # --- Early Stopping Check ---
        # --- FIX 4: Use the correctly defined variable for early stopping ---
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs with no improvement.")
                break
        # --------------------------
    print("--- Training Loop Completed ---")

def evaluate_gnn_model(model, test_data, target_test, device='cpu'):
    """
    Evaluates the trained GNN model on the test set.
    """
    model.eval()
    test_data = test_data.to(device)
    target_test = target_test.to(device)
    criterion = nn.MSELoss()
    with torch.no_grad():
        _, pred_test, _ = model(test_data)
        loss_test = criterion(pred_test, target_test).item()
    print(f"--- Test Set Evaluation ---")
    print(f"Test MSE Loss: {loss_test:.4f}")
    print("--------------------------")

def main():
    """Main GNN training function."""
    print("--- Training Supply GNN with Validation and Test Evaluation ---")
    
    # --- Configuration ---
    base_data_dir = 'data/processed/temporal_splits'
    train_nodes_file = os.path.join(base_data_dir, 'train', 'gnn_nodes.csv')
    train_edges_file = os.path.join(base_data_dir, 'train', 'gnn_edges.csv')
    val_nodes_file = os.path.join(base_data_dir, 'val', 'gnn_nodes.csv')
    val_edges_file = os.path.join(base_data_dir, 'val', 'gnn_edges.csv')
    test_nodes_file = os.path.join(base_data_dir, 'test', 'gnn_nodes.csv')
    test_edges_file = os.path.join(base_data_dir, 'test', 'gnn_edges.csv')
    
    num_epochs = 20 # Example, adjust as needed or use tuned value
    learning_rate = 0.000332 # Example from tuning, adjust as needed
    weight_decay = 0.000861  # Example from tuning, adjust as needed
    patience = 5 # Example patience for early stopping, adjust as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'models/saved/supply_gnn_final.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # ---------------------

    # --- 1. Load and Prepare Data ---
    print("\n--- Loading TRAINING Data ---")
    train_data, train_targets, train_scaler, train_le_node_type, train_node_id_map = load_and_prepare_gnn_data(train_nodes_file, train_edges_file)
    num_node_features = train_data.x.shape[1]
    print(f"Training data prepared. Num features: {num_node_features}, Device: {device}")

    print("\n--- Loading VALIDATION Data ---")
    val_data, val_targets, _, _, _ = load_and_prepare_gnn_data(val_nodes_file, val_edges_file)
    print("Validation data prepared.")

    print("\n--- Loading TEST Data ---")
    test_data, test_targets, _, _, _ = load_and_prepare_gnn_data(test_nodes_file, test_edges_file)
    print("Test data prepared.")
    # -------------------------------

    # --- 2. Initialize Model (Using Tuned Hyperparameters) ---
    model = SupplyGNN(
        num_node_features=num_node_features,
        hidden_dim=64,        # <-- Best from tuning
        output_dim=32,        # <-- Example: half of hidden_dim, adjust if needed
        num_heads=1,          # <-- Best from tuning
        num_layers=3,          # <-- Best from tuning
        dropout=0.0865,       # <-- Best from tuning (or round to 0.09)
    )
    print(f"\nGNN Model initialized. Num features: {num_node_features}, Device: {device}")
    # ------------------------

    # --- 3. Train Model with Validation ---
    print(f"\n--- Initiating Training (Max Epochs: {num_epochs}) ---")
    train_gnn_model_with_validation(
        model, train_data, val_data, train_targets, val_targets,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay, # Pass the tuned weight decay
        device=device
    )
    # -------------------------------------

    # --- 4. Evaluate on Test Set ---
    print("\n--- Initiating Test Evaluation ---")
    evaluate_gnn_model(model, test_data, test_targets, device=device)
    # -------------------------------

    # --- 5. Save Final Trained Model ---
    # Note: This saves the model in its final state after training/early stopping.
    # The cross-validation script uses this logic per fold.
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal GNN Model saved to {model_save_path}")
    # ---------------------------------

if __name__ == '__main__':
    main()

# src/models/tune_intent_model.py
"""
Script for hyperparameter tuning of the Intent Transformer model using Optuna.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import optuna
from functools import partial
import time # Import time for basic timing

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary functions from your existing training script
from models.intent_transformer import IntentTransformer, create_sequences

# --- Configuration (can be adjusted) ---
DATA_FILE = 'data/processed/processed_access_logs.csv'
SEQUENCE_LENGTH = 10
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Using CPU for tuning trials to make it more accessible, can be changed.
DEVICE = 'cpu' 
# --- End Configuration ---

def load_and_prepare_data_for_tuning(file_path, sequence_length):
    """Reuses logic from train_intent_model.py to load and prep data."""
    print("Loading processed access logs for tuning...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape}")

    df.dropna(subset=['Product Card Id'], inplace=True)
    df['Product Card Id'] = df['Product Card Id'].astype(int)

    unique_products = df['Product Card Id'].unique()
    product_to_idx = {pid: idx+1 for idx, pid in enumerate(unique_products)}
    vocab_size = len(product_to_idx) + 1
    print(f"Vocabulary size (products + padding): {vocab_size}")

    unique_users = df['user_id'].unique()
    user_to_idx = {uid: idx+1 for idx, uid in enumerate(unique_users)}

    sequences, targets_intent, targets_urgency, targets_nav_depth = create_sequences(
        df, product_to_idx, user_to_idx, sequence_length
    )
    print(f"Created {len(sequences)} sequences.")

    if not sequences:
        raise ValueError("No sequences created.")

    X = torch.tensor(sequences, dtype=torch.long)
    y_intent = torch.tensor(targets_intent, dtype=torch.float32).unsqueeze(1)
    y_urgency = torch.tensor(targets_urgency, dtype=torch.float32).unsqueeze(1)
    y_nav_depth = torch.tensor(targets_nav_depth, dtype=torch.float32).unsqueeze(1)

    return X, y_intent, y_urgency, y_nav_depth, vocab_size

def objective(trial, data_loaders_and_vocab):
    """Objective function for Optuna to minimize."""
    X, y_intent, y_urgency, y_nav_depth, vocab_size = data_loaders_and_vocab
    
    # --- Suggest Hyperparameters ---
    # Use smaller, potentially faster ranges for initial tuning on full data
    embedding_dim = trial.suggest_categorical('embedding_dim', [32, 64]) # Reduced options
    num_heads = trial.suggest_categorical('num_heads', [2, 4]) # Reduced options
    # Ensure num_heads divides embedding_dim
    if embedding_dim % num_heads != 0:
        raise optuna.TrialPruned()
        
    num_layers = trial.suggest_int('num_layers', 1, 3) # Reduced range
    dropout = trial.suggest_float('dropout', 0.0, 0.3) # Reduced range
    
    # Training hyperparameters - Favor faster training for trials
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Narrower range
    batch_size = trial.suggest_categorical('batch_size', [32, 64]) # Smaller batches might be slower per epoch but more stable
    num_epochs = trial.suggest_int('num_epochs', 2, 4) # Reduced epochs

    print(f"--- Starting Trial {trial.number} ---")
    print(f"  Hyperparameters: emb_dim={embedding_dim}, heads={num_heads}, layers={num_layers}, "
          f"dropout={dropout:.2f}, lr={learning_rate:.6f}, bs={batch_size}, epochs={num_epochs}")

    # --- Create Model ---
    model = IntentTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    model.to(DEVICE)

    # --- Create Data Loader ---
    dataset = TensorDataset(X, y_intent, y_urgency, y_nav_depth)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(dataloader)
    print(f"  Dataset size: {len(dataset)}, Batches per epoch: {total_batches}")

    # --- Define Loss and Optimizer ---
    criterion_intent = nn.BCELoss()
    criterion_urgency = nn.BCELoss()
    criterion_nav_depth = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop (Simplified for tuning) ---
    model.train()
    total_loss_over_trials = 0
    num_batches_evaluated = 0
    trial_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches_in_epoch = 0
        print(f"  Starting Epoch {epoch+1}/{num_epochs}...")
        
        for batch_idx, (data, target_intent, target_urgency, target_nav_depth) in enumerate(dataloader):
            data = data.to(DEVICE)
            target_intent = target_intent.to(DEVICE)
            target_urgency = target_urgency.to(DEVICE)
            target_nav_depth = target_nav_depth.to(DEVICE)

            optimizer.zero_grad()
            intent_logits, urgency_logits, nav_depth_logits = model(data)
            
            loss_intent = criterion_intent(intent_logits, target_intent)
            loss_urgency = criterion_urgency(urgency_logits, target_urgency)
            loss_nav_depth = criterion_nav_depth(nav_depth_logits, target_nav_depth)
            
            loss = loss_intent + loss_urgency + 0.1 * loss_nav_depth 
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches_in_epoch += 1
            
            # --- Detailed Batch Progress (every 10% of epoch or last batch) ---
            if batch_idx == 0 or batch_idx == total_batches - 1 or (batch_idx + 1) % max(1, total_batches // 10) == 0:
                 print(f"    Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}")

            # Accumulate loss for reporting (e.g., last epoch's average loss)
            if epoch == num_epochs - 1: # Only track loss in the last epoch
                 total_loss_over_trials += loss.item()
                 num_batches_evaluated += 1

        # --- Epoch Summary ---
        avg_epoch_loss = epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f}s")

    # --- Calculate and Return Objective Metric ---
    trial_duration = time.time() - trial_start_time
    if num_batches_evaluated == 0:
        final_loss = float('inf') 
    else:
        final_loss = total_loss_over_trials / num_batches_evaluated
    print(f"--- Trial {trial.number} finished ---")
    print(f"  Final Avg Loss (last epoch): {final_loss:.4f}")
    print(f"  Total Time: {trial_duration:.2f}s")
    return final_loss # Optuna will try to minimize this

def main():
    """Main tuning function."""
    print("--- Starting Hyperparameter Tuning for Intent Transformer ---")
    
    # --- 1. Load Data Once ---
    print("Preparing data for tuning...")
    data_tuple = load_and_prepare_data_for_tuning(DATA_FILE, SEQUENCE_LENGTH)
    print("Data preparation complete.")
    
    # --- 2. Define Objective with Data ---
    objective_with_data = partial(objective, data_loaders_and_vocab=data_tuple)

    # --- 3. Create Optuna Study ---
    study_name = "intent_transformer_tuning"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    
    # --- 4. Run Optimization ---
    # Reduce number of trials for initial full-data run to make it manageable
    # You can increase this later if the detailed output is helpful and runs are not too long
    n_trials = 10 # Reduced number of trials for the full-data run
    print(f"Starting optimization with {n_trials} trials...")
    print("=" * 50)
    study.optimize(objective_with_data, n_trials=n_trials)
    
    # --- 5. Print Results ---
    print("\n" + "=" * 50)
    print("--- Tuning Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print("\n--- To use the best parameters ---")
    print("Update the parameters in `src/models/train_intent_model.py` with these values and retrain.")

if __name__ == '__main__':
    main()
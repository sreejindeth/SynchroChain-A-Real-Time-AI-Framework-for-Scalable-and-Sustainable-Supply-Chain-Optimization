# src/models/train_intent_model.py
"""
Script to train the Intent Transformer model (Debug Version) with validation.
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys
import numpy as np

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.intent_transformer import IntentTransformer, create_sequences

def load_and_prepare_data(file_path, sequence_length=10):
    """Loads processed access logs and prepares data for the Transformer."""
    print("Loading processed access logs...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape}")

    # Handle missing Product Card Id (from preprocessing mismatch)
    df.dropna(subset=['Product Card Id'], inplace=True)
    df['Product Card Id'] = df['Product Card Id'].astype(int)

    # --- 1. Encode Categorical Features ---
    unique_products = df['Product Card Id'].unique()
    product_to_idx = {pid: idx+1 for idx, pid in enumerate(unique_products)} # 0 reserved for padding
    vocab_size = len(product_to_idx) + 1 # +1 for padding token
    print(f"Vocabulary size (products + padding): {vocab_size}")
    sample_mappings = {k: product_to_idx[k] for k in list(product_to_idx.keys())[:3]}
    print(f"Sample product_to_idx mappings: {sample_mappings}")

    unique_users = df['user_id'].unique()
    user_to_idx = {uid: idx+1 for idx, uid in enumerate(unique_users)} # 0 for unknown/padding

    # --- 2. Create Sequences ---
    print("Creating sequences...")
    sequences, targets_intent, targets_urgency, targets_nav_depth = create_sequences(
        df, product_to_idx, user_to_idx, sequence_length
    )
    print(f"Created {len(sequences)} sequences.")

    if not sequences:
        raise ValueError("No sequences created. Check data or sequence creation logic.")

    # --- 3. Convert to Tensors ---
    X = torch.tensor(sequences, dtype=torch.long) # (num_samples, seq_len)
    y_intent = torch.tensor(targets_intent, dtype=torch.float32).unsqueeze(1) # (num_samples, 1)
    y_urgency = torch.tensor(targets_urgency, dtype=torch.float32).unsqueeze(1)
    y_nav_depth = torch.tensor(targets_nav_depth, dtype=torch.float32).unsqueeze(1)

    return X, y_intent, y_urgency, y_nav_depth, vocab_size

def validate_model(model, dataloader, device):
    model.eval()
    correct_intent_preds = 0
    total_intent_preds = 0
    total_loss = 0.0
    criterion_intent = nn.BCELoss()
    with torch.no_grad():
        for data, target_intent, target_urgency, target_nav_depth in dataloader:
            data = data.to(device)
            target_intent = target_intent.to(device)

            intent_logits, _, _ = model(data)
            loss = criterion_intent(intent_logits, target_intent)
            total_loss += loss.item()

            predicted_intent = (intent_logits > 0.5).float()
            correct_intent_preds += (predicted_intent == target_intent).sum().item()
            total_intent_preds += target_intent.numel()
    avg_loss = total_loss / len(dataloader)
    acc = correct_intent_preds / total_intent_preds if total_intent_preds > 0 else 0.0
    return avg_loss, acc

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-4, device='cpu'):
    """Training loop with validation for simplified intent task and debugging."""
    model.to(device)
    criterion_intent = nn.BCELoss()
    criterion_urgency = nn.MSELoss()
    criterion_nav_depth = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        correct_intent_preds = 0
        total_intent_preds = 0

        for batch_idx, (data, target_intent, target_urgency, target_nav_depth) in enumerate(train_loader):
            data = data.to(device)
            target_intent = target_intent.to(device)
            target_urgency = target_urgency.to(device)
            target_nav_depth = target_nav_depth.to(device)

            optimizer.zero_grad()
            intent_logits, urgency_logits, nav_depth_logits = model(data)
            
            loss_intent = criterion_intent(intent_logits, target_intent)
            loss_urgency = criterion_urgency(urgency_logits, target_urgency)
            loss_nav_depth = criterion_nav_depth(nav_depth_logits, target_nav_depth)
            
            loss = loss_intent + loss_urgency + 0.1 * loss_nav_depth
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                predicted_intent = (intent_logits > 0.5).float()
                correct_intent_preds += (predicted_intent == target_intent).sum().item()
                total_intent_preds += target_intent.numel()

            if batch_idx == 0 and epoch == 0:
                print(f"\n--- DEBUG Model Outputs & Targets (Epoch {epoch+1}, Batch {batch_idx}) ---")
                print(f"Sample Target Intent (first 5): {target_intent[:5].flatten()}")
                print(f"Sample Raw Logits (first 5): {intent_logits[:5].flatten()}")
                import torch.nn.functional as F
                predicted_probs = F.sigmoid(intent_logits)
                print(f"Sample Predicted Probabilities (first 5): {predicted_probs[:5].flatten()}")
                saturated_count = ((predicted_probs < 0.01) | (predicted_probs > 0.99)).sum().item()
                total_count = predicted_probs.numel()
                print(f"Saturated Predictions: {saturated_count}/{total_count}")
                print(f"--- END DEBUG ---\n")

            if batch_idx % 200 == 0:
                batch_acc = (predicted_intent == target_intent).sum().item() / target_intent.numel()
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], '
                      f'Loss: {loss.item():.4f}, Intent Acc: {batch_acc:.4f}')

        avg_loss = total_loss / num_batches
        epoch_acc = correct_intent_preds / total_intent_preds if total_intent_preds > 0 else 0.0
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Epoch Intent Acc: {epoch_acc:.4f}')

        # Validation
        val_avg_loss, val_acc = validate_model(model, val_loader, device)
        print(f'Validation Loss: {val_avg_loss:.4f}, Validation Intent Acc: {val_acc:.4f}')

    return losses, epoch_acc

def main():
    print("--- Training Intent Transformer (Phase 2 - Debug with Validation) ---")

    data_file = 'data/processed/processed_access_logs.csv'
    sequence_length = 10
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'models/saved/intent_transformer_debug.pth'
    metrics_save_path = 'models/saved/intent_transformer_debug_metrics.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load and prepare data
    X, y_intent, y_urgency, y_nav_depth, vocab_size = load_and_prepare_data(data_file, sequence_length)

    # Split into train and validation sets
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train, X_val = X[train_idx], X[val_idx]
    y_intent_train, y_intent_val = y_intent[train_idx], y_intent[val_idx]
    y_urgency_train, y_urgency_val = y_urgency[train_idx], y_urgency[val_idx]
    y_nav_depth_train, y_nav_depth_val = y_nav_depth[train_idx], y_nav_depth[val_idx]

    train_dataset = TensorDataset(X_train, y_intent_train, y_urgency_train, y_nav_depth_train)
    val_dataset = TensorDataset(X_val, y_intent_val, y_urgency_val, y_nav_depth_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = IntentTransformer(
        vocab_size=vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    print(f"Model initialized. Vocab size: {vocab_size}, Device: {device}")

    # Train model with validation
    training_losses, final_epoch_acc = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Save model and metrics
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    torch.save({
        'training_losses': training_losses,
        'final_epoch_accuracy': final_epoch_acc,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer': 'AdamW'
    }, metrics_save_path)
    print(f"Training metrics saved to {metrics_save_path}")
    print(f"Final training accuracy (intent): {final_epoch_acc:.4f}")

if __name__ == '__main__':
    main()

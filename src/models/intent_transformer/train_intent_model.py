"""
Full training script for Intent Transformer with user embeddings.
Uses smaller model (1 layer, 2 heads), low learning rate, gradient clipping,
logging, early stopping, and plotting training curves.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- Fix import paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------

from src.models.intent_transformer.intent_transformer import IntentTransformer, create_sequences

def split_users(df, train_frac=0.7, val_frac=0.15, seed=42):
    users = df['user_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(users)
    n = len(users)
    train_users = users[:int(train_frac * n)]
    val_users = users[int(train_frac * n):int((train_frac + val_frac) * n)]
    test_users = users[int((train_frac + val_frac) * n):]

    df_train = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    df_val = df[df['user_id'].isin(val_users)].reset_index(drop=True)
    df_test = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    return df_train, df_val, df_test

def create_dataloader(sequences_cat, sequences_act, sequences_user, targets_intent, targets_urgency, targets_nav, batch_size=16, shuffle=True):
    X_cat = torch.tensor(sequences_cat, dtype=torch.long)
    X_act = torch.tensor(sequences_act, dtype=torch.long)
    X_user = torch.tensor(sequences_user, dtype=torch.long)
    y_intent = torch.tensor(targets_intent, dtype=torch.float32).unsqueeze(1)
    y_urgency = torch.tensor(targets_urgency, dtype=torch.float32).unsqueeze(1)
    y_nav = torch.tensor(targets_nav, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_cat, X_act, X_user, y_intent, y_urgency, y_nav)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            X_cat, X_act, X_user, y_intent, _, _ = batch
            X_cat, X_act, X_user, y_intent = (X_cat.to(device), X_act.to(device),
                                              X_user.to(device), y_intent.to(device))
            intent_logits, _, _ = model(X_cat, X_act, X_user)
            loss = criterion(intent_logits, y_intent)
            total_loss += loss.item()
            probs = torch.sigmoid(intent_logits)
            preds = (probs > 0.5).float()
            total_correct += (preds == y_intent).sum().item()
            total_samples += y_intent.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    model.train()
    return avg_loss, accuracy

def train(model, train_loader, val_loader, device, epochs=50, lr=0.0001, pos_weight=None, patience=5):
    model.to(device)
    if pos_weight is not None:
        criterion_intent = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion_intent = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)

    best_val_acc = 0
    epochs_without_improve = 0
    best_model_state = None

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            X_cat, X_act, X_user, y_intent, _, _ = batch
            X_cat, X_act, X_user, y_intent = (X_cat.to(device), X_act.to(device),
                                             X_user.to(device), y_intent.to(device))
            optimizer.zero_grad()
            intent_logits, _, _ = model(X_cat, X_act, X_user)
            loss = criterion_intent(intent_logits, y_intent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                probs = torch.sigmoid(intent_logits)
                preds = (probs > 0.5).float()
                acc = (preds == y_intent).float().mean().item()
                print(f"Epoch [{epoch}/{epochs}] Step [{batch_idx}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion_intent)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping after {patience} epochs with no improvement.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model

def evaluate_test(model, test_df, category_to_idx, user_to_idx, seq_len, batch_size, device):
    test_seq_cat, test_seq_act, test_seq_user, test_intent, test_urgency, test_nav = create_sequences(
        test_df, category_to_idx, user_to_idx, seq_len)
    test_loader = create_dataloader(test_seq_cat, test_seq_act, test_seq_user, test_intent, test_urgency,
                                    test_nav, batch_size, shuffle=False)
    criterion_intent = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, test_loader, device, criterion_intent)
    print(f"Test Loss: {test_loss:.4f}, Test Intent Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def main():
    device = torch.device('cpu')  # Force CPU

    data_file = os.path.join(project_root, 'data/processed/temporal_splits/train/processed_access_logs.csv')
    batch_size = 16
    epochs = 50
    lr = 0.0001
    seq_len = 10
    patience = 5
    model_save_path = os.path.join(project_root, 'models/saved/intent_transformer_semantic.pth')

    df = pd.read_csv(data_file)

    train_df, val_df, test_df = split_users(df)

    unique_categories = df['Category Name'].unique()
    category_to_idx = {cat: idx + 1 for idx, cat in enumerate(unique_categories)}
    unique_users = df['user_id'].unique()
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(unique_users)}

    train_seq_cat, train_seq_act, train_seq_user, train_intent, train_urgency, train_nav = create_sequences(
        train_df, category_to_idx, user_to_idx, seq_len)
    val_seq_cat, val_seq_act, val_seq_user, val_intent, val_urgency, val_nav = create_sequences(
        val_df, category_to_idx, user_to_idx, seq_len)

    pos_ratio = sum(train_intent) / len(train_intent) if train_intent else 0
    print(f"Positive intent ratio in train set: {pos_ratio:.4f}")
    pos_weight = None
    if 0 < pos_ratio < 1:
        pos_weight = torch.tensor((1 - pos_ratio) / (pos_ratio + 1e-6)).to(device)
        print(f"Using positive class weight: {pos_weight.item():.4f}")

    train_loader = create_dataloader(train_seq_cat, train_seq_act, train_seq_user, train_intent,
                                     train_urgency, train_nav, batch_size)
    val_loader = create_dataloader(val_seq_cat, val_seq_act, val_seq_user, val_intent,
                                   val_urgency, val_nav, batch_size, shuffle=False)

    model = IntentTransformer(
        vocab_size_category=len(category_to_idx) + 1,
        vocab_size_user=len(user_to_idx) + 1,
        embedding_dim_cat=16,
        embedding_dim_user=8,
        num_heads=2,
        num_layers=1,
        dropout=0.1,
        output_dim=16
    )

    model = train(model, train_loader, val_loader, device, epochs, lr, pos_weight=pos_weight, patience=patience)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\n--- Evaluating on TEST set ---")
    evaluate_test(model, test_df, category_to_idx, user_to_idx, seq_len, batch_size, device)

if __name__ == "__main__":
    main()

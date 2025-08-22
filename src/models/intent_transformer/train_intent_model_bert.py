# src/models/train_intent_model.py
"""Script to train the Intent Transformer model (CPU-Optimized) with session-outcome intent prediction."""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pickle
import math
import time
import collections

# --- Configuration (Optimized for CPU) ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_dir = os.path.join(project_root, 'data')
processed_data_path = os.path.join(data_dir, 'processed', 'processed_access_logs.csv')
raw_orders_data_path = os.path.join(data_dir, 'raw', 'DataCoSupplyChainDataset.csv')
model_save_path = os.path.join(project_root, 'models', 'intent_transformer_bert_cpu_session_outcome.pth')
tokenizer_save_path = os.path.join(project_root, 'models', 'distilbert_tokenizer')
product_desc_map_path = os.path.join(project_root, 'models', 'product_desc_mapping.pkl')

# BERT settings (Optimized for CPU speed)
PRETRAINED_BERT_MODEL = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 8
MAX_DESCRIPTION_LENGTH = 16
EMBEDDING_DIM_ACT = 8
D_MODEL = 768 + EMBEDDING_DIM_ACT
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# Training settings
BATCH_SIZE = 8
LEARNING_RATE_BERT = 1e-5
LEARNING_RATE_OTHER = 5e-4
NUM_EPOCHS = 8
DROPOUT = 0.2
PATIENCE = 3
USE_DATA_SUBSET = True
SUBSET_FRACTION = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class IntentTransformerBERT(nn.Module):
    def __init__(self, vocab_size_act, embedding_dim_act=EMBEDDING_DIM_ACT, d_model=D_MODEL, nhead=4,
                 num_encoder_layers=1, dim_feedforward=64, max_seq_length=MAX_SEQ_LENGTH, dropout=DROPOUT, device=DEVICE):
        super(IntentTransformerBERT, self).__init__()
        self.device = device
        self.d_model = d_model
        self.embedding_dim_act = embedding_dim_act
        self.max_seq_length = max_seq_length

        self.bert = DistilBertModel.from_pretrained(PRETRAINED_BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze DistilBERT

        self.action_embedding = nn.Embedding(vocab_size_act, embedding_dim_act, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.intent_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # binary classification
        )

    def forward(self, product_descriptions, action_indices, attention_mask=None):
        batch_size, seq_len = product_descriptions.shape[0], product_descriptions.shape[1]
        flat_descriptions = product_descriptions.view(-1, MAX_DESCRIPTION_LENGTH)
        if attention_mask is not None:
            flat_attention_mask = attention_mask.view(-1, MAX_DESCRIPTION_LENGTH)
        else:
            flat_attention_mask = None

        with torch.no_grad():
            bert_outputs = self.bert(input_ids=flat_descriptions, attention_mask=flat_attention_mask)
            product_embeddings_flat = bert_outputs.last_hidden_state[:, 0, :]

        product_embeddings = product_embeddings_flat.view(batch_size, seq_len, -1)
        action_embeds = self.action_embedding(action_indices)
        combined_embeddings = torch.cat((product_embeddings, action_embeds), dim=-1)
        combined_embeddings = self.positional_encoding(combined_embeddings)
        transformer_out = self.transformer_encoder(combined_embeddings)
        intent_logits = self.intent_head(transformer_out[:, -1, :])
        return intent_logits


def load_and_prepare_data_bert(processed_data_path, raw_orders_data_path, tokenizer,
                               max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH):
    print("Loading processed access logs...")
    df_access = pd.read_csv(processed_data_path)
    print(f"Access logs loaded: {df_access.shape}")

    print("Loading raw order data for product descriptions...")
    try:
        df_orders = pd.read_csv(raw_orders_data_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying 'ISO-8859-1'")
        df_orders = pd.read_csv(raw_orders_data_path, encoding='ISO-8859-1')
    print(f"Order data loaded: {df_orders.shape}")

    print("Creating Product ID -> Description mapping...")
    product_desc_map = dict(zip(df_orders['Product Card Id'], df_orders['Product Description']))
    with open(product_desc_map_path, 'wb') as f:
        pickle.dump(product_desc_map, f)
    print(f"Product description mapping saved to {product_desc_map_path}")

    print("Preparing sequences for SESSION-OUTCOME intent prediction...")

    session_outcomes = df_access.groupby('session_id')['action'].apply(lambda actions: 'add_to_cart' in actions.values).to_dict()
    print(f"Session outcomes calculated for {len(session_outcomes)} sessions.")

    sessions = df_access.groupby('session_id')

    # Map session_id to index for splitting later
    session_id_to_index = dict()
    sequences_desc_tokens = []
    sequences_actions = []
    sequences_attention_masks = []
    targets_intent = []
    valid_session_ids = []

    action_encoder = LabelEncoder()
    all_actions = df_access['action'].unique()
    action_encoder.fit(all_actions)
    vocab_size_action = len(action_encoder.classes_)
    print(f"Action vocabulary size: {vocab_size_action}")

    num_intents = 2
    print(f"Intent task: Binary Session Outcome Prediction (0: No add_to_cart, 1: Has add_to_cart)")

    total_sessions = len(sessions)
    processed_sessions = 0
    valid_sessions_for_sequences = 0

    for session_id, group in sessions:
        processed_sessions += 1
        if processed_sessions % 50000 == 0 or processed_sessions == total_sessions:
            print(f"Processed {processed_sessions}/{total_sessions} sessions for sequences.")

        session_has_add_to_cart = session_outcomes.get(session_id, False)
        intent_label = 1 if session_has_add_to_cart else 0

        group_sorted = group.sort_values('timestamp')
        actions = group_sorted['action'].tolist()

        if 'Product Card Id' not in group_sorted.columns:
            continue

        product_ids = group_sorted['Product Card Id'].tolist()

        desc_tokens_list = []
        attention_masks_list = []
        for pid in product_ids:
            desc = str(product_desc_map.get(pid, ""))
            encoded = tokenizer(
                desc,
                add_special_tokens=True,
                max_length=max_desc_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            desc_tokens_list.append(encoded['input_ids'].squeeze(0))
            attention_masks_list.append(encoded['attention_mask'].squeeze(0))

        action_indices = action_encoder.transform(actions)

        sequence_length = len(desc_tokens_list)
        if sequence_length < 2:
            continue

        valid_sessions_for_sequences += 1
        valid_session_ids.append(session_id)

        end_idx_exclusive = min(sequence_length, max_seq_length)

        seq_desc_tokens = torch.stack(desc_tokens_list[:end_idx_exclusive])
        seq_attention_masks = torch.stack(attention_masks_list[:end_idx_exclusive])
        seq_action_indices = torch.tensor(action_indices[:end_idx_exclusive], dtype=torch.long)

        current_len = seq_desc_tokens.shape[0]
        if current_len < max_seq_length:
            pad_len = max_seq_length - current_len
            pad_desc = torch.zeros((pad_len, max_desc_length), dtype=torch.long)
            pad_attention = torch.zeros((pad_len, max_desc_length), dtype=torch.long)
            pad_actions = torch.zeros(pad_len, dtype=torch.long)

            seq_desc_tokens = torch.cat([pad_desc, seq_desc_tokens], dim=0)
            seq_attention_masks = torch.cat([pad_attention, seq_attention_masks], dim=0)
            seq_action_indices = torch.cat([pad_actions, seq_action_indices], dim=0)

        sequences_desc_tokens.append(seq_desc_tokens)
        sequences_attention_masks.append(seq_attention_masks)
        sequences_actions.append(seq_action_indices)
        targets_intent.append(intent_label)

    if not sequences_desc_tokens:
        raise ValueError("No valid sequences created. Check data alignment or sequence creation logic.")

    X_desc_tokens = torch.stack(sequences_desc_tokens)
    X_actions = torch.stack(sequences_actions)
    X_attention_masks = torch.stack(sequences_attention_masks)
    y_intent = torch.tensor(targets_intent, dtype=torch.long)

    print(f"Final tensor shapes - Descriptions: {X_desc_tokens.shape}, Actions: {X_actions.shape}, "
          f"Attention Masks: {X_attention_masks.shape}, Intent Targets: {y_intent.shape}")
    unique_targets, counts = torch.unique(y_intent, return_counts=True)
    print(f"Unique intent targets and counts: {dict(zip(unique_targets.tolist(), counts.tolist()))}")
    print(f"Total sessions processed: {processed_sessions}, Valid sessions used: {valid_sessions_for_sequences}")

    return (X_desc_tokens, X_actions, X_attention_masks, y_intent, valid_session_ids), vocab_size_action, num_intents


def train_model():
    print("Initializing DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"DistilBERT tokenizer saved to {tokenizer_save_path}")

    print("Loading and preparing data...")
    (X_desc_tokens, X_actions, X_attention_masks, y_intent, valid_session_ids), vocab_size_act, num_intents = load_and_prepare_data_bert(
        processed_data_path, raw_orders_data_path, tokenizer,
        max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH
    )

    if len(y_intent) == 0:
        print("No data available for training.")
        return

    train_session_ids, val_session_ids = train_test_split(
        valid_session_ids, test_size=0.2, random_state=42, stratify=y_intent.numpy()
    )

    session_id_to_indices = {}
    for idx, session_id in enumerate(valid_session_ids):
        session_id_to_indices.setdefault(session_id, []).append(idx)

    train_indices = [idx for sid in train_session_ids for idx in session_id_to_indices[sid]]
    val_indices = [idx for sid in val_session_ids for idx in session_id_to_indices[sid]]

    train_dataset = torch.utils.data.TensorDataset(
        X_desc_tokens[train_indices], X_actions[train_indices], y_intent[train_indices], X_attention_masks[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_desc_tokens[val_indices], X_actions[val_indices], y_intent[val_indices], X_attention_masks[val_indices]
    )

    if USE_DATA_SUBSET and len(train_dataset) > 1000:
        subset_size = max(1000, int(SUBSET_FRACTION * len(train_dataset)))
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
        print(f"Using subset of {len(train_dataset)} samples for CPU training.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Number of batches per epoch (train): {len(train_loader)}")

    model = IntentTransformerBERT(
        vocab_size_act=vocab_size_act,
        embedding_dim_act=EMBEDDING_DIM_ACT,
        d_model=D_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        device=DEVICE,
        num_encoder_layers=1,
        dim_feedforward=64
    )

    model.intent_head = nn.Sequential(
        nn.Linear(D_MODEL, 32),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(32, max(num_intents, 2)),
    ).to(DEVICE)

    try:
        all_targets = [train_dataset[i][2].item() for i in range(len(train_dataset))]
        class_weights = compute_class_weight('balanced', classes=np.unique(all_targets), y=all_targets)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"Class weights computed: {class_weights}")
    except ValueError as e:
        print(f"Could not compute class weights, using uniform weights. Error: {e}")
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Warning: No trainable parameters found. Check model freezing logic.")
        trainable_params = list(model.intent_head.parameters()) + list(model.action_embedding.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE_OTHER, weight_decay=1e-4)

    model.to(DEVICE)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        num_batches = 0
        for batch_idx, (desc_tokens_batch, action_batch, intent_target_batch, attention_mask_batch) in enumerate(train_loader):
            desc_tokens_batch = desc_tokens_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)
            intent_target_batch = intent_target_batch.to(DEVICE)
            attention_mask_batch = attention_mask_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch)
            loss = criterion(outputs, intent_target_batch)

            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {epoch+1}, batch {batch_idx}, skipping update.")
                continue

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

            # Print progress every 20 batches for timely updates during training
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_preds = 0
        val_num_batches = 0
        with torch.no_grad():
            for desc_tokens_batch, action_batch, intent_target_batch, attention_mask_batch in val_loader:
                desc_tokens_batch = desc_tokens_batch.to(DEVICE)
                action_batch = action_batch.to(DEVICE)
                intent_target_batch = intent_target_batch.to(DEVICE)
                attention_mask_batch = attention_mask_batch.to(DEVICE)

                outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch)
                loss = criterion(outputs, intent_target_batch)
                total_val_loss += loss.item()
                val_num_batches += 1

                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == intent_target_batch).sum().item()
                total_preds += intent_target_batch.size(0)

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_loss = total_val_loss / val_num_batches if val_num_batches > 0 else float('inf')
        val_accuracy = correct_preds / total_preds if total_preds > 0 else 0
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    print("Training complete.")


    # Manual inspection of some predictions on validation set
    model.eval()
    print("\nSample validation predictions:")
    with torch.no_grad():
        for i in range(min(10, len(val_dataset))):
            desc_tokens, action_indices, label, att_mask = val_dataset[i]
            desc_tokens = desc_tokens.unsqueeze(0).to(DEVICE)
            action_indices = action_indices.unsqueeze(0).to(DEVICE)
            att_mask = att_mask.unsqueeze(0).to(DEVICE)
            output = model(desc_tokens, action_indices, attention_mask=att_mask)
            pred_label = torch.argmax(output, dim=1).item()
            true_label = label.item()
            print(f"Sample {i}: True label = {true_label}, Predicted = {pred_label}")

if __name__ == "__main__":
    train_model()

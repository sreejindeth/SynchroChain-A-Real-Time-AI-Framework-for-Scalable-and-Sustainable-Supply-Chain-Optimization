# src/models/intent_transformer/train_intent_model.py
"""Script to train the Intent Transformer model using BERT embeddings for next-action prediction."""
import os
import sys
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

# --- Configuration (Optimized for CPU) ---
PROJECT_ROOT = r"D:\VIT\Project\Project" # HARDCODED project root
sys.path.insert(0, PROJECT_ROOT) # Add to path for imports

data_dir = os.path.join(PROJECT_ROOT, 'data')
processed_data_path = os.path.join(data_dir, 'processed', 'processed_access_logs.csv')
raw_orders_data_path = os.path.join(data_dir, 'raw', 'DataCoSupplyChainDataset.csv')
model_save_path = os.path.join(PROJECT_ROOT, 'models', 'intent_transformer_bert_next_action.pth')
tokenizer_save_path = os.path.join(PROJECT_ROOT, 'models', 'distilbert_tokenizer_intent')
product_desc_map_path = os.path.join(PROJECT_ROOT, 'models', 'product_desc_mapping.pkl')

# BERT settings
PRETRAINED_BERT_MODEL = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 8
MAX_DESCRIPTION_LENGTH = 32 # Truncate/pad descriptions to this length
EMBEDDING_DIM_ACT = 16
D_MODEL = 768 + EMBEDDING_DIM_ACT # BERT embedding size (768) + action embedding size
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")
print(f"Task: Predict the *next* action based on the sequence of BERT product embeddings and actions.")

# Training settings
BATCH_SIZE = 8
LEARNING_RATE_BERT = 1e-5 # Lower LR for BERT
LEARNING_RATE_OTHER = 1e-3 # Higher LR for new layers
NUM_EPOCHS = 10
DROPOUT = 0.1
PATIENCE = 3
USE_DATA_SUBSET = True # Flag to use a subset for faster CPU training
SUBSET_FRACTION = 0.1  # Use 10% of data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class IntentTransformerBERTNextAction(nn.Module):
    def __init__(self, vocab_size_act, embedding_dim_act=EMBEDDING_DIM_ACT, d_model=D_MODEL, nhead=4,
                 num_encoder_layers=2, dim_feedforward=128, max_seq_length=MAX_SEQ_LENGTH, dropout=DROPOUT, device=DEVICE):
        super(IntentTransformerBERTNextAction, self).__init__()
        self.device = device
        self.embedding_dim_act = embedding_dim_act
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Load pretrained BERT model (frozen for efficiency on CPU)
        self.bert = DistilBertModel.from_pretrained(PRETRAINED_BERT_MODEL)
        # Freeze BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False

        # Action Embedding
        self.action_embedding = nn.Embedding(vocab_size_act, embedding_dim_act, padding_idx=0)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Prediction Head for next action (binary: view vs add_to_cart)
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Single output logit for binary classification
        )

    def forward(self, product_descriptions, action_indices, attention_mask=None):
        """
        Args:
            product_descriptions: Tensor of tokenized product descriptions [batch_size, seq_len, MAX_DESCRIPTION_LENGTH]
            action_indices: Tensor of action indices [batch_size, seq_len]
            attention_mask: Tensor of attention masks [batch_size, seq_len, MAX_DESCRIPTION_LENGTH]
        Returns:
            intent_logits: Logits for next action prediction [batch_size, 1]
        """
        batch_size, seq_len = product_descriptions.shape[0], product_descriptions.shape[1]
        
        # Flatten for BERT processing
        flat_descriptions = product_descriptions.view(-1, MAX_DESCRIPTION_LENGTH) # [batch_size * seq_len, MAX_DESCRIPTION_LENGTH]
        if attention_mask is not None:
            flat_attention_mask = attention_mask.view(-1, MAX_DESCRIPTION_LENGTH) # [batch_size * seq_len, MAX_DESCRIPTION_LENGTH]
        else:
            flat_attention_mask = None

        # Get BERT embeddings for each product description in the sequence
        with torch.no_grad(): # Since BERT is frozen
            bert_outputs = self.bert(input_ids=flat_descriptions, attention_mask=flat_attention_mask)
            # Use the [CLS] token embedding (index 0) as the representation for the description
            product_embeddings_flat = bert_outputs.last_hidden_state[:, 0, :] # [batch_size * seq_len, 768]

        # Reshape back to sequence format
        product_embeddings = product_embeddings_flat.view(batch_size, seq_len, -1) # [batch_size, seq_len, 768]

        # Get action embeddings
        action_embeds = self.action_embedding(action_indices) # [batch_size, seq_len, embedding_dim_act]

        # Concatenate product and action embeddings
        combined_embeddings = torch.cat((product_embeddings, action_embeds), dim=-1) # [batch_size, seq_len, d_model]

        # Apply positional encoding
        combined_embeddings = self.positional_encoding(combined_embeddings)

        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(combined_embeddings) # [batch_size, seq_len, d_model]

        # Apply prediction head (e.g., for intent at the last item in the sequence)
        # We predict intent for the last item in the sequence
        intent_logits = self.intent_head(transformer_out[:, -1, :]) # [batch_size, 1]

        return intent_logits

def load_and_prepare_data_bert_next_action(processed_data_path, raw_orders_data_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH):
    """
    Loads data and prepares tensors for predicting the next action using BERT embeddings.
    """
    print("Loading processed access logs...")
    df_access = pd.read_csv(processed_data_path)
    print(f"Access logs loaded: {df_access.shape}")

    print("Loading raw order data for product descriptions...")
    try:
        df_orders = pd.read_csv(raw_orders_data_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed for order data, trying 'ISO-8859-1'")
        df_orders = pd.read_csv(raw_orders_data_path, encoding='ISO-8859-1')
    print(f"Order data loaded: {df_orders.shape}")

    # 1. Create Product ID -> Description mapping
    print("Creating Product ID -> Description mapping...")
    # Assuming 'Product Card Id' and 'Product Description' are correct column names
    product_desc_map = dict(zip(df_orders['Product Card Id'], df_orders['Product Description']))
    
    # Save the mapping for potential later use
    with open(product_desc_map_path, 'wb') as f:
        pickle.dump(product_desc_map, f)
    print(f"Product description mapping saved to {product_desc_map_path}")

    # 2. Prepare sequences
    print("Preparing sequences for 'Next Action Prediction'...")
    sessions = df_access.groupby('session_id')
    
    sequences_desc_tokens = []
    sequences_actions = []
    sequences_attention_masks = []
    targets_next_action = [] # Predict the *next* action (0=view, 1=add_to_cart)

    action_encoder = LabelEncoder()
    all_actions = df_access['action'].unique()
    action_encoder.fit(all_actions)
    vocab_size_action = len(action_encoder.classes_)
    print(f"Action vocabulary size: {vocab_size_action}")

    total_sessions = len(sessions)
    processed_sessions = 0

    for session_id, group in sessions:
        processed_sessions += 1
        if processed_sessions % 50000 == 0 or processed_sessions == total_sessions:
             print(f"Processed {processed_sessions}/{total_sessions} sessions for sequences.")

        group_sorted = group.sort_values('timestamp')
        product_names = group_sorted['product_name'].tolist()
        actions = group_sorted['action'].tolist()
        
        # Map product names to IDs (assuming product_name in access logs matches Product Name in orders)
        # This might need adjustment based on your exact matching logic from preprocessing
        # For now, let's assume 'Product Card Id' is now in df_access after preprocessing
        if 'Product Card Id' not in group_sorted.columns:
             print(f"Warning: 'Product Card Id' not found in session {session_id} data. Skipping session.")
             continue
             
        product_ids = group_sorted['Product Card Id'].tolist()

        # Tokenize product descriptions for the sequence
        desc_tokens_list = []
        attention_masks_list = []
        for pid in product_ids:
            desc = str(product_desc_map.get(pid, "")) # Get description or empty string
            if not desc: # If description is missing, might skip or use a placeholder
                # print(f"Warning: No description found for Product ID {pid}")
                pass # We'll tokenize the empty string or placeholder
            # Tokenize description
            encoded_desc = tokenizer(
                desc,
                add_special_tokens=True,
                max_length=max_desc_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            desc_tokens_list.append(encoded_desc['input_ids'].squeeze(0)) # [MAX_DESCRIPTION_LENGTH]
            attention_masks_list.append(encoded_desc['attention_mask'].squeeze(0)) # [MAX_DESCRIPTION_LENGTH]

        # Encode actions
        action_indices = action_encoder.transform(actions)

        # Create sequences: predict the action at time t+1 based on history up to time t
        for i in range(1, len(desc_tokens_list)): # Start from 1 to have a history
             # Input sequence: history up to time i-1
             seq_desc_tokens_input = torch.stack(desc_tokens_list[max(0, i-1-max_seq_length+1):i])
             seq_attention_masks_input = torch.stack(attention_masks_list[max(0, i-1-max_seq_length+1):i])
             seq_action_indices_input = torch.tensor(action_indices[max(0, i-1-max_seq_length+1):i], dtype=torch.long)
             
             # Pad sequences if shorter than max_seq_length
             current_len_input = seq_desc_tokens_input.shape[0]
             if current_len_input < max_seq_length:
                 pad_len_input = max_seq_length - current_len_input
                 # Pad description tokens
                 pad_desc_input = torch.zeros((pad_len_input, max_desc_length), dtype=torch.long)
                 seq_desc_tokens_input = torch.cat([pad_desc_input, seq_desc_tokens_input], dim=0)
                 # Pad attention masks
                 pad_attention_input = torch.zeros((pad_len_input, max_desc_length), dtype=torch.long)
                 seq_attention_masks_input = torch.cat([pad_attention_input, seq_attention_masks_input], dim=0)
                 # Pad action indices
                 pad_actions_input = torch.zeros(pad_len_input, dtype=torch.long)
                 seq_action_indices_input = torch.cat([pad_actions_input, seq_action_indices_input], dim=0)
             
             sequences_desc_tokens.append(seq_desc_tokens_input) # [max_seq_length, MAX_DESCRIPTION_LENGTH]
             sequences_attention_masks.append(seq_attention_masks_input) # [max_seq_length, MAX_DESCRIPTION_LENGTH]
             sequences_actions.append(seq_action_indices_input) # [max_seq_length]
             
             # Target: the action at time i
             next_action = actions[i]
             target_label = 1 if next_action == 'add_to_cart' else 0 # Default to 'view' (0)
             targets_next_action.append(target_label)

    # Convert lists to tensors
    if not sequences_desc_tokens:
        raise ValueError("No valid sequences created. Check data alignment or sequence creation logic.")
        
    X_desc_tokens = torch.stack(sequences_desc_tokens) # [num_samples, max_seq_length, MAX_DESCRIPTION_LENGTH]
    X_actions = torch.stack(sequences_actions) # [num_samples, max_seq_length]
    X_attention_masks = torch.stack(sequences_attention_masks) # [num_samples, max_seq_length, MAX_DESCRIPTION_LENGTH]
    y_next_action = torch.tensor(targets_next_action, dtype=torch.float).unsqueeze(1) # [num_samples, 1]

    print(f"Final tensor shapes - Descriptions: {X_desc_tokens.shape}, Actions: {X_actions.shape}, Attention Masks: {X_attention_masks.shape}, Next Action Targets: {y_next_action.shape}")
    
    unique_targets, counts = torch.unique(y_next_action, return_counts=True)
    print(f"Unique next action targets and counts: {dict(zip(unique_targets.tolist(), counts.tolist()))}")
    
    if len(counts) > 0:
        majority_baseline_acc = max(counts).item() / len(y_next_action)
        print(f"Majority class baseline accuracy: {majority_baseline_acc:.4f}")
    else:
        print("Warning: No targets found.")

    dataset = TensorDataset(X_desc_tokens, X_actions, y_next_action, X_attention_masks)
    return dataset, vocab_size_action

def train_model():
    """Main training function."""
    print("Initializing BERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"BERT tokenizer saved to {tokenizer_save_path}")

    print("Loading and preparing data...")
    dataset, vocab_size_act = load_and_prepare_data_bert_next_action(
        processed_data_path, raw_orders_data_path, tokenizer,
        max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH
    )
    
    if len(dataset) == 0:
        print("No data available for training.")
        return

    # --- Use a subset of data for CPU training ---
    if USE_DATA_SUBSET and len(dataset) > 1000: # Only subset if large enough
        subset_size = max(1000, int(SUBSET_FRACTION * len(dataset))) # Ensure at least 1000 samples
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices)
        print(f"Using a subset of {len(dataset)} samples for CPU training.")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Number of batches per epoch (train): {len(train_loader)}")

    # Initialize model
    model = IntentTransformerBERTNextAction(
        vocab_size_act=vocab_size_act,
        embedding_dim_act=EMBEDDING_DIM_ACT,
        d_model=D_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        device=DEVICE,
    )
    model.to(DEVICE)
    print(model)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() # More numerically stable than sigmoid + BCELoss

    # Differential learning rate: AdamW for better weight decay handling
    # Filter parameters that require gradients (BERT is frozen, so only others are trained)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Warning: No trainable parameters found. Check model freezing logic.")
        # Fallback: train the head layers regardless
        trainable_params = list(model.intent_head.parameters()) + list(model.action_embedding.parameters()) + list(model.positional_encoding.parameters()) + list(model.transformer_encoder.parameters())
        
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE_OTHER, weight_decay=1e-4)

    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        correct_preds = 0
        total_preds = 0
        num_batches = 0
        for batch_idx, (desc_tokens_batch, action_batch, next_action_target_batch, attention_mask_batch) in enumerate(train_loader):
            desc_tokens_batch = desc_tokens_batch.to(DEVICE) # [B, S, D]
            action_batch = action_batch.to(DEVICE) # [B, S]
            next_action_target_batch = next_action_target_batch.to(DEVICE) # [B, 1]
            attention_mask_batch = attention_mask_batch.to(DEVICE) # [B, S, D]

            optimizer.zero_grad()

            outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch) # [B, 1]
            loss = criterion(outputs, next_action_target_batch)
            
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss encountered at epoch {epoch+1}, batch {batch_idx}, skipping update.")
                continue
                
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

            # Calculate accuracy for this batch
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float() # Apply sigmoid and threshold
                correct_preds += (preds == next_action_target_batch).sum().item()
                total_preds += next_action_target_batch.size(0) # Number of samples in batch

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val_preds = 0
        total_val_preds = 0
        val_num_batches = 0
        with torch.no_grad():
            for desc_tokens_batch, action_batch, next_action_target_batch, attention_mask_batch in val_loader:
                desc_tokens_batch = desc_tokens_batch.to(DEVICE)
                action_batch = action_batch.to(DEVICE)
                next_action_target_batch = next_action_target_batch.to(DEVICE)
                attention_mask_batch = attention_mask_batch.to(DEVICE)

                outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch)
                loss = criterion(outputs, next_action_target_batch)
                total_val_loss += loss.item()
                val_num_batches += 1

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val_preds += (preds == next_action_target_batch).sum().item()
                total_val_preds += next_action_target_batch.size(0)

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_loss = total_val_loss / val_num_batches if val_num_batches > 0 else float('inf')
        train_accuracy = correct_preds / total_preds if total_preds > 0 else 0
        val_accuracy = correct_val_preds / total_val_preds if total_val_preds > 0 else 0

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Time: {epoch_time:.2f}s")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")

if __name__ == "__main__":
    train_model()

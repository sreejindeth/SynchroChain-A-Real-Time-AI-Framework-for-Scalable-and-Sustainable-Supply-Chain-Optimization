# src/models/intent_transformer.py
"""
Transformer Model for Intent Detection from Access Logs.
This model takes sequences of user-product interactions and outputs intent/urgency scores.
"""
import torch
import torch.nn as nn
import math
import pandas as pd

class IntentTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(IntentTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Simplified positional encoding for sequences
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True # Important for our data format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Head for intent prediction (e.g., a score between 0 and 1)
        self.intent_head = nn.Linear(embedding_dim, 1)
        # Head for urgency prediction (e.g., a score between 0 and 1)
        self.urgency_head = nn.Linear(embedding_dim, 1)
        # Head for navigation depth prediction (auxiliary task)
        self.nav_depth_head = nn.Linear(embedding_dim, 1)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.intent_head.bias.data.zero_()
        self.intent_head.weight.data.uniform_(-initrange, initrange)
        self.urgency_head.bias.data.zero_()
        self.urgency_head.weight.data.uniform_(-initrange, initrange)
        self.nav_depth_head.bias.data.zero_()
        self.nav_depth_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len) containing token indices.
            src_key_padding_mask: Tensor of shape (batch_size, seq_len) indicating padding.
        Returns:
            intent_logits: (batch_size, 1)
            urgency_logits: (batch_size, 1)
            nav_depth_logits: (batch_size, 1)
        """
        # src shape: (batch_size, seq_len)
        embedded = self.embedding(src) * math.sqrt(self.embedding_dim)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.pos_encoder(embedded)
        
        # Pass through Transformer Encoder
        # Note: TransformerEncoder expects (batch_size, seq_len, embedding_dim)
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        # output shape: (batch_size, seq_len, embedding_dim)
        
        # Use the representation of the last item in the sequence for prediction
        # This captures the cumulative intent up to the last action
        last_output = output[:, -1, :] # (batch_size, embedding_dim)
        
        intent_logits = torch.sigmoid(self.intent_head(last_output)) # (batch_size, 1)
        urgency_logits = torch.sigmoid(self.urgency_head(last_output)) # (batch_size, 1)
        nav_depth_logits = torch.relu(self.nav_depth_head(last_output)) # (batch_size, 1)
        
        return intent_logits, urgency_logits, nav_depth_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :] # Add positional encoding
        return self.dropout(x)

# --- Utility Functions for Data Preparation ---

# src/models/intent_transformer.py

# ... (keep the IntentTransformer and PositionalEncoding classes as they are) ...

# --- Updated Utility Function for Data Preparation ---

# ... (IntentTransformer class definition remains the same) ...

# --- Updated Utility Function for Data Preparation ---
def create_sequences(df_access, product_to_idx, user_to_idx, sequence_length=10):
    """
    Creates sequences of product indices for training the Transformer.
    Simplified target creation for debugging.
    """
    sequences = []
    # --- Simplified Targets ---
    targets_intent = [] # Binary: 1 if last action is 'add_to_cart', else 0
    targets_urgency = [] # Predict last navigation_depth (continuous)
    targets_nav_depth = [] # Predict last navigation_depth (continuous)
    # --------------------------

    # Ensure Product Card Id is numeric
    df_access['Product Card Id'] = pd.to_numeric(df_access['Product Card Id'], errors='coerce')
    df_access.dropna(subset=['Product Card Id'], inplace=True)
    df_access['Product Card Id'] = df_access['Product Card Id'].astype(int)

    for user_id, group in df_access.groupby('user_id'):
        # Get list of raw Product Card Ids for this user group
        raw_product_ids = group['Product Card Id'].tolist() 
        actions = group['action'].tolist()
        nav_depths = group['navigation_depth'].tolist()
        
        # --- Convert raw Product Card Ids to indices ---
        indexed_products = [product_to_idx.get(pid, 0) for pid in raw_product_ids]
        # ----------------------------------------------------------

        # Create sequences of indices
        if len(indexed_products) >= sequence_length:
             for i in range(len(indexed_products) - sequence_length + 1):
                 seq_indices = indexed_products[i:i+sequence_length] # Now contains indices 0-72
                 
                 # --- Simplified Target Creation ---
                 # 1. Intent: Binary label based on the last action in the sequence window
                 last_action_in_seq = actions[i+sequence_length-1] if (i+sequence_length-1) < len(actions) else 'view'
                 simple_intent_label = 1.0 if last_action_in_seq == 'add_to_cart' else 0.0
                 
                 # 2. Urgency/Nav Depth: Predict the actual last navigation depth in the window
                 last_nav_depth_in_seq = nav_depths[i+sequence_length-1] if (i+sequence_length-1) < len(nav_depths) else 0.0
                 # For this simplified version, we'll use nav_depth as a proxy for both targets
                 # You can experiment with different proxies for urgency if desired.
                 simple_urgency_label = last_nav_depth_in_seq # Or a fixed value like 0.5 for testing
                 nav_depth_target = last_nav_depth_in_seq
                 # -----------------------------------

                 sequences.append(seq_indices)
                 targets_intent.append(simple_intent_label)
                 targets_urgency.append(simple_urgency_label)
                 targets_nav_depth.append(nav_depth_target)
                 
        # --- Optional: Handle shorter sequences by padding ---
        # This part is omitted for brevity but can be added if needed.
        # -----------------------------------------------------
    return sequences, targets_intent, targets_urgency, targets_nav_depth
# ... (rest of the file remains the same) ...

# ... (rest of the file remains the same) ...
# src/models/intent_transformer/intent_transformer.py
import torch
import torch.nn as nn
import math
from transformers import DistilBertModel

# --- 1. DEFINE PositionalEncoding HERE ---
# This class MUST be in this file for the import in train_intent_model.py to work.
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
        self.register_buffer('pe', pe) # Saves pe in model state, not as a parameter

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :] # Add positional encoding
        return self.dropout(x) # Apply dropout
# --- END 1. DEFINE PositionalEncoding ---

# --- 2. DEFINE IntentTransformer ---
class IntentTransformer(nn.Module):
    def __init__(self, vocab_size_act, embedding_dim_act=16, d_model=784, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(IntentTransformer, self).__init__()
        self.embedding_dim_act = embedding_dim_act
        self.d_model = d_model

        # Load DistilBERT model (now trainable)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Action Embedding
        self.action_embedding = nn.Embedding(vocab_size_act, embedding_dim_act, padding_idx=0)

        # Positional Encoding (defined above in the same file)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Multi-Output Heads
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.nav_depth_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, product_descriptions, action_indices, attention_mask=None):
        batch_size, seq_len = product_descriptions.shape[0], product_descriptions.shape[1]
        
        # Flatten for BERT processing
        flat_descriptions = product_descriptions.view(-1, 16) # Assuming MAX_DESCRIPTION_LENGTH is 16
        if attention_mask is not None:
            flat_attention_mask = attention_mask.view(-1, 16)
        else:
            flat_attention_mask = None

        # Get BERT embeddings (now trainable)
        bert_outputs = self.bert(input_ids=flat_descriptions, attention_mask=flat_attention_mask)
        product_embeddings_flat = bert_outputs.last_hidden_state[:, 0, :] # [CLS] token

        # Reshape back to sequence
        product_embeddings = product_embeddings_flat.view(batch_size, seq_len, -1)
        
        # Get action embeddings
        action_embeds = self.action_embedding(action_indices)
        
        # Combine embeddings
        combined_embeddings = torch.cat((product_embeddings, action_embeds), dim=-1)
        
        # Apply positional encoding
        combined_embeddings = self.positional_encoding(combined_embeddings)
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(combined_embeddings)
        
        # Use the output from the last time step for prediction
        last_output = transformer_out[:, -1, :]
        
        # Predict all outputs
        intent_logits = self.intent_head(last_output)
        urgency_logits = self.urgency_head(last_output)
        nav_depth_logits = self.nav_depth_head(last_output)
        
        return intent_logits, urgency_logits, nav_depth_logits
# --- END 2. DEFINE IntentTransformer ---
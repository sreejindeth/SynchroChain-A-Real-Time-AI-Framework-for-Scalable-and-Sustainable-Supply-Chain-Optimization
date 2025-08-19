import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Minimal PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Minimal IntentTransformer
class IntentTransformer(nn.Module):
    def __init__(self, vocab_cat, vocab_user, emb_dim_cat=16, emb_dim_user=8,
                 num_heads=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.cat_emb = nn.Embedding(vocab_cat, emb_dim_cat, padding_idx=0)
        self.act_emb = nn.Embedding(2, emb_dim_cat, padding_idx=0)
        self.user_emb = nn.Embedding(vocab_user, emb_dim_user, padding_idx=0)
        self.combined_dim = emb_dim_cat * 2
        self.pos_enc = PositionalEncoding(self.combined_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.combined_dim,
                                                   nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        head_dim = self.combined_dim + emb_dim_user
        self.intent_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 1)
        )
    def forward(self, cat_seq, act_seq, user_idx):
        cat_e = self.cat_emb(cat_seq) * math.sqrt(self.cat_emb.embedding_dim)
        act_e = self.act_emb(act_seq) * math.sqrt(self.act_emb.embedding_dim)
        combined = torch.cat([cat_e, act_e], dim=-1)
        combined = self.pos_enc(combined)
        transformed = self.transformer(combined)
        last = transformed[:, -1, :]
        user_e = self.user_emb(user_idx)
        final = torch.cat([last, user_e], dim=-1)
        intent_logit = self.intent_head(final)
        return intent_logit.squeeze(-1)

def create_sequences(df, cat_to_idx, user_to_idx, seq_len=10):
    action_map = {'view':0,'add_to_cart':1}
    seq_cat, seq_act, seq_user, target_intent = [], [], [], []
    for u, g in df.groupby('user_id'):
        cats = [cat_to_idx.get(c,0) for c in g['Category Name']]
        acts = [action_map.get(a,0) for a in g['action']]
        user_idx = user_to_idx.get(u, 0)
        if len(cats) >= seq_len +1:
            for i in range(len(cats) - seq_len):
                seq_cat.append(cats[i:i+seq_len])
                seq_act.append(acts[i:i+seq_len])
                seq_user.append(user_idx)
                nxt_act = g['action'].iloc[i+seq_len]
                target_intent.append(1.0 if nxt_act == 'add_to_cart' else 0.0)
    print(f"Created {len(seq_cat)} sequences.")
    return seq_cat, seq_act, seq_user, target_intent

def debug_overfit_small(model, seq_cat, seq_act, seq_user, targets, device, batch_size=32, epochs=30):
    print("Sample sequences and labels (first 5):")
    for i in range(min(5,len(seq_cat))):
        print(f"Seq cat: {seq_cat[i]}, Seq act: {seq_act[i]}, User idx: {seq_user[i]}, Intent: {targets[i]}")
    X_cat = torch.tensor(seq_cat[:batch_size], dtype=torch.long).to(device)
    X_act = torch.tensor(seq_act[:batch_size], dtype=torch.long).to(device)
    X_user = torch.tensor(seq_user[:batch_size], dtype=torch.long).to(device)
    y = torch.tensor(targets[:batch_size], dtype=torch.float32).to(device)

    dataset = TensorDataset(X_cat, X_act, X_user, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss=0
        for Xc,Xa,Xu,yb in loader:
            optimizer.zero_grad()
            logits = model(Xc,Xa,Xu)
            loss = criterion(logits,yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    print("Finished overfitting small batch test.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file = 'data/processed/temporal_splits/train/processed_access_logs.csv'  # Update path if needed

    df = pd.read_csv(data_file)
    cat_to_idx = {cat: i+1 for i, cat in enumerate(df['Category Name'].unique())}
    user_to_idx = {uid: i+1 for i, uid in enumerate(df['user_id'].unique())}

    seq_cat, seq_act, seq_user, targets = create_sequences(df, cat_to_idx, user_to_idx, seq_len=10)

    model = IntentTransformer(
        vocab_cat=len(cat_to_idx)+1,
        vocab_user=len(user_to_idx)+1,
        emb_dim_cat=16,
        emb_dim_user=8,
        num_heads=2,
        num_layers=1,
        dropout=0.1
    ).to(device)

    debug_overfit_small(model, seq_cat, seq_act, seq_user, targets, device)

if __name__ == "__main__":
    main()

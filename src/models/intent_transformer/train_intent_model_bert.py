import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import pickle
import math
import time

# Constants adjusted for CPU and balanced training time vs accuracy
PRETRAINED_BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 15
MAX_DESCRIPTION_LENGTH = 64
EMBEDDING_DIM_ACT = 16
D_MODEL = 768 + EMBEDDING_DIM_ACT
DEVICE = torch.device('cpu')
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
DROPOUT = 0.1
PATIENCE = 3

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_dir = os.path.join(project_root, 'data')
processed_data_path = os.path.join(data_dir, 'processed', 'processed_access_logs.csv')
raw_orders_data_path = os.path.join(data_dir, 'raw', 'DataCoSupplyChainDataset.csv')
model_save_path = os.path.join(project_root, 'models', 'intent_transformer_bert.pth')
tokenizer_save_path = os.path.join(project_root, 'models', 'bert_tokenizer')
product_desc_map_path = os.path.join(project_root, 'models', 'product_desc_mapping.pkl')

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
    def __init__(self, vocab_size_act, embedding_dim_act=EMBEDDING_DIM_ACT, d_model=D_MODEL, nhead=8,
                 num_encoder_layers=2, dim_feedforward=512, max_seq_length=MAX_SEQ_LENGTH, dropout=DROPOUT, device=DEVICE):
        super(IntentTransformerBERT, self).__init__()
        self.device = device
        self.d_model = d_model
        self.embedding_dim_act = embedding_dim_act
        self.max_seq_length = max_seq_length
        self.bert = BertModel.from_pretrained(PRETRAINED_BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False
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
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
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

def load_and_prepare_data_bert(processed_data_path, raw_orders_data_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH):
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

    print("Creating Product ID -> Description mapping...")
    product_desc_map = dict(zip(df_orders['Product Card Id'], df_orders['Product Description']))
    with open(product_desc_map_path, 'wb') as f:
        pickle.dump(product_desc_map, f)
    print(f"Product description mapping saved to {product_desc_map_path}")

    print("Preparing sequences...")
    sessions = list(df_access.groupby('session_id'))
    total_sessions = len(sessions)

    sequences_desc_tokens = []
    sequences_actions = []
    sequences_attention_masks = []
    targets_intent = []

    action_encoder = LabelEncoder()
    all_actions = df_access['action'].unique()
    action_encoder.fit(all_actions)
    vocab_size_action = len(action_encoder.classes_)
    print(f"Action vocabulary size: {vocab_size_action}")

    intent_mapping = {'view': 0, 'add_to_cart': 1, 'purchase': 2}
    if 'purchase' not in intent_mapping:
        intent_mapping = {'view': 0, 'add_to_cart': 1}
    num_intents = len(intent_mapping)
    print(f"Intent classes: {intent_mapping}")

    for idx, (session_id, group) in enumerate(sessions):
        if idx % 10000 == 0:
            print(f"Processing session {idx}/{total_sessions}")
        group_sorted = group.sort_values('timestamp')
        actions = group_sorted['action'].tolist()

        if 'Product Card Id' not in group_sorted.columns:
            print(f"Warning: 'Product Card Id' not found in session {session_id} data. Skipping session.")
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

        for i in range(1, len(desc_tokens_list)):
            seq_desc_tokens = torch.stack(desc_tokens_list[max(0, i - max_seq_length + 1):i + 1])
            seq_attention_masks = torch.stack(attention_masks_list[max(0, i - max_seq_length + 1):i + 1])
            seq_action_indices = torch.tensor(action_indices[max(0, i - max_seq_length + 1):i + 1], dtype=torch.long)

            current_len = seq_desc_tokens.shape[0]
            if current_len < max_seq_length:
                pad_len = max_seq_length - current_len
                pad_desc = torch.zeros((pad_len, max_desc_length), dtype=torch.long)
                pad_attention = torch.zeros(pad_len, max_desc_length, dtype=torch.long)
                pad_actions = torch.zeros(pad_len, dtype=torch.long)

                seq_desc_tokens = torch.cat([pad_desc, seq_desc_tokens], dim=0)
                seq_attention_masks = torch.cat([pad_attention, seq_attention_masks], dim=0)
                seq_action_indices = torch.cat([pad_actions, seq_action_indices], dim=0)

            sequences_desc_tokens.append(seq_desc_tokens)
            sequences_attention_masks.append(seq_attention_masks)
            sequences_actions.append(seq_action_indices)
            targets_intent.append(intent_mapping.get(actions[i], 0))

    if not sequences_desc_tokens:
        raise ValueError("No valid sequences created. Check data alignment or sequence creation logic.")

    X_desc_tokens = torch.stack(sequences_desc_tokens)
    X_actions = torch.stack(sequences_actions)
    X_attention_masks = torch.stack(sequences_attention_masks)
    y_intent = torch.tensor(targets_intent, dtype=torch.long)

    print(f"Final tensor shapes - Descriptions: {X_desc_tokens.shape}, Actions: {X_actions.shape}, Intent Targets: {y_intent.shape}")
    print(f"Unique intent targets: {torch.unique(y_intent)}")

    return TensorDataset(X_desc_tokens, X_actions, y_intent, X_attention_masks), vocab_size_action, num_intents

def train_model():
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"BERT tokenizer saved to {tokenizer_save_path}")
    print("Loading and preparing data...")
    dataset, vocab_size_act, num_intents = load_and_prepare_data_bert(
        processed_data_path, raw_orders_data_path, tokenizer,
        max_seq_length=MAX_SEQ_LENGTH, max_desc_length=MAX_DESCRIPTION_LENGTH
    )
    if len(dataset) == 0:
        print("No data available for training.")
        return

    # Use only a subset to reduce training time (e.g., 20% data)
    subset_size = int(0.2 * len(dataset))
    dataset_subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size
    train_dataset, val_dataset = random_split(dataset_subset, [train_size, val_size])

    global_model = IntentTransformerBERT(
        vocab_size_act=vocab_size_act,
        embedding_dim_act=EMBEDDING_DIM_ACT,
        d_model=D_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        device=DEVICE,
    )
    global_model.intent_head = nn.Sequential(
        nn.Linear(D_MODEL, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_intents),
    )
    model = global_model.to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Number of batches per epoch (train): {len(train_loader)}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            desc_tokens_batch = batch[0].to(DEVICE)
            action_batch = batch[1].to(DEVICE)
            intent_target_batch = batch[2].to(DEVICE)
            attention_mask_batch = batch[3].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch)
            loss = criterion(outputs, intent_target_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} Batch {batch_idx}/{len(train_loader)} - Avg Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s")

        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in val_loader:
                desc_tokens_batch = batch[0].to(DEVICE)
                action_batch = batch[1].to(DEVICE)
                intent_target_batch = batch[2].to(DEVICE)
                attention_mask_batch = batch[3].to(DEVICE)
                outputs = model(desc_tokens_batch, action_batch, attention_mask=attention_mask_batch)
                loss = criterion(outputs, intent_target_batch)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == intent_target_batch).sum().item()
                total_preds += intent_target_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s"
        )

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

if __name__ == "__main__":
    train_model()

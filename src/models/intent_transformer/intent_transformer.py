"""
Transformer Model for Intent Detection from Access Logs.
Includes User Embeddings and next-action intent labeling.
"""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class IntentTransformer(nn.Module):
    def __init__(self, vocab_size_category, vocab_size_user, embedding_dim_cat=32, embedding_dim_user=16,
                 num_heads=4, num_layers=2, dropout=0.1, output_dim=16):
        super(IntentTransformer, self).__init__()
        self.embedding_dim_cat = embedding_dim_cat
        self.embedding_dim_user = embedding_dim_user
        self.output_dim = output_dim
        
        self.category_embedding = nn.Embedding(vocab_size_category, embedding_dim_cat, padding_idx=0)
        self.action_embedding = nn.Embedding(2, embedding_dim_cat, padding_idx=0)  # actions: view=0, add_to_cart=1
        self.user_embedding = nn.Embedding(vocab_size_user, embedding_dim_user, padding_idx=0)
        
        self.combined_feature_dim = embedding_dim_cat + embedding_dim_cat  # cat + action
        
        self.pos_encoder = PositionalEncoding(self.combined_feature_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.combined_feature_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        head_input_dim = self.combined_feature_dim + embedding_dim_user
        
        self.intent_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, 1),
            nn.Sigmoid()
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, 1),
            nn.Sigmoid()
        )
        self.nav_depth_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, 1),
            nn.ReLU()
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.category_embedding.weight.data.uniform_(-initrange, initrange)
        self.action_embedding.weight.data.uniform_(-initrange, initrange)
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        
        for head in [self.intent_head, self.urgency_head, self.nav_depth_head]:
            for module in head:
                if isinstance(module, nn.Linear):
                    module.weight.data.uniform_(-initrange, initrange)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def forward(self, src_cat, src_action, src_user, src_key_padding_mask=None):
        embedded_cat = self.category_embedding(src_cat) * math.sqrt(self.embedding_dim_cat)
        embedded_action = self.action_embedding(src_action) * math.sqrt(self.embedding_dim_cat)
        embedded_combined = torch.cat([embedded_cat, embedded_action], dim=-1)  # (B, seq_len, dim)

        embedded_pos = self.pos_encoder(embedded_combined)

        transformer_out = self.transformer_encoder(embedded_pos, src_key_padding_mask=src_key_padding_mask)

        last_output = transformer_out[:, -1, :]  # (B, combined_feature_dim)

        user_emb = self.user_embedding(src_user)  # (B, embedding_dim_user)

        final_output = torch.cat([last_output, user_emb], dim=-1)  # (B, combined_feature_dim + embedding_dim_user)

        intent_logits = self.intent_head(final_output)
        urgency_logits = self.urgency_head(final_output)
        nav_depth_logits = self.nav_depth_head(final_output)

        return intent_logits, urgency_logits, nav_depth_logits

def create_sequences(df_access, category_to_idx, user_to_idx, sequence_length=10):
    """
    Creates sequences and next-action intent labels.

    Args:
        df_access: DataFrame with columns 'user_id', 'Category Name', 'action', 'navigation_depth'
        category_to_idx: dict mapping category names to indices
        user_to_idx: dict mapping user ids to indices
        sequence_length: length of input sequences

    Returns:
        sequences_cat, sequences_action, sequences_user, targets_intent, targets_urgency, targets_nav_depth
    """
    print("Creating sequences for Intent Transformer...")
    sequences_cat = []
    sequences_action = []
    sequences_user = []
    targets_intent = []
    targets_urgency = []
    targets_nav_depth = []

    action_to_idx = {'view': 0, 'add_to_cart': 1}

    for user_id, group in df_access.groupby('user_id'):
        categories = group['Category Name'].tolist()
        actions = group['action'].tolist()
        nav_depths = group['navigation_depth'].tolist()

        indexed_categories = [category_to_idx.get(cat, 0) for cat in categories]
        indexed_actions = [action_to_idx.get(act, 0) for act in actions]
        user_idx = user_to_idx.get(user_id, 0)

        if len(indexed_categories) >= sequence_length + 1:  # need next action for label
            for i in range(len(indexed_categories) - sequence_length):
                seq_cat = indexed_categories[i:i + sequence_length]
                seq_action = indexed_actions[i:i + sequence_length]
                seq_user = user_idx

                # Next-action intent label:
                next_action = actions[i + sequence_length]
                intent_target = 1.0 if next_action == 'add_to_cart' else 0.0

                last_nav_depth = nav_depths[i + sequence_length - 1]
                urgency_target = min(1.0, last_nav_depth / 20.0)
                nav_depth_target = float(last_nav_depth)

                sequences_cat.append(seq_cat)
                sequences_action.append(seq_action)
                sequences_user.append(seq_user)
                targets_intent.append(intent_target)
                targets_urgency.append(urgency_target)
                targets_nav_depth.append(nav_depth_target)

    print(f"  Created {len(sequences_cat)} sequences.")
    return sequences_cat, sequences_action, sequences_user, targets_intent, targets_urgency, targets_nav_depth

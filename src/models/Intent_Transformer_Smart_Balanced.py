"""
Smart Balanced Intent Transformer
- Maintains natural class distribution (not perfect 33/33/33)
- Uses stratified focal loss with moderate gamma
- Ensures Medium Intent is learnable without collapsing to single-class prediction
- Target: 70%+ accuracy with ALL classes having F1 > 0.45
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

torch.manual_seed(42)
np.random.seed(42)

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss - moderate gamma to avoid collapse"""
    def __init__(self, alpha=None, gamma=1.5):  # Lower gamma than before
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class SmartIntentTransformer(nn.Module):
    """Transformer with session-level features to distinguish Low Intent"""
    def __init__(self, vocab_size, d_model=112, nhead=4, num_layers=3, dropout=0.35, num_features=3):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feature processing (for session-level features like duration, unique_pages, etc.)
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )
        
        # Classification head: sequence features + session features
        # Input size: (d_model * 2) from sequence pooling + (d_model // 4) from features
        self.intent_head = nn.Sequential(
            nn.Linear(d_model * 2 + d_model // 4, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 3)
        )
        
        # REMOVED: Urgency head causes target leak
        # Urgency is deterministic function of intent - removing multi-task learning
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, features=None):
        # Sequence processing
        x = self.embedding(x) * np.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        
        # Dual pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        sequence_pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Process session-level features if provided
        if features is not None:
            feature_vec = self.feature_proj(features)
            combined = torch.cat([sequence_pooled, feature_vec], dim=1)
        else:
            # Fallback: zero features if not provided
            batch_size = sequence_pooled.size(0)
            feature_vec = torch.zeros(batch_size, self.d_model // 4, device=sequence_pooled.device)
            combined = torch.cat([sequence_pooled, feature_vec], dim=1)
        
        combined = self.dropout(combined)
        intent_logits = self.intent_head(combined)
        
        return intent_logits

class SmartDataset(Dataset):
    def __init__(self, sessions, intents, session_features=None):
        # session_features: [session_duration, unique_pages, action_density]
        self.sessions = sessions
        self.intents = intents
        self.session_features = session_features if session_features is not None else []
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        if len(self.session_features) > 0:
            feat = self.session_features[idx]
            # Ensure features is always 1D array (handle 0-d arrays/scalars)
            if isinstance(feat, np.ndarray):
                if feat.ndim == 0:
                    feat = np.array([feat.item()])
                feat = feat.flatten()
            elif not isinstance(feat, (list, np.ndarray)):
                feat = [float(feat)]
            features = torch.FloatTensor(np.array(feat).flatten())
            # Ensure it's exactly 3 features
            if len(features) < 3:
                features = torch.cat([features, torch.zeros(3 - len(features))])
            elif len(features) > 3:
                features = features[:3]
        else:
            features = torch.zeros(3)  # Default: 3 features
        
        return (
            torch.LongTensor(self.sessions[idx]),
            torch.LongTensor([self.intents[idx]]),
            features
        )

def prepare_smart_balanced_data():
    """Smart balancing: Keep natural distribution but boost Medium Intent"""
    print("📊 Loading and preparing SMART BALANCED dataset...")
    
    df = pd.read_csv('data/processed/processed_access_logs.csv')
    print(f"   Loaded {len(df)} records")
    
    # Sample sessions
    unique_sessions = df['session_id'].unique()
    max_sessions = 50000
    if len(unique_sessions) > max_sessions:
        np.random.seed(42)
        sampled_sessions = np.random.choice(unique_sessions, max_sessions, replace=False)
        df = df[df['session_id'].isin(sampled_sessions)]
    
    # Create intent labels with BETTER thresholds
    session_stats = df.groupby('session_id').agg({
        'action': ['count', lambda x: (x == 'add_to_cart').sum()]
    }).round(3)
    session_stats.columns = ['total_actions', 'add_to_cart_count']
    session_stats['add_to_cart_ratio'] = session_stats['add_to_cart_count'] / session_stats['total_actions']
    
    # Optimized thresholds to get better Medium Intent representation
    def get_intent_label(ratio):
        if ratio >= 0.35:  # High: strong cart activity
            return 'high'
        elif ratio >= 0.015:  # Medium: some cart activity
            return 'medium'
        else:
            return 'low'  # Low: browsing only
    
    session_stats['intent'] = session_stats['add_to_cart_ratio'].apply(get_intent_label)
    df_with_intent = df.merge(session_stats[['intent', 'add_to_cart_ratio']], 
                             left_on='session_id', right_index=True, how='left')
    
    def get_urgency(intent, ratio):
        if intent == 'high':
            return min(0.9, 0.4 + ratio)
        elif intent == 'medium':
            return 0.2 + ratio * 0.5
        else:
            return ratio * 0.4
    
    df_with_intent['urgency'] = df_with_intent.apply(
        lambda x: get_urgency(x['intent'], x['add_to_cart_ratio']), axis=1
    )
    
    session_intents = df_with_intent.groupby('session_id')['intent'].first()
    intent_dist = session_intents.value_counts()
    print(f"   Raw distribution: {intent_dist.to_dict()}")
    
    # SMART BALANCING: Not perfect 33/33/33, but ensure Medium is well-represented
    high_sessions = session_intents[session_intents == 'high'].index
    medium_sessions = session_intents[session_intents == 'medium'].index
    low_sessions = session_intents[session_intents == 'low'].index
    
    # Target distribution: 40% High, 30% Medium, 30% Low (more natural)
    total_target = min(20000, len(session_intents))
    target_high = int(total_target * 0.40)
    target_medium = int(total_target * 0.30)
    target_low = int(total_target * 0.30)
    
    print(f"   Target: High={target_high}, Medium={target_medium}, Low={target_low}")
    
    balanced_sessions = []
    
    # Sample High
    if len(high_sessions) >= target_high:
        balanced_sessions.extend(np.random.choice(high_sessions, target_high, replace=False))
    else:
        balanced_sessions.extend(high_sessions)
        print(f"   ⚠️  Only {len(high_sessions)} High samples available")
    
    # Sample Medium (with oversampling if needed)
    if len(medium_sessions) >= target_medium:
        balanced_sessions.extend(np.random.choice(medium_sessions, target_medium, replace=False))
    else:
        # Moderate oversampling for Medium
        oversample_factor = min(2.0, target_medium / len(medium_sessions))
        actual_medium = int(len(medium_sessions) * oversample_factor)
        balanced_sessions.extend(np.random.choice(medium_sessions, actual_medium, replace=True))
        print(f"   ℹ️  Oversampled Medium: {len(medium_sessions)} → {actual_medium}")
    
    # Sample Low
    if len(low_sessions) >= target_low:
        balanced_sessions.extend(np.random.choice(low_sessions, target_low, replace=False))
    else:
        balanced_sessions.extend(low_sessions)
    
    df_balanced = df_with_intent[df_with_intent['session_id'].isin(balanced_sessions)]
    
    balanced_dist = df_balanced.groupby('session_id')['intent'].first().value_counts()
    print(f"   Balanced distribution: {balanced_dist.to_dict()}")
    
    # Create sequences
    actions = df_balanced['action'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    print(f"   Action vocabulary: {len(actions)} actions")
    
    sessions = []
    intents_list = []
    urgencies_list = []
    
    max_seq_length = 50
    intent_map = {'high': 0, 'medium': 1, 'low': 2}
    intent_encoder = LabelEncoder()
    intent_encoder.classes_ = np.array(['High Intent', 'Medium Intent', 'Low Intent'])
    
    unique_sessions = df_balanced['session_id'].unique()
    print(f"   Processing {len(unique_sessions)} sessions...")
    
    # CRITICAL FIX: Convert timestamps ONCE before loop (prevents thousands of warnings)
    print("   Converting timestamps...")
    df_balanced['timestamp'] = pd.to_datetime(df_balanced['timestamp'], errors='coerce')
    
    # CRITICAL FIX: Get add_to_cart index to filter it out (prevents data leak)
    if 'add_to_cart' in action_to_idx:
        add_to_cart_idx = action_to_idx['add_to_cart']
        print(f"   ⚠️  Filtering out 'add_to_cart' action (idx={add_to_cart_idx}) to prevent data leak")
    else:
        add_to_cart_idx = None
        print(f"   ℹ️  'add_to_cart' not found in vocabulary - no filtering needed")
    
    # Track sessions with only add_to_cart (will be filtered out)
    empty_sessions_count = 0
    empty_sessions = []
    session_features_list = []
    
    for session_id in unique_sessions:
        session_data = df_balanced[df_balanced['session_id'] == session_id].sort_values('timestamp')
        
        intent_str = session_data['intent'].iloc[0]
        urgency_val = session_data['urgency'].iloc[0]
        intent = intent_map[intent_str]
        
        # CRITICAL FIX: Remove 'add_to_cart' from input sequence to prevent data leak
        action_list = [action_to_idx[action] for action in session_data['action'].values]
        if add_to_cart_idx is not None:
            action_sequence = [action_id for action_id in action_list if action_id != add_to_cart_idx]
        else:
            action_sequence = action_list
        
        # Skip sessions that become empty after filtering
        if len(action_sequence) == 0:
            empty_sessions_count += 1
            empty_sessions.append(session_id)
            continue
        
        # FEATURE ENGINEERING: Calculate session-level features
        # These features help distinguish Low Intent (aimless browsing) from High Intent (focused)
        
        # 1. Session duration (in seconds)
        if len(session_data) > 1:
            # Timestamps already converted above
            duration_seconds = (session_data['timestamp'].iloc[-1] - session_data['timestamp'].iloc[0]).total_seconds()
            # Handle zero or very short durations
            if duration_seconds <= 0 or pd.isna(duration_seconds):
                duration_seconds = 1.0  # Minimum 1 second
        else:
            duration_seconds = 1.0
        
        # 2. Unique pages/products viewed (exclude add_to_cart from count)
        if 'product_name' in session_data.columns:
            # Count unique products viewed (not including add_to_cart)
            viewed_products = session_data[session_data['action'] != 'add_to_cart']['product_name'].nunique()
        elif 'Product Card Id' in session_data.columns:
            viewed_products = session_data[session_data['action'] != 'add_to_cart']['Product Card Id'].nunique()
        else:
            # Fallback: use unique actions (excluding add_to_cart)
            viewed_products = session_data[session_data['action'] != 'add_to_cart']['action'].nunique()
        
        if viewed_products == 0:
            viewed_products = 1  # Minimum 1 to avoid division by zero
        
        # 3. Action density (actions per second) - measures "urgency" or "franticness"
        total_actions = len(action_sequence)  # Already filtered
        action_density = total_actions / duration_seconds if duration_seconds > 0 else 0
        
        # Normalize features (log scale for duration, linear for others)
        # Duration: log transform and normalize (sessions can range from 1s to hours)
        log_duration = np.log1p(duration_seconds) / 10.0  # Normalize to ~[0, 1] range
        
        # Unique pages: normalize by typical range (1-50 pages)
        normalized_pages = min(viewed_products / 50.0, 1.0)
        
        # Action density: normalize (typical range 0.1-10 actions/sec)
        normalized_density = min(action_density / 10.0, 1.0)
        
        session_features = [log_duration, normalized_pages, normalized_density]
        session_features_list.append(session_features)
        
        if len(action_sequence) > max_seq_length:
            action_sequence = action_sequence[:max_seq_length]
        else:
            action_sequence = action_sequence + [0] * (max_seq_length - len(action_sequence))
        
        sessions.append(action_sequence)
        intents_list.append(intent)
        urgencies_list.append(urgency_val)
    
    intents = np.array(intents_list)
    urgencies = np.array(urgencies_list)
    
    # Report filtered sessions
    if empty_sessions_count > 0:
        print(f"   ⚠️  Filtered out {empty_sessions_count} sessions with only 'add_to_cart' actions (not learnable)")
    
    print(f"   Final class distribution: {np.bincount(intents)}")
    
    session_features_array = np.array(session_features_list)
    
    # Stratified split (including features)
    train_sessions, test_sessions, train_intents, test_intents, train_urgencies, test_urgencies, train_features, test_features = \
        train_test_split(sessions, intents, urgencies, session_features_array, test_size=0.15, stratify=intents, random_state=42)
    
    train_sessions, val_sessions, train_intents, val_intents, train_urgencies, val_urgencies, train_features, val_features = \
        train_test_split(train_sessions, train_intents, train_urgencies, train_features,
                         test_size=0.15, stratify=train_intents, random_state=42)
    
    print(f"   Train: {len(train_sessions)} | Val: {len(val_sessions)} | Test: {len(test_sessions)}")
    print(f"   Train class dist: {np.bincount(train_intents)}")
    print(f"   ✓ Session features: [duration (log), unique_pages, action_density]")
    
    return {
        'train': (train_sessions, train_intents, train_urgencies, train_features),
        'val': (val_sessions, val_intents, val_urgencies, val_features),
        'test': (test_sessions, test_intents, test_urgencies, test_features),
        'action_to_idx': action_to_idx,
        'intent_encoder': intent_encoder
    }

def train_model(model, train_loader, val_loader, device, class_weights, num_epochs=35):
    """Training with adaptive focal loss - NO TARGET LEAK"""
    print("\n🚀 Starting training with Adaptive Focal Loss (no target leak)...")
    
    # Adaptive focal loss (lower gamma)
    focal_loss = AdaptiveFocalLoss(alpha=class_weights, gamma=1.5)
    # REMOVED: urgency_criterion - causes target leak
    
    # Adam optimizer with moderate LR
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.005)
    
    # Step LR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    best_val_f1 = 0
    patience = 8
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, intents, features in train_loader:
            sequences = sequences.to(device)
            intents = intents.to(device).squeeze()
            features = features.to(device)
            
            optimizer.zero_grad()
            
            intent_logits = model(sequences, features)  # Pass features
            
            loss = focal_loss(intent_logits, intents)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(intent_logits, 1)
            train_correct += (predicted == intents).sum().item()
            train_total += intents.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, intents, features in val_loader:
                sequences = sequences.to(device)
                intents = intents.to(device).squeeze()
                features = features.to(device)
                
                intent_logits = model(sequences, features)  # Pass features
                
                loss = focal_loss(intent_logits, intents)
                
                val_loss += loss.item()
                _, predicted = torch.max(intent_logits, 1)
                val_correct += (predicted == intents).sum().item()
                val_total += intents.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(intents.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Calculate macro F1
        _, _, val_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Per-class F1 for monitoring
        _, _, val_f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        print(f"  Per-class F1: High={val_f1_per_class[0]:.3f}, Med={val_f1_per_class[1]:.3f}, Low={val_f1_per_class[2]:.3f}")
        
        scheduler.step()
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'models/smart_balanced_intent_transformer.pth')
            print(f"  ✅ New best! (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping")
                break
    
    print(f"\n✅ Training done! Best Val F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load('models/smart_balanced_intent_transformer.pth'))
    
    return history

def evaluate_model(model, test_loader, intent_encoder, device):
    """Evaluation - NO TARGET LEAK"""
    print("\n📊 Evaluating (no target leak)...")
    
    model.eval()
    all_intents = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if len(batch) == 3:
                sequences, intents, features = batch
            elif len(batch) == 2:
                sequences, intents = batch
                batch_size = sequences.shape[0]
                features = torch.zeros(batch_size, 3, device=device)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} items")
            
            sequences = sequences.to(device)
            intents = intents.squeeze()
            # Ensure intents is always 1D (handle 0-d array from single sample)
            if intents.dim() == 0:
                intents = intents.unsqueeze(0)
            features = features.to(device)
            
            # Ensure features are properly shaped [batch_size, 3]
            if features.dim() == 0:
                # 0-d scalar - create a proper tensor
                features = torch.zeros(1, 3, device=device)
            elif features.dim() == 1:
                # Single sample - ensure it's [1, 3] not [3]
                if len(features) == 3:
                    features = features.unsqueeze(0)
                else:
                    features = features.unsqueeze(0) if features.shape[0] == sequences.shape[0] else torch.zeros(sequences.shape[0], 3, device=device)
            
            # Ensure features match batch size
            batch_size = sequences.shape[0]
            if features.shape[0] != batch_size:
                features = torch.zeros(batch_size, 3, device=device)
            
            # Ensure exactly 3 features
            if features.shape[1] != 3:
                if features.shape[1] < 3:
                    padding = torch.zeros(batch_size, 3 - features.shape[1], device=device)
                    features = torch.cat([features, padding], dim=1)
                else:
                    features = features[:, :3]
            
            try:
                intent_logits = model(sequences, features)  # Pass features
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(f"  sequences shape: {sequences.shape}")
                print(f"  features shape: {features.shape}")
                raise
            probs = torch.softmax(intent_logits, dim=1)
            _, predicted = torch.max(intent_logits, 1)
            
            # Convert to numpy, ensuring they're 1D arrays
            intents_np = intents.cpu().numpy()
            if intents_np.ndim == 0:
                intents_np = np.array([intents_np.item()])
            all_intents.extend(intents_np.flatten())
            
            pred_np = predicted.cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np.item()])
            all_predictions.extend(pred_np.flatten())
            
            probs_np = probs.cpu().numpy()
            all_probs.extend(probs_np if probs_np.ndim > 0 else [probs_np])
    
    accuracy = accuracy_score(all_intents, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_intents, all_predictions, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_intents, all_predictions, average=None, zero_division=0)
    
    cm = confusion_matrix(all_intents, all_predictions)
    
    all_probs = np.array(all_probs)
    auroc = roc_auc_score(all_intents, all_probs, multi_class='ovr', average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'confusion_matrix': cm.tolist(),
        'class_names': intent_encoder.classes_.tolist(),
        'per_class_metrics': {
            intent_encoder.classes_[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support[i])
            }
            for i in range(len(intent_encoder.classes_))
        },
        'test_samples': len(all_intents)
    }
    
    print("\n" + "="*60)
    print("📊 SMART BALANCED MODEL RESULTS")
    print("="*60)
    print(f"Overall Accuracy:  {accuracy*100:.2f}%")
    print(f"Weighted F1:       {f1:.4f}")
    print(f"AUROC:             {auroc:.4f}")
    print(f"\nPer-Class F1:")
    for i, class_name in enumerate(intent_encoder.classes_):
        status = "✅" if f1_per_class[i] >= 0.45 else "⚠️ "
        print(f"  {status} {class_name:15s}: {f1_per_class[i]:.4f} (n={support[i]})")
    print("="*60 + "\n")
    
    return metrics

def create_visualizations(history, metrics, intent_encoder):
    """Create visualizations"""
    os.makedirs('results/smart_balanced_intent_transformer', exist_ok=True)
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=intent_encoder.classes_,
                yticklabels=intent_encoder.classes_)
    plt.title('Smart Balanced Intent Transformer - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/smart_balanced_intent_transformer/confusion_matrix.png', dpi=300)
    plt.close()

def main():
    """Main"""
    print("="*70)
    print("  SMART BALANCED INTENT TRANSFORMER")
    print("  Target: 70%+ accuracy with ALL classes F1 > 0.45")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}\n")
    
    data = prepare_smart_balanced_data()
    
    # CORRECT class weights using sklearn's balanced calculation
    print("\n📊 Calculating 'balanced' class weights...")
    train_intents = data['train'][1]
    
    # This is the correct way to calculate balanced weights
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_intents),
        y=train_intents
    )
    
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"   Correct balanced weights: {class_weights.cpu().numpy()}\n")
    
    # Create datasets with features
    train_dataset = SmartDataset(data['train'][0], data['train'][1], data['train'][3])
    val_dataset = SmartDataset(data['val'][0], data['val'][1], data['val'][3])
    test_dataset = SmartDataset(data['test'][0], data['test'][1], data['test'][3])
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)
    
    # Initialize model with 3 session-level features
    model = SmartIntentTransformer(
        vocab_size=len(data['action_to_idx']),
        d_model=112,
        nhead=4,
        num_layers=3,
        dropout=0.35,
        num_features=3  # [duration, unique_pages, action_density]
    ).to(device)
    
    print(f"📊 Params: {sum(p.numel() for p in model.parameters()):,}\n")
    
    history = train_model(model, train_loader, val_loader, device, class_weights, num_epochs=35)
    metrics = evaluate_model(model, test_loader, data['intent_encoder'], device)
    create_visualizations(history, metrics, data['intent_encoder'])
    
    os.makedirs('results/smart_balanced_intent_transformer', exist_ok=True)
    
    with open('results/smart_balanced_intent_transformer/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save encoders for production use
    os.makedirs('models', exist_ok=True)
    with open('models/intent_encoders.pkl', 'wb') as f:
        pickle.dump({
            'action_to_idx': data['action_to_idx'],
            'intent_encoder': data['intent_encoder']
        }, f)
    print("✅ Saved encoders to models/intent_encoders.pkl")
    
    print("="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nResults: results/smart_balanced_intent_transformer/")
    print(f"Model: models/smart_balanced_intent_transformer.pth")
    print(f"Encoders: models/intent_encoders.pkl\n")
    
    # Final check
    medium_f1 = metrics['per_class_metrics']['Medium Intent']['f1_score']
    if medium_f1 >= 0.45 and metrics['accuracy'] >= 0.68:
        print("✅ SUCCESS: Model training completed successfully!")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"   Medium F1: {medium_f1:.4f}\n")
    else:
        print("⚠️  Model needs improvement")
        print(f"   Target: Acc≥68% AND Medium F1≥0.45")
        print(f"   Current: Acc={metrics['accuracy']*100:.2f}%, Medium F1={medium_f1:.4f}\n")

if __name__ == "__main__":
    main()


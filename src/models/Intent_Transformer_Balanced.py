"""
Balanced Intent Transformer - Stable + Business-Smart
Addresses under-classification while maintaining stability
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class BalancedIntentDataset(Dataset):
    """Balanced dataset with business-focused augmentation."""
    
    def __init__(self, sequences, actions, intents, urgencies, augment=True):
        self.sequences = sequences
        self.actions = actions
        self.intents = intents
        self.urgencies = urgencies
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].clone()
        intent = self.intents[idx]
        urgency = self.urgencies[idx]
        
        # Business-focused augmentation
        if self.augment and torch.rand(1) < 0.15:
            # Light shuffling that preserves intent signals
            if len(sequence) > 4:
                middle = sequence[2:-2]
                if len(middle) > 1:
                    perm = torch.randperm(len(middle))
                    sequence[2:-2] = middle[perm]
        
        return {
            'sequence': sequence,
            'action': sequence,
            'intent': intent,
            'urgency': urgency
        }

class BalancedIntentTransformer(nn.Module):
    """Balanced transformer with business-focused architecture."""
    
    def __init__(self, vocab_size, d_model=112, nhead=4, num_layers=3, num_classes=3, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        
        # Balanced embeddings
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(100, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Balanced transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*3,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Business-focused classification heads
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
        self.urgency_regressor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights for better learning
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, action_sequences):
        batch_size, seq_len = action_sequences.shape
        
        # Embeddings
        action_emb = self.action_embedding(action_sequences)
        pos_emb = self.pos_embedding(torch.arange(seq_len, device=action_sequences.device))
        x = action_emb + pos_emb
        x = self.embed_dropout(x)
        x = self.layer_norm(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Combined pooling for better feature extraction
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        x = avg_pool + max_pool
        
        # Predictions
        intent_logits = self.classifier(x)
        urgency_pred = torch.sigmoid(self.urgency_regressor(x))
        
        return intent_logits, urgency_pred

def setup_colab_environment():
    """Set up Colab environment."""
    print("🛠️ Setting up Colab environment...")
    
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device('cuda')
    else:
        print("⚠️ No GPU available, using CPU")
        device = torch.device('cpu')
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("✅ CUDA optimizations enabled")
    
    return device

def load_and_balance_data(max_sessions=50000):
    """Load data and create business-balanced dataset."""
    print("📊 Loading and creating business-balanced dataset...")
    
    # Load processed data
    processed_path = 'data/processed/processed_access_logs.csv'
    df = pd.read_csv(processed_path)
    print(f"✅ Loaded processed data: {len(df)} records")
    
    # Sample sessions
    unique_sessions = df['session_id'].unique()
    if len(unique_sessions) > max_sessions:
        print(f"   Sampling {max_sessions} sessions from {len(unique_sessions)}...")
        np.random.seed(42)
        sampled_sessions = np.random.choice(unique_sessions, max_sessions, replace=False)
        df = df[df['session_id'].isin(sampled_sessions)]
    
    # Create intent labels with business-focused thresholds
    session_stats = df.groupby('session_id').agg({
        'action': ['count', lambda x: (x == 'add_to_cart').sum()]
    }).round(3)
    session_stats.columns = ['total_actions', 'add_to_cart_count']
    session_stats['add_to_cart_ratio'] = session_stats['add_to_cart_count'] / session_stats['total_actions']
    
    # Business-focused intent labeling (more sensitive to high-value users)
    def get_intent_label(ratio):
        if ratio >= 0.3:  # Lowered from 0.4 - catch more high intent
            return 'high'
        elif ratio >= 0.05:  # Lowered from 0.1 - catch more medium intent
            return 'medium'
        else:
            return 'low'
    
    session_stats['intent'] = session_stats['add_to_cart_ratio'].apply(get_intent_label)
    df_with_intent = df.merge(session_stats[['intent', 'add_to_cart_ratio']], 
                             left_on='session_id', right_index=True, how='left')
    
    def get_urgency(intent, ratio):
        if intent == 'high':
            return min(0.9, 0.4 + ratio)  # More sensitive
        elif intent == 'medium':
            return 0.2 + ratio * 0.5  # More sensitive
        else:
            return ratio * 0.4  # More sensitive
    
    df_with_intent['urgency'] = df_with_intent.apply(
        lambda x: get_urgency(x['intent'], x['add_to_cart_ratio']), axis=1
    )
    
    # Check class distribution
    intent_dist = df_with_intent.groupby('session_id')['intent'].first().value_counts()
    print(f"   Business-focused intent distribution: {intent_dist.to_dict()}")
    
    # Create balanced dataset with business focus
    print("   🔄 Creating business-balanced dataset...")
    session_intents = df_with_intent.groupby('session_id')['intent'].first()
    
    # Business-focused balancing: More high and medium intent samples
    high_sessions = session_intents[session_intents == 'high'].index
    medium_sessions = session_intents[session_intents == 'medium'].index
    low_sessions = session_intents[session_intents == 'low'].index
    
    # Use more high and medium intent samples
    target_high = min(len(high_sessions), 8000)
    target_medium = min(len(medium_sessions), 8000)
    target_low = min(len(low_sessions), 6000)  # Fewer low intent samples
    
    print(f"   Target distribution: High={target_high}, Medium={target_medium}, Low={target_low}")
    
    # Sample sessions
    balanced_sessions = []
    
    if len(high_sessions) >= target_high:
        sampled_high = np.random.choice(high_sessions, target_high, replace=False)
        balanced_sessions.extend(sampled_high)
    else:
        balanced_sessions.extend(high_sessions)
    
    if len(medium_sessions) >= target_medium:
        sampled_medium = np.random.choice(medium_sessions, target_medium, replace=False)
        balanced_sessions.extend(sampled_medium)
    else:
        balanced_sessions.extend(medium_sessions)
    
    if len(low_sessions) >= target_low:
        sampled_low = np.random.choice(low_sessions, target_low, replace=False)
        balanced_sessions.extend(sampled_low)
    else:
        balanced_sessions.extend(low_sessions)
    
    # Filter to balanced sessions
    df_balanced = df_with_intent[df_with_intent['session_id'].isin(balanced_sessions)]
    
    # Check new distribution
    balanced_dist = df_balanced.groupby('session_id')['intent'].first().value_counts()
    print(f"   Business-balanced distribution: {balanced_dist.to_dict()}")
    
    return df_balanced

def prepare_balanced_data(df, max_seq_length=12):
    """Prepare balanced data for training."""
    print("🔄 Preparing business-balanced data...")
    
    # Create action vocabulary
    actions = df['action'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    print(f"   Action vocabulary: {len(actions)} actions")
    
    # Create sequences
    sequences = []
    intents = []
    urgencies = []
    
    unique_sessions = df['session_id'].unique()
    print(f"   Processing {len(unique_sessions)} sessions...")
    
    for i, session_id in enumerate(unique_sessions):
        if i % 5000 == 0:
            print(f"   Progress: {i}/{len(unique_sessions)} ({i/len(unique_sessions)*100:.1f}%)")
        
        session_data = df[df['session_id'] == session_id].sort_values('timestamp')
        
        intent_str = session_data['intent'].iloc[0]
        urgency_val = session_data['urgency'].iloc[0]
        
        intent_map = {'high': 0, 'medium': 1, 'low': 2}
        intent = intent_map[intent_str]
        
        action_sequence = [action_to_idx[action] for action in session_data['action'].values]
        
        if len(action_sequence) > max_seq_length:
            action_sequence = action_sequence[:max_seq_length]
        else:
            action_sequence = action_sequence + [0] * (max_seq_length - len(action_sequence))
        
        sequences.append(action_sequence)
        intents.append(intent)
        urgencies.append(urgency_val)
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    intents = torch.tensor(intents, dtype=torch.long)
    urgencies = torch.tensor(urgencies, dtype=torch.float).unsqueeze(1)
    
    print(f"   Sequences: {sequences.shape}")
    print(f"   Intents: {intents.shape}")
    print(f"   Urgencies: {urgencies.shape}")
    
    return sequences, intents, urgencies, action_to_idx

def train_balanced_model(model, train_loader, val_loader, device, num_epochs=30):
    """Train with business-focused settings."""
    print("🚀 Training business-focused model...")
    
    # Business-focused loss with class weights
    class_weights = torch.tensor([1.5, 1.2, 0.8], device=device)  # Favor high and medium intent
    intent_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    urgency_criterion = nn.MSELoss()
    
    # Balanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=7e-4, weight_decay=8e-4)
    
    # Balanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//2, eta_min=1e-5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            intents = batch['intent'].to(device)
            urgencies = batch['urgency'].to(device)
            
            optimizer.zero_grad()
            
            intent_logits, urgency_pred = model(sequences)
            
            intent_loss = intent_criterion(intent_logits, intents)
            urgency_loss = urgency_criterion(urgency_pred, urgencies)
            total_loss = intent_loss + 0.3 * urgency_loss
            
            total_loss.backward()
            
            # Balanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(intent_logits.data, 1)
            train_total += intents.size(0)
            train_correct += (predicted == intents).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                intents = batch['intent'].to(device)
                urgencies = batch['urgency'].to(device)
                
                intent_logits, urgency_pred = model(sequences)
                
                intent_loss = intent_criterion(intent_logits, intents)
                urgency_loss = urgency_criterion(urgency_pred, urgencies)
                total_loss = intent_loss + 0.3 * urgency_loss
                
                val_loss += total_loss.item()
                _, predicted = torch.max(intent_logits.data, 1)
                val_total += intents.size(0)
                val_correct += (predicted == intents).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            status = "BEST!"
        else:
            patience_counter += 1
            status = f"Patience: {patience_counter}/{patience}"
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {current_lr:.2e} {status}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc, train_losses, val_losses, train_accs, val_accs, learning_rates

def plot_training_progress(train_losses, val_losses, train_accs, val_accs, learning_rates):
    """Plot training progress."""
    print("📊 Plotting training progress...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss (Business-Focused)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy (Business-Focused)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(learning_rates, label='Learning Rate', color='green')
    ax3.set_title('Learning Rate Schedule (Balanced)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')
    
    # Combined plot
    ax4.plot(train_accs, label='Training Accuracy', color='blue', alpha=0.7)
    ax4.plot(val_accs, label='Validation Accuracy', color='red', alpha=0.7)
    ax4.plot([x * 100 for x in train_losses], label='Training Loss (scaled)', color='blue', linestyle='--', alpha=0.5)
    ax4.plot([x * 100 for x in val_losses], label='Validation Loss (scaled)', color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Combined Training Progress (Business-Focused)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('balanced_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training progress plots saved as 'balanced_training_progress.png'")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    print("📊 Plotting confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Business-Focused Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('balanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Confusion matrix saved as 'balanced_confusion_matrix.png'")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def main():
    """Main function."""
    print("=" * 60)
    print("🧠 BALANCED INTENT TRANSFORMER - BUSINESS-FOCUSED")
    print("=" * 60)
    
    try:
        # Setup
        device = setup_colab_environment()
        
        # Load and balance data
        df_balanced = load_and_balance_data(max_sessions=50000)
        
        # Prepare data
        sequences, intents, urgencies, action_to_idx = prepare_balanced_data(df_balanced)
        
        # Create datasets
        train_dataset = BalancedIntentDataset(sequences, sequences, intents, urgencies, augment=True)
        val_dataset = BalancedIntentDataset(sequences, sequences, intents, urgencies, augment=False)
        test_dataset = BalancedIntentDataset(sequences, sequences, intents, urgencies, augment=False)
        
        # Split data
        train_size = int(0.8 * len(sequences))
        val_size = int(0.1 * len(sequences))
        test_size = len(sequences) - train_size - val_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, len(sequences)))
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
        
        print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create balanced model
        vocab_size = len(action_to_idx)
        model = BalancedIntentTransformer(vocab_size, dropout=0.4).to(device)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        best_acc, train_losses, val_losses, train_accs, val_accs, learning_rates = train_balanced_model(
            model, train_loader, val_loader, device
        )
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Best validation accuracy: {best_acc:.2f}%")
        
        # Plot training progress
        plot_training_progress(train_losses, val_losses, train_accs, val_accs, learning_rates)
        
        # Test and confusion matrix
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(device)
                intents = batch['intent'].to(device)
                
                intent_logits, _ = model(sequences)
                _, predicted = torch.max(intent_logits.data, 1)
                test_total += intents.size(0)
                test_correct += (predicted == intents).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(intents.cpu().numpy())
        
        test_acc = 100 * test_correct / test_total
        print(f"📊 Test accuracy: {test_acc:.2f}%")
        
        # Plot confusion matrix
        class_names = ['High Intent', 'Medium Intent', 'Low Intent']
        plot_confusion_matrix(all_targets, all_predictions, class_names)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/balanced_intent_transformer.pth')
        print("💾 Model saved to models/balanced_intent_transformer.pth")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



















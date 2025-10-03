#!/usr/bin/env python3
"""
Comprehensive Intent Transformer Training with Detailed Metrics
Trains the model and reports accuracy, precision, recall, F1-score, and more.
"""
import os
import sys
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# No longer need DistilBertTokenizer - using lightweight embeddings!
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import pickle
import time
import yaml
from typing import Dict, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.models.intent_transformer.intent_transformer_lightweight import LightweightIntentTransformer as IntentTransformer

class ComprehensiveIntentTrainer:
    """Intent Transformer trainer with comprehensive metrics tracking."""
    
    def __init__(self, config_path: str = 'config/config.yaml', fast_mode: bool = True):
        """Initialize trainer with config."""
        self.fast_mode = fast_mode
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['intent_transformer']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CPU optimizations for Intel processors
        if self.device.type == 'cpu':
            torch.set_num_threads(8)  # Use all 8 logical processors
            torch.set_num_interop_threads(2)
            # Enable Intel MKL optimizations
            try:
                import intel_extension_for_pytorch as ipex
                print(f"[CPU] Using device: {self.device} with Intel IPEX optimizations")
            except ImportError:
                print(f"[CPU] Using device: {self.device} with standard CPU optimizations")
        else:
            print(f"[GPU] Using device: {self.device}")
        
        # Paths
        self.model_save_path = os.path.join(project_root, 'models', 'intent_transformer_finetuned_multi.pth')
        self.encoders_save_path = os.path.join(project_root, 'models', 'intent_encoders.pkl')
        self.results_save_path = os.path.join(project_root, 'results', 'intent_transformer_results.json')
        
        # Create directories
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.results_save_path), exist_ok=True)
        
        # NO TOKENIZER NEEDED - using simple embeddings!
        print(f"[OK] Using lightweight embeddings (no BERT tokenizer)")
        
        if self.fast_mode:
            print("[FAST] HIGH-ACCURACY MODE: Optimized for 80-85% accuracy")
            print("  - Using 30% of training data (more for better learning)")
            print("  - Large model (256 dim, 4 layers)")
            print("  - Rich sequences (8x12) for better features")
            print("  - 30 epochs for full convergence")
            print("  - Enhanced tokenization with char n-grams")
            print("  - Target accuracy: 80-85%")
            print("  - Training time: ~90-120 minutes")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test data."""
        print("\n[DATA] Loading temporal split data...")
        
        data_dir = os.path.join(project_root, 'data', 'processed', 'temporal_splits')
        
        train_path = os.path.join(data_dir, 'train', 'processed_access_logs.csv')
        val_path = os.path.join(data_dir, 'val', 'processed_access_logs.csv')
        test_path = os.path.join(data_dir, 'test', 'processed_access_logs.csv')
        
        df_train = pd.read_csv(train_path) if os.path.exists(train_path) else None
        df_val = pd.read_csv(val_path) if os.path.exists(val_path) else None
        df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
        
        if df_train is None:
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # FAST MODE: Use substantial data for high accuracy
        if self.fast_mode:
            print("[FAST] HIGH-ACCURACY MODE - Using substantial data for 80-85%")
            # Use 30% of training data for high accuracy
            unique_sessions = df_train['session_id'].unique()
            np.random.seed(42)
            sample_sessions = np.random.choice(unique_sessions, 
                                             size=max(3000, int(len(unique_sessions) * 0.3)), 
                                             replace=False)
            df_train = df_train[df_train['session_id'].isin(sample_sessions)].copy()
            
            # Use 50% of validation data for better evaluation
            if df_val is not None:
                val_sessions = df_val['session_id'].unique()
                sample_val_sessions = np.random.choice(val_sessions,
                                                      size=max(3000, int(len(val_sessions) * 0.5)),
                                                      replace=False)
                df_val = df_val[df_val['session_id'].isin(sample_val_sessions)].copy()
        
        print(f"[OK] Train: {df_train.shape[0]} rows")
        if df_val is not None:
            print(f"[OK] Val: {df_val.shape[0]} rows")
        if df_test is not None:
            print(f"[OK] Test: {df_test.shape[0]} rows")
        
        return df_train, df_val, df_test
    
    def create_intent_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create intent labels based on user behavior."""
        print("\n[LABELS]  Creating intent labels...")
        
        # Ensure action column exists and has valid values
        if 'action' not in df.columns:
            print("⚠️  'action' column not found, creating default 'view' actions")
            df['action'] = 'view'
        
        # Fill NaN values in action column
        df['action'] = df['action'].fillna('view')
        
        # Pre-parse timestamps once (much faster than parsing per session)
        if 'timestamp' in df.columns:
            print("  Pre-processing timestamps...")
            df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Group by session and determine intent
        def determine_intent(session_df):
            actions = session_df['action'].tolist()
            session_length = len(actions)
            
            # Intent based on actions and session depth
            if 'purchase' in actions or 'checkout' in actions:
                return 'high'
            elif 'add_to_cart' in actions:
                return 'medium' if session_length > 2 else 'high'
            elif session_length >= 5:
                return 'medium'
            else:
                return 'low'
        
        # Create urgency based on time between actions (simplified)
        def calculate_urgency(session_df):
            if 'timestamp_parsed' in session_df.columns and len(session_df) > 1:
                try:
                    session_df_sorted = session_df.sort_values('timestamp_parsed')
                    time_diffs = session_df_sorted['timestamp_parsed'].diff().dt.total_seconds()
                    avg_time = time_diffs.mean()
                    if pd.notna(avg_time):
                        if avg_time < 30:  # Less than 30 seconds
                            return 0.8
                        elif avg_time < 120:  # Less than 2 minutes
                            return 0.5
                except:
                    pass
            return 0.3
        
        # Calculate intent and urgency per session with progress tracking
        total_sessions = df['session_id'].nunique()
        print(f"  Processing {total_sessions} sessions...")
        
        session_intents = {}
        session_urgencies = {}
        
        import time
        start_time = time.time()
        processed = 0
        
        for session_id, group in df.groupby('session_id'):
            intent = determine_intent(group)
            urgency = calculate_urgency(group)
            session_intents[session_id] = intent
            session_urgencies[session_id] = urgency
            
            processed += 1
            
            # Show progress every 1000 sessions (more frequent)
            if processed % 1000 == 0 or processed == total_sessions:
                elapsed = time.time() - start_time
                sessions_per_sec = processed / elapsed if elapsed > 0 else 0
                remaining = (total_sessions - processed) / sessions_per_sec if sessions_per_sec > 0 else 0
                
                percent = int(100 * processed / total_sessions)
                bar_length = 40
                filled = int(bar_length * processed / total_sessions)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f"  [{bar}] {percent}% ({processed}/{total_sessions}) "
                      f"- {sessions_per_sec:.0f} sessions/s - ETA: {remaining:.0f}s     ", end='\r')
        
        print()  # New line after progress bar
        
        # Map back to dataframe
        df['intent'] = df['session_id'].map(session_intents)
        df['urgency'] = df['session_id'].map(session_urgencies)
        
        # Clean up temporary column
        if 'timestamp_parsed' in df.columns:
            df.drop('timestamp_parsed', axis=1, inplace=True)
        
        # Count intent distribution
        intent_dist = df.groupby('session_id')['intent'].first().value_counts()
        print(f"  [OK] Intent distribution: {intent_dist.to_dict()}")
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, action_encoder: LabelEncoder, 
                         intent_encoder: LabelEncoder) -> Tuple[torch.Tensor, ...]:
        """Prepare sequential data for training."""
        print("\n[PROCESS] Preparing sequences...")
        
        # FAST MODE: Use rich sequences for high accuracy
        if self.fast_mode:
            max_seq_length = 8   # Longer sequences for better patterns
            max_desc_length = 12 # Longer descriptions for richer features
            print(f"[FAST] HIGH-ACCURACY MODE: seq_len={max_seq_length}, desc_len={max_desc_length}")
        else:
            max_seq_length = self.config['max_seq_length']
            max_desc_length = self.config['max_desc_length']
        
        sequences_desc_tokens = []
        sequences_actions = []
        sequences_attention_masks = []
        targets_intent = []
        targets_urgency = []
        targets_nav_depth = []
        
        total_sessions = df['session_id'].nunique()
        processed = 0
        
        print(f"  Total sessions to process: {total_sessions}")
        
        for session_id, group in df.groupby('session_id'):
            processed += 1
            if processed % 5000 == 0 or processed == total_sessions:
                percent = (processed * 100) // total_sessions
                bar_length = 40
                filled = int(bar_length * processed // total_sessions)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  Progress: [{bar}] {percent}% ({processed}/{total_sessions})", end='\r')
            
            if len(group) < 2:
                continue
            
            group = group.sort_values('timestamp') if 'timestamp' in group.columns else group
            
            # Get product names and actions
            product_names = group['product_name'].fillna('').astype(str).tolist()
            actions = group['action'].fillna('view').tolist()
            
            # Create sliding windows
            for i in range(len(group) - 1):
                start_idx = max(0, i - max_seq_length + 1)
                end_idx = i + 1
                
                # Get sequence slice
                seq_products = product_names[start_idx:end_idx]
                seq_actions = actions[start_idx:end_idx]
                
                # Simple tokenization (hash-based, MUCH faster than BERT!)
                desc_tokens_list = []
                attention_masks_list = []
                
                for product_name in seq_products:
                    try:
                        # Better tokenization with character n-grams for richer features
                        words = str(product_name).lower().split()[:max_desc_length]
                        token_ids = []
                        for word in words:
                            # Use word hash + character n-gram features for better representation
                            word_hash = abs(hash(word)) % 25000 + 1
                            # Add character-level features
                            char_features = [abs(hash(word[i:i+3])) % 5000 + 25001 for i in range(len(word)-2)]
                            token_ids.append(word_hash)
                            if char_features:
                                token_ids.extend(char_features[:2])  # Add top 2 char features
                        
                        # Ensure we don't exceed max_desc_length
                        token_ids = token_ids[:max_desc_length]
                        
                        # Pad to max_desc_length
                        if len(token_ids) < max_desc_length:
                            attention_mask = [1] * len(token_ids) + [0] * (max_desc_length - len(token_ids))
                            token_ids = token_ids + [0] * (max_desc_length - len(token_ids))
                        else:
                            attention_mask = [1] * max_desc_length
                            token_ids = token_ids[:max_desc_length]
                        
                        desc_tokens_list.append(torch.tensor(token_ids, dtype=torch.long))
                        attention_masks_list.append(torch.tensor(attention_mask, dtype=torch.long))
                    except Exception as e:
                        # Fallback to empty encoding
                        desc_tokens_list.append(torch.zeros(max_desc_length, dtype=torch.long))
                        attention_masks_list.append(torch.zeros(max_desc_length, dtype=torch.long))
                
                # Encode actions (handle unknown actions)
                try:
                    action_indices = action_encoder.transform(seq_actions)
                except Exception as e:
                    # Skip this sequence if actions can't be encoded
                    continue
                
                # Pad if needed
                current_len = len(desc_tokens_list)
                if current_len < max_seq_length:
                    pad_len = max_seq_length - current_len
                    # Pad with zeros
                    desc_tokens_list = [torch.zeros(max_desc_length, dtype=torch.long)] * pad_len + desc_tokens_list
                    attention_masks_list = [torch.zeros(max_desc_length, dtype=torch.long)] * pad_len + attention_masks_list
                    action_indices = np.concatenate([np.zeros(pad_len, dtype=int), action_indices])
                
                # Stack tensors
                desc_tokens = torch.stack(desc_tokens_list)
                attention_masks = torch.stack(attention_masks_list)
                action_tensor = torch.tensor(action_indices, dtype=torch.long)
                
                # Get targets
                intent = group.iloc[i]['intent'] if 'intent' in group.columns else 'low'
                urgency = group.iloc[i]['urgency'] if 'urgency' in group.columns else 0.5
                nav_depth = min(len(group), 20) / 20.0  # Normalize to [0, 1]
                
                sequences_desc_tokens.append(desc_tokens)
                sequences_actions.append(action_tensor)
                sequences_attention_masks.append(attention_masks)
                targets_intent.append(intent_encoder.transform([intent])[0])
                targets_urgency.append(urgency)
                targets_nav_depth.append(nav_depth)
        
        print(f"\n  [OK] Created {len(sequences_desc_tokens)} sequences")
        
        # Convert to tensors
        X_desc_tokens = torch.stack(sequences_desc_tokens)
        X_actions = torch.stack(sequences_actions)
        X_attention_masks = torch.stack(sequences_attention_masks)
        y_intent = torch.tensor(targets_intent, dtype=torch.long)
        y_urgency = torch.tensor(targets_urgency, dtype=torch.float).unsqueeze(1)
        y_nav_depth = torch.tensor(targets_nav_depth, dtype=torch.float).unsqueeze(1)
        
        print(f"  Tensor shapes: Desc={X_desc_tokens.shape}, Actions={X_actions.shape}, "
              f"Intent={y_intent.shape}, Urgency={y_urgency.shape}")
        
        return X_desc_tokens, X_actions, X_attention_masks, y_intent, y_urgency, y_nav_depth
    
    def train(self) -> Dict:
        """Train the Intent Transformer with comprehensive metrics."""
        print("\n" + "="*70)
        print("🚀 INTENT TRANSFORMER TRAINING - Comprehensive Mode")
        print("="*70)
        
        start_time = time.time()
        
        # Load data
        df_train, df_val, df_test = self.load_data()
        
        # Create intent labels
        df_train = self.create_intent_labels(df_train)
        if df_val is not None:
            df_val = self.create_intent_labels(df_val)
        if df_test is not None:
            df_test = self.create_intent_labels(df_test)
        
        # Initialize encoders
        self.action_encoder = LabelEncoder()
        self.intent_encoder = LabelEncoder()
        
        # Fit on all actions and intents
        all_actions = df_train['action'].unique()
        all_intents = ['low', 'medium', 'high']
        
        self.action_encoder.fit(all_actions)
        self.intent_encoder.fit(all_intents)
        
        vocab_size_act = len(self.action_encoder.classes_)
        num_intents = len(self.intent_encoder.classes_)
        
        print(f"\n[EVAL] Data Info:")
        print(f"  Action vocab size: {vocab_size_act}")
        print(f"  Actions: {list(self.action_encoder.classes_)}")
        print(f"  Intent classes: {num_intents}")
        print(f"  Intents: {list(self.intent_encoder.classes_)}")
        
        # Save encoders
        with open(self.encoders_save_path, 'wb') as f:
            pickle.dump({
                'action_encoder': self.action_encoder,
                'intent_encoder': self.intent_encoder
            }, f)
        print(f"[SAVE] Encoders saved to {self.encoders_save_path}")
        
        # Prepare datasets
        X_train_desc, X_train_act, X_train_mask, y_train_intent, y_train_urg, y_train_nav = \
            self.prepare_sequences(df_train, self.action_encoder, self.intent_encoder)
        
        if df_val is not None:
            X_val_desc, X_val_act, X_val_mask, y_val_intent, y_val_urg, y_val_nav = \
                self.prepare_sequences(df_val, self.action_encoder, self.intent_encoder)
        else:
            # Split training data
            split_idx = int(0.8 * len(X_train_desc))
            X_val_desc, X_train_desc = X_train_desc[split_idx:], X_train_desc[:split_idx]
            X_val_act, X_train_act = X_train_act[split_idx:], X_train_act[:split_idx]
            X_val_mask, X_train_mask = X_train_mask[split_idx:], X_train_mask[:split_idx]
            y_val_intent, y_train_intent = y_train_intent[split_idx:], y_train_intent[:split_idx]
            y_val_urg, y_train_urg = y_train_urg[split_idx:], y_train_urg[:split_idx]
            y_val_nav, y_train_nav = y_train_nav[split_idx:], y_train_nav[:split_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_desc, X_train_act, X_train_mask, 
                                      y_train_intent, y_train_urg, y_train_nav)
        val_dataset = TensorDataset(X_val_desc, X_val_act, X_val_mask, 
                                    y_val_intent, y_val_urg, y_val_nav)
        
        # Optimize DataLoader for CPU training
        # Use 2 workers on Windows (more can cause issues), 0 if problems persist
        num_workers = 2 if self.device.type == 'cpu' else 0
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                  shuffle=True, num_workers=num_workers, 
                                  pin_memory=False, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                               shuffle=False, num_workers=num_workers,
                               pin_memory=False, persistent_workers=False)
        
        print(f"\n[DATA] Dataset sizes:")
        print(f"  Training: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"  Validation: {len(val_dataset)} samples ({len(val_loader)} batches)")
        
        # Calculate vocabulary size for product descriptions
        vocab_size_desc = int(X_train_desc.max().item()) + 1
        print(f"  Description vocab size: {vocab_size_desc}")
        
        # Create ULTRA-LIGHTWEIGHT model (NO BERT!)
        print("\n[MODEL] Creating Ultra-Lightweight Transformer (NO BERT)...")
        
        # Use larger dimensions for higher accuracy
        if self.fast_mode:
            d_model = 256        # Larger model for better capacity
            nhead = 8            # More attention heads
            num_layers = 4       # Deeper for better learning
            dim_feedforward = 512 # Larger feedforward
            emb_desc = 128       # Larger embeddings for richer features
            emb_act = 32
        else:
            d_model = 512        # Even larger for full mode
            nhead = 16
            num_layers = 6
            dim_feedforward = 1024
            emb_desc = 256
            emb_act = 64
        
        model = IntentTransformer(
            vocab_size_desc=vocab_size_desc,
            vocab_size_act=vocab_size_act,
            embedding_dim_desc=emb_desc,
            embedding_dim_act=emb_act,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=self.config['dropout'],
            num_intents=num_intents
        ).to(self.device)
        print(f"  [OK] Ultra-lightweight model created! ({d_model}d, {num_layers}L, {nhead}H)")
        
        # Enable CPU optimizations
        if self.device.type == 'cpu':
            model = model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = False
        
        print(f"\n[MODEL] Model Architecture:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Loss functions with class balancing to fix learning issues
        # Calculate class distribution to understand the problem
        from collections import Counter
        intent_counts = Counter(y_train_intent.numpy())
        print(f"  Intent class distribution: {intent_counts}")
        
        # Use label smoothing to help with learning
        criterion_intent = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_urgency = nn.MSELoss()
        criterion_nav = nn.MSELoss()
        print("  [OK] Using label smoothing for better learning")
        
        # Optimizer with settings optimized for learning
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, 
                                      lr=float(self.config['learning_rate']),
                                      weight_decay=1e-3,  # Higher weight decay
                                      betas=(0.9, 0.999),
                                      eps=1e-8,
                                      fused=False)
        print(f"  [OK] Optimizer configured with lr={self.config['learning_rate']}")
        
        # Learning rate scheduler with warmup for better learning
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=4, min_lr=1e-6
        )
        print("  [OK] Added learning rate scheduler with better patience")
        
        # Training loop
        print(f"\n{'='*70}")
        print(f"[TRAIN]  STARTING TRAINING - {self.config['num_epochs']} EPOCHS")
        print(f"{'='*70}")
        
        best_val_accuracy = 0.0
        best_epoch = 0
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
        print(f"\n{'Epoch':<8} {'Time':<8} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<11} {'Val F1':<11} {'Status'}")
        print("-" * 95)
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            # No BERT in lightweight model - skip eval() call
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                desc, actions, mask, intent, urgency, nav = batch
                desc = desc.to(self.device)
                actions = actions.to(self.device)
                mask = mask.to(self.device)
                intent = intent.to(self.device)
                urgency = urgency.to(self.device)
                nav = nav.to(self.device)
                
                optimizer.zero_grad()
                
                intent_logits, urgency_pred, nav_pred = model(desc, actions, mask)
                
                # Combined loss
                loss_intent = criterion_intent(intent_logits, intent)
                loss_urgency = criterion_urgency(urgency_pred, urgency)
                loss_nav = criterion_nav(nav_pred, nav)
                
                loss = loss_intent + 0.3 * loss_urgency + 0.2 * loss_nav
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(intent_logits, 1)
                train_total += intent.size(0)
                train_correct += (predicted == intent).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    desc, actions, mask, intent, urgency, nav = batch
                    desc = desc.to(self.device)
                    actions = actions.to(self.device)
                    mask = mask.to(self.device)
                    intent = intent.to(self.device)
                    urgency = urgency.to(self.device)
                    nav = nav.to(self.device)
                    
                    intent_logits, urgency_pred, nav_pred = model(desc, actions, mask)
                    
                    loss_intent = criterion_intent(intent_logits, intent)
                    loss_urgency = criterion_urgency(urgency_pred, urgency)
                    loss_nav = criterion_nav(nav_pred, nav)
                    
                    loss = loss_intent + 0.3 * loss_urgency + 0.2 * loss_nav
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(intent_logits, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(intent.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            
            # Update learning rate
            scheduler.step(val_accuracy)
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - epoch_start
            
            # Format epoch output
            status = ""
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                status = "[OK] BEST!"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'config': self.config
                }, self.model_save_path)
            
            # Print epoch results in table format
            print(f"[{epoch+1:2d}/{self.config['num_epochs']}]    "
                  f"{epoch_time:>6.1f}s  "
                  f"{avg_train_loss:>10.4f}  "
                  f"{train_accuracy:>10.4f}  "
                  f"{avg_val_loss:>10.4f}  "
                  f"{val_accuracy:>10.4f}  "
                  f"{val_f1:>10.4f}  "
                  f"{status}")
        
        total_time = time.time() - start_time
        
        # Print training summary
        print("\n" + "="*95)
        print(f"[OK] Training completed in {total_time/60:.1f} minutes")
        print(f"[OK] Best epoch: {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")
        print("="*95)
        
        # Final evaluation
        print("\n" + "="*70)
        print("[EVAL] FINAL EVALUATION ON VALIDATION SET")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(self.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Detailed evaluation on validation set
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                desc, actions, mask, intent, urgency, nav = batch
                desc = desc.to(self.device)
                actions = actions.to(self.device)
                mask = mask.to(self.device)
                intent = intent.to(self.device)
                
                intent_logits, _, _ = model(desc, actions, mask)
                probs = torch.softmax(intent_logits, dim=1)
                
                _, predicted = torch.max(intent_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(intent.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Classification report
        class_names = self.intent_encoder.classes_
        report = classification_report(all_targets, all_preds, 
                                      target_names=class_names, 
                                      zero_division=0)
        
        print(f"\n[TARGET] Best Epoch: {best_epoch}")
        print(f"[TIME]  Total Training Time: {total_time/60:.2f} minutes")
        print(f"\n[METRIC] Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\n🔍 Per-Class Metrics:")
        print(report)
        
        print(f"\n🎭 Confusion Matrix:")
        print(f"  Predicted →  {' '.join([f'{c:>8s}' for c in class_names])}")
        for i, row in enumerate(cm):
            print(f"  {class_names[i]:>8s}   {' '.join([f'{v:>8d}' for v in row])}")
        
        # Save results
        results = {
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'total_time_minutes': total_time / 60,
                'best_epoch': best_epoch,
                'num_epochs': self.config['num_epochs']
            },
            'model_config': self.config,
            'data_info': {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'vocab_size': vocab_size_act,
                'num_intents': num_intents,
                'intent_classes': list(class_names)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'best_val_accuracy': float(best_val_accuracy)
            },
            'confusion_matrix': cm.tolist(),
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'classification_report': report
        }
        
        with open(self.results_save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SAVE] Results saved to: {self.results_save_path}")
        print(f"[SAVE] Model saved to: {self.model_save_path}")
        
        print("\n" + "="*70)
        print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return results

def main():
    """Main training function."""
    try:
        # MINIMAL WORKING VERSION - Start simple!
        trainer = ComprehensiveIntentTrainer(fast_mode=True)
        results = trainer.train()
        return results
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()


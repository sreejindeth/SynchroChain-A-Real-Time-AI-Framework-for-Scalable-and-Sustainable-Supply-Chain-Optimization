"""
Baseline Comparison Evaluation
Compares SynchroChain (Hybrid) vs Rule-Only vs ML-Only baselines
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure stdout for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

# ============================================================================
# BASELINE 1: RULE-ONLY
# ============================================================================

class RuleOnlyIntentPredictor:
    """Rule-based intent prediction using simple heuristics."""
    
    def predict(self, user_session):
        """
        Simple rule-based intent prediction.
        
        Rules:
        - High intent: add_to_cart count >= 2 OR checkout action exists
        - Medium intent: add_to_cart count == 1 OR view count > 10
        - Low intent: otherwise
        """
        if isinstance(user_session, str):
            user_session = [user_session]
        
        add_to_cart_count = user_session.count('add_to_cart') if isinstance(user_session, list) else 0
        has_checkout = 'checkout' in user_session if isinstance(user_session, list) else False
        view_count = user_session.count('view') if isinstance(user_session, list) else 0
        session_length = len(user_session) if isinstance(user_session, list) else 1
        
        # Rule-based classification
        if has_checkout or add_to_cart_count >= 2:
            intent_class = 0  # High
            intent_score = min(1.0, 0.7 + (add_to_cart_count * 0.1))
            urgency = min(1.0, 0.8 if has_checkout else 0.6)
        elif add_to_cart_count == 1 or view_count > 10:
            intent_class = 1  # Medium
            intent_score = min(1.0, 0.5 + (view_count / 20))
            urgency = min(1.0, 0.4 + (session_length / 30))
        else:
            intent_class = 2  # Low
            intent_score = min(1.0, 0.2 + (session_length / 50))
            urgency = min(1.0, 0.2 + (session_length / 40))
        
        return intent_class, intent_score, urgency


class RuleOnlyDelayRiskPredictor:
    """Rule-based delay risk prediction."""
    
    def predict(self, order_data):
        """
        Simple rule-based delay risk.
        
        Rules based on:
        - Shipping mode: Same Day (high risk), International (high risk)
        - Order value: High value (medium risk)
        - Order quantity: Large orders (medium risk)
        """
        delay_risk = 0.1  # Base risk
        
        shipping_mode = order_data.get('Shipping Mode', order_data.get('shipping_mode', 'Standard Class'))
        if shipping_mode == 'Same Day':
            delay_risk += 0.3
        if order_data.get('is_international', False):
            delay_risk += 0.2
        
        order_value = order_data.get('Sales', order_data.get('order_value', 0))
        if order_value > 1000:
            delay_risk += 0.15
        elif order_value > 500:
            delay_risk += 0.1
        
        quantity = order_data.get('Order Item Quantity', order_data.get('order_quantity', 1))
        if quantity > 5:
            delay_risk += 0.1
        
        # Convert to binary: threshold at 0.5
        delay_risk_binary = 1 if delay_risk >= 0.5 else 0
        
        return delay_risk, delay_risk_binary


# ============================================================================
# BASELINE 2: ML-ONLY (Simple Models)
# ============================================================================

class LSTMBaselineIntentPredictor:
    """Simple LSTM for intent prediction (ML-only baseline)."""
    
    def __init__(self, vocab_size, hidden_dim=64, num_layers=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        class SimpleLSTM(nn.Module):
            def __init__(self, vocab_size, hidden_dim, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
                self.classifier = nn.Linear(hidden_dim, 3)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                # Use last timestep
                last_hidden = lstm_out[:, -1, :]
                output = self.classifier(last_hidden)
                return output
        
        self.model = SimpleLSTM(vocab_size, hidden_dim, num_layers).to(self.device)
    
    def load_or_train(self, train_sequences, train_labels, action_encoder):
        """Load model if exists, otherwise train."""
        model_path = 'models/lstm_baseline_intent.pth'
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.action_encoder = action_encoder
            print(f"  [OK] Loaded LSTM baseline from {model_path}")
        else:
            print("  [INFO] Training LSTM baseline...")
            self._train(train_sequences, train_labels, action_encoder)
            torch.save(self.model.state_dict(), model_path)
            print(f"  [OK] Saved LSTM baseline to {model_path}")
        
        self.action_encoder = action_encoder
    
    def _train(self, train_sequences, train_labels, action_encoder):
        """Train simple LSTM model."""
        # Prepare data
        max_len = 50
        X_train = []
        y_train = []
        
        for seq, label in zip(train_sequences[:1000], train_labels[:1000]):  # Use subset for speed
            if isinstance(seq, str):
                seq = [seq]
            encoded = []
            for a in seq:
                if a in action_encoder.classes_:
                    try:
                        encoded.append(action_encoder.transform([a])[0])
                    except:
                        continue
            if len(encoded) == 0:
                continue
            padded = encoded[:max_len] + [0] * (max_len - len(encoded))
            X_train.append(padded)
            y_train.append(label)
        
        if len(X_train) == 0:
            print("  [WARNING] No valid training data, skipping LSTM training")
            return
        
        X_train = torch.LongTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(5):  # Quick training
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/5: Loss = {loss.item():.4f}")
    
    def predict(self, user_session):
        """Predict intent using LSTM."""
        if not hasattr(self, 'action_encoder'):
            return 2, 0.2, 0.2  # Default
        
        if isinstance(user_session, str):
            user_session = [user_session]
        
        # Encode session
        encoded = []
        for a in user_session:
            if a in self.action_encoder.classes_:
                try:
                    encoded.append(self.action_encoder.transform([a])[0])
                except:
                    continue
        
        if len(encoded) == 0:
            return 2, 0.2, 0.2  # Default: Low intent
        
        # Pad to max_len
        max_len = 50
        padded = encoded[:max_len] + [0] * (max_len - len(encoded))
        x = torch.LongTensor([padded]).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)[0]
            intent_class = output.argmax().item()
            intent_score = probs[intent_class].item()
            urgency = min(1.0, intent_score * 1.2)  # Simple urgency estimation
        
        return intent_class, intent_score, urgency


class MLOnlyDelayRiskPredictor:
    """Simple ML classifier for delay risk (ML-only baseline)."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def load_or_train(self, train_data, train_labels):
        """Load model if exists, otherwise train."""
        model_path = 'models/ml_baseline_delay_risk.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model, self.scaler = pickle.load(f)
            print(f"  [OK] Loaded ML baseline from {model_path}")
        else:
            print("  [INFO] Training ML baseline (Logistic Regression)...")
            self._train(train_data, train_labels)
            with open(model_path, 'wb') as f:
                pickle.dump((self.model, self.scaler), f)
            print(f"  [OK] Saved ML baseline to {model_path}")
    
    def _train(self, X_train, y_train):
        """Train logistic regression model."""
        # Use subset for faster training
        subset_size = min(5000, len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_subset)
        
        # Train logistic regression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_subset)
        
        train_acc = self.model.score(X_train_scaled, y_subset)
        print(f"    Training accuracy: {train_acc:.4f}")
    
    def predict(self, order_features):
        """Predict delay risk using ML."""
        if self.model is None:
            # Return default if not trained
            return 0.5, 0
        
        # Scale features
        features_scaled = self.scaler.transform([order_features])
        
        # Predict
        prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (late)
        binary = self.model.predict(features_scaled)[0]  # Binary prediction
        
        return prob, binary


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_intent_prediction(system_name, predictor, test_sequences, test_intent_labels):
    """Evaluate intent prediction system."""
    print(f"\n  Evaluating {system_name} Intent Prediction...")
    
    all_predictions = []
    all_labels = []
    
    # Limit evaluation to avoid long runtime
    eval_size = min(500, len(test_sequences))
    
    for i in range(eval_size):
        seq = test_sequences[i]
        true_label = test_intent_labels[i]
        try:
            pred_class, _, _ = predictor.predict(seq)
            all_predictions.append(pred_class)
            all_labels.append(true_label)
        except Exception as e:
            # Skip problematic sequences
            continue
    
    if len(all_predictions) == 0:
        print(f"  [WARNING] No valid predictions for {system_name}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': len(all_predictions)
    }


def evaluate_delay_risk_prediction(system_name, predictor, test_data, test_labels):
    """Evaluate delay risk prediction system."""
    print(f"\n  Evaluating {system_name} Delay Risk Prediction...")
    
    all_predictions = []
    all_labels = []
    
    # Limit evaluation to avoid long runtime
    eval_size = min(5000, len(test_data))
    
    for i in range(eval_size):
        order_data = test_data[i]
        true_label = test_labels[i]
        try:
            _, pred_binary = predictor.predict(order_data)
            all_predictions.append(int(pred_binary))
            all_labels.append(int(true_label))
        except Exception as e:
            # Skip problematic samples
            continue
    
    if len(all_predictions) == 0:
        print(f"  [WARNING] No valid predictions for {system_name}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': len(all_predictions)
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation."""
    print("="*80)
    print("BASELINE COMPARISON EVALUATION")
    print("="*80)
    
    results = {}
    
    # ========================================================================
    # 1. INTENT PREDICTION COMPARISON
    # ========================================================================
    print("\n[1] INTENT PREDICTION COMPARISON")
    print("-" * 80)
    
    try:
        # Load test data for intent
        print("\n  Loading test data for intent prediction...")
        from evaluate_intent_transformer_genuine import load_test_data, load_model_and_encoders
        
        _, action_encoder, intent_encoder = load_model_and_encoders()
        test_sequences, test_intent_labels, _ = load_test_data(action_encoder, intent_encoder)
        print(f"  [OK] Loaded {len(test_sequences)} test sequences")
        
        # SynchroChain (use actual metrics from file)
        print("\n  [SYNCHROCHAIN] Using actual metrics from results/intent_transformer/metrics.json")
        with open('results/intent_transformer/metrics.json', 'r') as f:
            synchrochain_metrics = json.load(f)
        
        results['synchrochain_intent'] = {
            'accuracy': synchrochain_metrics['accuracy'],
            'precision': synchrochain_metrics['precision'],
            'recall': synchrochain_metrics['recall'],
            'f1_score': synchrochain_metrics['f1_score']
        }
        
        # Rule-Only Baseline
        print("\n  [RULE-ONLY] Evaluating rule-based baseline...")
        rule_intent = RuleOnlyIntentPredictor()
        rule_result = evaluate_intent_prediction(
            'Rule-Only', rule_intent, test_sequences, test_intent_labels
        )
        if rule_result:
            results['rule_intent'] = rule_result
        
        # ML-Only Baseline (LSTM)
        print("\n  [ML-ONLY] Preparing LSTM baseline...")
        lstm_intent = LSTMBaselineIntentPredictor(vocab_size=len(action_encoder.classes_))
        # Use subset for training
        train_sequences = test_sequences[:500]
        train_labels = test_intent_labels[:500]
        lstm_intent.load_or_train(train_sequences, train_labels, action_encoder)
        lstm_result = evaluate_intent_prediction(
            'ML-Only', lstm_intent, test_sequences, test_intent_labels
        )
        if lstm_result:
            results['ml_intent'] = lstm_result
            
    except Exception as e:
        print(f"  [ERROR] Intent evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 2. DELAY RISK PREDICTION COMPARISON
    # ========================================================================
    print("\n\n[2] DELAY RISK PREDICTION COMPARISON")
    print("-" * 80)
    
    try:
        # Load test data for delay risk
        print("\n  Loading test data for delay risk prediction...")
        test_path = 'data/processed/temporal_splits_classification/test/gnn_nodes.csv'
        
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            print(f"  [OK] Loaded {len(df)} test samples")
            
            # Prepare features (matching training)
            feature_cols = [
                'Product Price',
                'Order Item Quantity',
                'Order Item Discount Rate',
                'Order Item Profit Ratio',
                'Sales',
                'Days for shipment (scheduled)',
                'Customer Segment',
                'Shipping Mode',
                'Market'
            ]
            
            X_test = df[feature_cols].fillna(0).values
            y_test = df['is_late'].values.astype(int)
            
            # Encode categorical features
            le_segment = LabelEncoder()
            le_shipping = LabelEncoder()
            le_market = LabelEncoder()
            
            X_test_encoded = X_test.copy()
            X_test_encoded[:, 6] = le_segment.fit_transform(df['Customer Segment'].astype(str))
            X_test_encoded[:, 7] = le_shipping.fit_transform(df['Shipping Mode'].astype(str))
            X_test_encoded[:, 8] = le_market.fit_transform(df['Market'].astype(str))
            
            # Convert to order_data format for rule-based predictor
            test_order_data = []
            for _, row in df.iterrows():
                order_data = {
                    'Shipping Mode': row['Shipping Mode'],
                    'Sales': row['Sales'],
                    'Order Item Quantity': row['Order Item Quantity'],
                    'is_international': False  # Simplified
                }
                test_order_data.append(order_data)
            
            # SynchroChain (use actual metrics from file)
            print("\n  [SYNCHROCHAIN] Using actual metrics from results/delay_classification_gnn/metrics.json")
            with open('results/delay_classification_gnn/metrics.json', 'r') as f:
                synchrochain_delay_metrics = json.load(f)
            
            results['synchrochain_delay'] = {
                'accuracy': synchrochain_delay_metrics['accuracy'],
                'precision': synchrochain_delay_metrics['precision'],
                'recall': synchrochain_delay_metrics['recall'],
                'f1_score': synchrochain_delay_metrics['f1_score']
            }
            
            # Rule-Only Baseline
            print("\n  [RULE-ONLY] Evaluating rule-based baseline...")
            rule_delay = RuleOnlyDelayRiskPredictor()
            rule_delay_result = evaluate_delay_risk_prediction(
                'Rule-Only', rule_delay, test_order_data, y_test
            )
            if rule_delay_result:
                results['rule_delay'] = rule_delay_result
            
            # ML-Only Baseline (Logistic Regression)
            print("\n  [ML-ONLY] Preparing ML baseline...")
            ml_delay = MLOnlyDelayRiskPredictor()
            # Train on subset
            train_subset_size = min(5000, len(X_test_encoded))
            ml_delay.load_or_train(X_test_encoded[:train_subset_size], y_test[:train_subset_size])
            ml_delay_result = evaluate_delay_risk_prediction(
                'ML-Only', ml_delay, X_test_encoded, y_test
            )
            if ml_delay_result:
                results['ml_delay'] = ml_delay_result
                
        else:
            print(f"  [WARNING] Test data not found: {test_path}")
            print("  [INFO] Using placeholder values (you should run fix_gnn_proper_classification.py first)")
            
    except Exception as e:
        print(f"  [ERROR] Delay risk evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 3. SAVE RESULTS
    # ========================================================================
    print("\n\n[3] SAVING RESULTS")
    print("-" * 80)
    
    os.makedirs('results/baseline_comparison', exist_ok=True)
    
    with open('results/baseline_comparison/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("  [OK] Results saved to results/baseline_comparison/comparison_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    
    if 'synchrochain_intent' in results:
        print("\n[INTENT PREDICTION]")
        print(f"  SynchroChain: Acc={results['synchrochain_intent']['accuracy']:.4f}, "
              f"F1={results['synchrochain_intent']['f1_score']:.4f}")
        if 'rule_intent' in results:
            print(f"  Rule-Only:   Acc={results['rule_intent']['accuracy']:.4f}, "
                  f"F1={results['rule_intent']['f1_score']:.4f}")
        if 'ml_intent' in results:
            print(f"  ML-Only:     Acc={results['ml_intent']['accuracy']:.4f}, "
                  f"F1={results['ml_intent']['f1_score']:.4f}")
    
    if 'synchrochain_delay' in results:
        print("\n[DELAY RISK PREDICTION]")
        print(f"  SynchroChain: Acc={results['synchrochain_delay']['accuracy']:.4f}, "
              f"F1={results['synchrochain_delay']['f1_score']:.4f}")
        if 'rule_delay' in results:
            print(f"  Rule-Only:   Acc={results['rule_delay']['accuracy']:.4f}, "
                  f"F1={results['rule_delay']['f1_score']:.4f}")
        if 'ml_delay' in results:
            print(f"  ML-Only:     Acc={results['ml_delay']['accuracy']:.4f}, "
                  f"F1={results['ml_delay']['f1_score']:.4f}")
    
    print("\n" + "="*80)
    print("[OK] Baseline comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()







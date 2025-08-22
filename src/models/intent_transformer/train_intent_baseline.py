# src/models/train_intent_baseline.py (Refined Task: Near-Future Prediction)
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_access_logs.csv')
MAX_SEQ_LENGTH = 8  # Number of initial actions to base prediction on
FUTURE_WINDOW = 3   # Predict if 'add_to_cart' happens in the next 3 actions
print(f"Using sequence length: {MAX_SEQ_LENGTH}, Future window: {FUTURE_WINDOW}")

print("Loading processed access logs...")
df_access = pd.read_csv(processed_data_path)
print(f"Access logs loaded: {df_access.shape}")

# Group by session_id
sessions = df_access.groupby('session_id')

# Prepare lists to collect features and labels
sequence_features = []
near_future_labels = [] # New label: 1 if 'add_to_cart' in next K actions

print("Preparing sequences for 'Near-Future Action Prediction'...")
for session_id, group in sessions:
    group_sorted = group.sort_values('timestamp').reset_index(drop=True)
    actions = group_sorted['action'].tolist()

    # We need at least MAX_SEQ_LENGTH + FUTURE_WINDOW actions to create a valid sample
    # (Or at least MAX_SEQ_LENGTH + 1 to check if future window exists)
    if len(actions) <= MAX_SEQ_LENGTH:
        continue

    # Iterate through the session to create overlapping sequences
    # Stop early enough so the future window fits within the session
    # If session is too short for the full window, check the remaining actions
    for i in range(len(actions) - MAX_SEQ_LENGTH):
        # Check bounds for the future window
        future_start = i + MAX_SEQ_LENGTH
        future_end = min(future_start + FUTURE_WINDOW, len(actions))
        
        # If there are no future actions to check, skip
        if future_start >= len(actions):
            continue

        # The sequence of actions to use as features
        seq_actions = actions[i:i + MAX_SEQ_LENGTH]
        
        # --- Label: Does 'add_to_cart' happen in the next FUTURE_WINDOW actions? ---
        future_actions = actions[future_start:future_end]
        label = 1 if 'add_to_cart' in future_actions else 0

        # --- Create Features from the sequence of actions (same as before) ---
        num_views = seq_actions.count('view')
        num_add_to_carts = seq_actions.count('add_to_cart')
        seq_len = len(seq_actions)
        unique_actions = len(set(seq_actions))
        last_action = seq_actions[-1] if seq_actions else 'none'
        last_action_encoded = 1 if last_action == 'add_to_cart' else (2 if last_action == 'view' else 0)

        # Add features and label
        sequence_features.append([num_views, num_add_to_carts, seq_len, unique_actions, last_action_encoded])
        near_future_labels.append(label)

# Convert to numpy arrays
if not sequence_features:
    raise ValueError("No valid sequences created. Check data or sequence length.")
    
X = np.array(sequence_features)
y = np.array(near_future_labels)

print(f"Final tensor shapes - Features: {X.shape}, Labels: {y.shape}")
unique_labels, counts = np.unique(y, return_counts=True)
print(f"Label distribution: {dict(zip(unique_labels, counts))}")
if len(counts) > 1:
    majority_baseline_acc = max(counts) / len(y)
    print(f"Majority class baseline accuracy: {majority_baseline_acc:.4f}")
else:
    print("Warning: Only one class found in labels.")

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Logistic Regression
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)

print(f"\n--- Logistic Regression Baseline Results (Near-Future Prediction, K={FUTURE_WINDOW}) ---")
print(f"Validation Accuracy: {acc:.4f}")
if len(counts) > 1:
    print(f"Improvement over baseline: {acc - majority_baseline_acc:.4f}")
print(f"Classification Report:")
print(classification_report(y_val, y_pred))
print(f"Feature Coefficients: {model.coef_[0]}")

# --- Sample Predictions ---
print("\n--- Sample Predictions ---")
for i in range(min(5, len(X_val_scaled))):
    features = X_val_scaled[i]
    true_label = y_val[i]
    pred_label = y_pred[i]
    pred_proba = model.predict_proba([features])[0]
    print(f"Sample {i}: Features={features[:3]}..., True={true_label}, Pred={pred_label}, Prob={pred_proba}")

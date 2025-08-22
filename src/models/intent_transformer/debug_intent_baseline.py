# src/models/debug_intent_baseline.py
"""Script to train a simple Logistic Regression baseline for the 'add_to_cart in next 3' task.
This helps diagnose if the signal is learnable with basic features.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# --- Configuration ---
# Assuming project structure and file locations from previous context
PROJECT_ROOT = r"D:\VIT\Project\Project" # HARDCODED project root

data_dir = os.path.join(PROJECT_ROOT, 'data')
processed_data_path = os.path.join(data_dir, 'processed', 'processed_access_logs.csv')
raw_orders_data_path = os.path.join(data_dir, 'raw', 'DataCoSupplyChainDataset.csv')

# Parameters matching the LSTM task
MAX_SEQ_LENGTH = 8
FUTURE_WINDOW = 3 # Predict 'add_to_cart' in the next 3 actions
print(f"Task: Predict if 'add_to_cart' happens in the next {FUTURE_WINDOW} actions based on the first {MAX_SEQ_LENGTH} actions.")

def load_and_prepare_baseline_data(processed_data_path, raw_orders_data_path, max_seq_length=MAX_SEQ_LENGTH, future_window=FUTURE_WINDOW):
    """Loads data and prepares simple features for the baseline model."""
    print("Loading processed access logs...")
    df_access = pd.read_csv(processed_data_path)
    print(f"Access logs loaded: {df_access.shape}")

    # Optional: Load raw data if you want to include product categories or other features
    # For this baseline, we'll stick to action and product name features.
    # print("Loading raw order data...")
    # try:
    #     df_orders = pd.read_csv(raw_orders_data_path, encoding='utf-8')
    # except UnicodeDecodeError:
    #     print("UTF-8 failed for order data, trying 'ISO-8859-1'")
    #     df_orders = pd.read_csv(raw_orders_data_path, encoding='ISO-8859-1')
    # print(f"Order data loaded: {df_orders.shape}")
    # # Create a product name -> category mapping if needed
    # # product_category_map = dict(zip(df_orders['Product Name'], df_orders['Category Name']))

    print("Preparing sequences and extracting baseline features...")
    sessions = df_access.groupby('session_id')

    # Feature lists
    features_list = []
    labels_list = []

    action_encoder = LabelEncoder()
    all_actions = df_access['action'].unique()
    action_encoder.fit(all_actions)
    print(f"Action vocabulary size: {len(action_encoder.classes_)}")

    total_sessions = len(sessions)
    processed_sessions = 0
    
    for session_id, group in sessions:
        processed_sessions += 1
        if processed_sessions % 50000 == 0 or processed_sessions == total_sessions:
             print(f"Processed {processed_sessions}/{total_sessions} sessions for features.")

        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        actions = group_sorted['action'].tolist()
        # product_names = group_sorted['product_name'].tolist() # If you want to use product names

        # Need enough actions for a full input sequence AND a future window to check
        if len(actions) < (max_seq_length + future_window):
            continue

        # --- Create sequences by sliding a window ---
        for i in range(len(actions) - max_seq_length - future_window + 1):
            # --- Input Sequence: Actions from index i to i+max_seq_length-1 ---
            input_actions = actions[i : i + max_seq_length]
            # input_product_names = product_names[i : i + max_seq_length] # If used

            # --- Label: Does 'add_to_cart' happen in the NEXT FUTURE_WINDOW actions? ---
            future_start = i + max_seq_length
            future_end = future_start + future_window
            future_actions = actions[future_start : future_end]
            label = 1 if 'add_to_cart' in future_actions else 0

            # --- Extract Simple Features from the Input Sequence ---
            features = {}
            
            # 1. Count of each action type
            for action_type in action_encoder.classes_:
                features[f'count_{action_type}'] = input_actions.count(action_type)

            # 2. Sequence length (should be max_seq_length, but safe)
            features['seq_length'] = len(input_actions)

            # 3. Diversity: Number of unique actions
            features['unique_actions'] = len(set(input_actions))

            # 4. Position of last action (1-based index)
            features['last_action_pos'] = max_seq_length # Always the last in the fixed window

            # 5. Last action type (encoded numerically)
            last_action = input_actions[-1] if input_actions else 'none'
            features['last_action_encoded'] = action_encoder.transform([last_action])[0]

            # 6. First action type (encoded numerically)
            first_action = input_actions[0] if input_actions else 'none'
            features['first_action_encoded'] = action_encoder.transform([first_action])[0]

            # Add features and label to lists
            # Convert features dict to a list in a consistent order
            feature_vector = [features[key] for key in sorted(features.keys())]
            features_list.append(feature_vector)
            labels_list.append(label)
            # --- End Feature Extraction ---

    if not features_list:
        raise ValueError("No valid sequences created for baseline features.")

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"Final feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    if len(counts) > 0:
        majority_baseline_acc = max(counts) / len(y)
        print(f"Majority class baseline accuracy: {majority_baseline_acc:.4f}")
    else:
        print("Warning: No labels found.")

    return X, y

def train_baseline_model():
    """Main function to train and evaluate the Logistic Regression baseline."""
    print("Loading and preparing data for Logistic Regression baseline...")
    X, y = load_and_prepare_baseline_data(
        processed_data_path, raw_orders_data_path, 
        max_seq_length=MAX_SEQ_LENGTH, future_window=FUTURE_WINDOW
    )

    if len(X) == 0:
        print("No data available for training the baseline.")
        return

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape[0]} samples, Val set: {X_val.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    start_time = time.time()
    
    # Increase max_iter if convergence warnings occur
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    # Predict and evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)

    print(f"\n--- Logistic Regression Baseline Results ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Improvement over baseline: {val_acc - (max(np.unique(y, return_counts=True)[1]) / len(y)):.4f}")
    print(f"Classification Report (Validation):")
    print(classification_report(y_val, y_pred_val))
    
    # Show feature importance (coefficients)
    # Note: This requires the feature names, which we lost by converting to array.
    # Let's recreate them for interpretation.
    feature_names = []
    # Recreate feature names based on the logic in load_and_prepare_baseline_data
    action_encoder = LabelEncoder()
    action_encoder.fit(pd.read_csv(processed_data_path)['action'].unique())
    for action_type in action_encoder.classes_:
        feature_names.append(f'count_{action_type}')
    feature_names.extend(['seq_length', 'unique_actions', 'last_action_pos', 'last_action_encoded', 'first_action_encoded'])
    
    print(f"\n--- Top Feature Coefficients ---")
    coef = model.coef_[0]
    # Sort features by absolute coefficient value
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    for i in range(min(10, len(sorted_idx))): # Top 10
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]}: {coef[idx]:.4f}")

if __name__ == "__main__":
    train_baseline_model()
    print("\nBaseline training and evaluation complete.")
"""
Rebuild GNN Scaler from Training Data
Fixes the scaler mismatch issue by fitting StandardScaler on all 9 features
"""
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_SPLIT = 'data/processed/temporal_splits_classification/train/gnn_nodes.csv'
SCALER_OUT = 'models/gnn_scalers.pkl'

feature_cols = [
    'Product Price',
    'Order Item Quantity',
    'Order Item Discount Rate',
    'Order Item Profit Ratio',
    'Sales',
    'Days for shipment (scheduled)',
    'Customer Segment',
    'Shipping Mode',
    'Market',
]

def main():
    print("="*60)
    print("REBUILDING GNN SCALER")
    print("="*60)
    
    # Check if training data exists
    if not os.path.exists(TRAIN_SPLIT):
        raise FileNotFoundError(f"Training data not found: {TRAIN_SPLIT}")
    
    print(f"\n[1/3] Loading training data from: {TRAIN_SPLIT}")
    df = pd.read_csv(TRAIN_SPLIT)
    print(f"  Loaded {len(df)} training samples")
    
    # Extract features (categoricals are already label-encoded in the split)
    print(f"\n[2/3] Extracting features (all 9 columns):")
    for i, col in enumerate(feature_cols, 1):
        if col in df.columns:
            print(f"  {i}. {col} - ✓")
        else:
            print(f"  {i}. {col} - ✗ MISSING!")
            raise ValueError(f"Column not found: {col}")
    
    X = df[feature_cols].values
    print(f"  Feature matrix shape: {X.shape}")
    
    # Fit StandardScaler on all 9 features
    print(f"\n[3/3] Fitting StandardScaler on all 9 features...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    print(f"  Mean: {scaler.mean_[:3]}... (first 3)")
    print(f"  Std:  {scaler.scale_[:3]}... (first 3)")
    print(f"  n_features_in_: {scaler.n_features_in_}")
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open(SCALER_OUT, 'wb') as f:
        pickle.dump({'feature_scaler': scaler}, f)
    
    print(f"\n✅ SUCCESS!")
    print(f"  Scaler saved to: {SCALER_OUT}")
    print(f"  Features: {scaler.n_features_in_}")
    print("="*60)
    print("\nNext: Restart dashboard to use the rebuilt scaler")
    print("="*60)

if __name__ == "__main__":
    main()











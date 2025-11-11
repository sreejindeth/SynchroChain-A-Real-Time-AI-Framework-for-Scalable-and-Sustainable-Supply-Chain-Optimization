"""
Diagnose data leakage in GNN splits
"""
import pandas as pd
import numpy as np

print("="*80)
print("DATA LEAKAGE DIAGNOSIS")
print("="*80)

# Load splits
train = pd.read_csv('data/processed/temporal_splits/train/gnn_nodes.csv')
val = pd.read_csv('data/processed/temporal_splits/val/gnn_nodes.csv')
test = pd.read_csv('data/processed/temporal_splits/test/gnn_nodes.csv')

print(f"\n[1] Split Sizes:")
print(f"    Train: {len(train)} rows, {len(train.node_id.unique())} unique nodes")
print(f"    Val:   {len(val)} rows, {len(val.node_id.unique())} unique nodes")
print(f"    Test:  {len(test)} rows, {len(test.node_id.unique())} unique nodes")

# Check for node ID overlap
train_nodes = set(train.node_id)
val_nodes = set(val.node_id)
test_nodes = set(test.node_id)

overlap_train_val = train_nodes.intersection(val_nodes)
overlap_train_test = train_nodes.intersection(test_nodes)
overlap_val_test = val_nodes.intersection(test_nodes)

print(f"\n[2] Node ID Overlap (DATA LEAKAGE CHECK):")
print(f"    Train-Val overlap:  {len(overlap_train_val)} nodes")
print(f"    Train-Test overlap: {len(overlap_train_test)} nodes")
print(f"    Val-Test overlap:   {len(overlap_val_test)} nodes")

if len(overlap_train_val) > 0 or len(overlap_train_test) > 0:
    print(f"\n    🚨 CRITICAL: DATA LEAKAGE DETECTED!")
    print(f"       Same nodes appear in multiple splits!")
    print(f"       Examples: {list(overlap_train_test)[:10]}")

# Check delay_risk distribution
print(f"\n[3] Delay Risk Distribution:")
print(f"    Train:")
print(f"      Mean:   {train.delay_risk.mean():.4f}")
print(f"      Std:    {train.delay_risk.std():.4f}")
print(f"      Min:    {train.delay_risk.min():.4f}")
print(f"      Max:    {train.delay_risk.max():.4f}")
print(f"      Unique: {len(train.delay_risk.unique())} values")

print(f"\n    Val:")
print(f"      Mean:   {val.delay_risk.mean():.4f}")
print(f"      Std:    {val.delay_risk.std():.4f}")
print(f"      Min:    {val.delay_risk.min():.4f}")
print(f"      Max:    {val.delay_risk.max():.4f}")
print(f"      Unique: {len(val.delay_risk.unique())} values")

print(f"\n    Test:")
print(f"      Mean:   {test.delay_risk.mean():.4f}")
print(f"      Std:    {test.delay_risk.std():.4f}")
print(f"      Min:    {test.delay_risk.min():.4f}")
print(f"      Max:    {test.delay_risk.max():.4f}")
print(f"      Unique: {len(test.delay_risk.unique())} values")

# Check if delay_risk is constant
if train.delay_risk.std() < 0.01:
    print(f"\n    🚨 CRITICAL: Delay risk is nearly constant!")
    print(f"       This explains the near-zero validation loss.")

# Check value counts
print(f"\n[4] Most Common Delay Risk Values:")
print(f"    Train top 5:")
print(train.delay_risk.value_counts().head())
print(f"\n    Val top 5:")
print(val.delay_risk.value_counts().head())
print(f"\n    Test top 5:")
print(test.delay_risk.value_counts().head())

# Check feature variance
print(f"\n[5] Feature Statistics (Train):")
for col in ['Product Price', 'inventory_level', 'carbon_cost', 'supplier_reliability']:
    if col in train.columns:
        print(f"    {col:30s}: mean={train[col].mean():8.4f}, std={train[col].std():8.4f}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)







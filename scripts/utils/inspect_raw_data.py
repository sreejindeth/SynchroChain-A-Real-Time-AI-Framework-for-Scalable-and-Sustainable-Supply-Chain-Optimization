"""
Inspect raw data to understand proper target variable
"""
import pandas as pd
import numpy as np

print("="*80)
print("RAW DATA INSPECTION")
print("="*80)

# Load raw data (try different encodings)
try:
    df = pd.read_csv('data/raw/DataCoSupplyChainDataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/raw/DataCoSupplyChainDataset.csv', encoding='latin-1')

print(f"\n[1] Raw Data Shape: {df.shape}")
print(f"    {df.shape[0]} rows, {df.shape[1]} columns")

print(f"\n[2] Columns ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:2d}. {col}")

# Find delay/risk related columns
print(f"\n[3] Delay/Risk Related Columns:")
risk_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['delay', 'risk', 'late', 'delivery'])]
print(f"    Found: {risk_cols}")

if risk_cols:
    print(f"\n[4] Distribution of Risk/Delay Columns:")
    for col in risk_cols:
        print(f"\n    {col}:")
        print(f"      Data type: {df[col].dtype}")
        print(f"      Unique values: {df[col].nunique()}")
        print(f"      Missing: {df[col].isna().sum()}")
        
        if df[col].dtype in ['int64', 'float64']:
            print(f"      Mean: {df[col].mean():.4f}")
            print(f"      Std:  {df[col].std():.4f}")
            print(f"      Min:  {df[col].min():.4f}")
            print(f"      Max:  {df[col].max():.4f}")
        
        print(f"      Value counts (top 10):")
        print(df[col].value_counts().head(10))

# Check for temporal columns
print(f"\n[5] Temporal Columns:")
temporal_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['date', 'time', 'day', 'month', 'year', 'order'])]
print(f"    Found: {temporal_cols[:5]}")  # Show first 5

print("\n" + "="*80)


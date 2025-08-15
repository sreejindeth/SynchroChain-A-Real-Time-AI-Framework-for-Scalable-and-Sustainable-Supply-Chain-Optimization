# src/utils/check_logs.py
import pandas as pd

df = pd.read_csv("../../data/preprocessed/cleaned_access_logs.csv")
print("✅ Loaded cleaned access logs")
print(f"📊 Shape: {df.shape}")
print(f"🔍 Actions: {df['action'].unique()}")
print(f"🧩 Columns: {list(df.columns)}")
print("\n📅 Sample:\n", df.head())
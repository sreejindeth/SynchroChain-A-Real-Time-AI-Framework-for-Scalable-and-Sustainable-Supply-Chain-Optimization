# inspect_logs.py
import pandas as pd

data_file_path = 'data/raw/tokenized_access_logs.csv'

try:
    df = pd.read_csv(data_file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 failed, trying 'ISO-8859-1'")
    df = pd.read_csv(data_file_path, encoding='ISO-8859-1')

print("--- Data Shape ---")
print(df.shape)

print("\n--- Column Names ---")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\n--- First Few Rows ---")
print(df.head(2))

print("\n--- Unique Values in Key Columns ---")
print("Product:", df['Product'].unique()[:10])
print("Category:", df['Category'].unique()[:10])
print("Department:", df['Department'].unique()[:10])
print("Sample IPs:", df['ip'].unique()[:10])

print("\n--- URL Patterns ---")
print(df['url'].head(10))

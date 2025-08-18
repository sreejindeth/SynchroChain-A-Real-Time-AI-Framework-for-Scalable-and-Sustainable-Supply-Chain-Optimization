# inspect_data.py
import pandas as pd

# Load the raw data
data_file_path = 'data/raw/DataCoSupplyChainDataset.csv'
# The file likely uses a specific separator (e.g., comma or semicolon)
# Let's try comma first, then semicolon if needed.
try:
    df = pd.read_csv(data_file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 failed, trying 'ISO-8859-1'")
    df = pd.read_csv(data_file_path, encoding='ISO-8859-1')

print("--- Data Shape ---")
print(df.shape)

print("\n--- Column Names ---")
# Print columns in a readable format
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\n--- First Few Rows ---")
print(df.head(2))

print("\n--- Unique Values in Key Columns ---")
print("Type:", df['Type'].unique())
print("Delivery Status:", df['Delivery Status'].unique())
print("Category Name:", df['Category Name'].unique()[:10]) # Show first 10
print("Customer Country:", df['Customer Country'].unique()[:10]) # Show first 10

print("\n--- Checking for Access Log Data ---")
# The access log file seems empty, but perhaps access-like data is derivable.
# Let's see if we can find unique customer-product interactions that could
# represent 'views' or 'interest events'.
# We'll use Customer Id, Product Card Id, and Order Date for this.
# Let's see if Order Item Id groups multiple items per order.
print("Sample Order Item Ids:")
print(df[['Order Item Id', 'Customer Id', 'Product Card Id', 'order date (DateOrders)']].head(10))

# src/utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_supply_chain_data(input_path, output_path):
    """
    Clean and preprocess the main supply chain dataset
    """
    print("🔍 Loading supply chain data...")
    df = pd.read_csv(input_path, on_bad_lines='skip',encoding='latin-1')

    print(f"📊 Initial shape: {df.shape}")
    print(f"🧩 Columns: {list(df.columns)}")

    # --- 1. Drop Sensitive/Useless Columns ---
    cols_to_drop = ['Customer Email', 'Customer Password', 'Product Image', 'Product Description']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # --- 2. Fix Country & Market Names ---
    country_map = {
        'EE. UU.': 'USA',
        'Francia': 'France',
        'Alemania': 'Germany',
        'Italia': 'Italy',
        'Egipto': 'Egypt',
        'Pakistán': 'Pakistan',
        'Filipinas': 'Philippines',
        'Brasil': 'Brazil',
        'México': 'Mexico',
        'Puerto Rico': 'USA',
        'Estados Unidos': 'USA',
        'Reino Unido': 'UK',
        'Irán': 'Iran',
        'Turquía': 'Turkey',
        'Camboya': 'Cambodia',
        'Irak': 'Iraq',
        'Cina': 'China'
    }
    df['Order Country'] = df['Order Country'].replace(country_map)
    df['Customer Country'] = df['Customer Country'].replace(country_map)

    # Fix Market encoding (e.g., "Alsacia-Champaña")
    df['Market'] = df['Market'].str.replace('', 'a').str.replace('', 'i').str.replace('', 'u')

    # --- 3. Parse Dates ---
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    df['Shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')

    df['order_hour'] = df['order date (DateOrders)'].dt.hour
    df['order_dayofweek'] = df['order date (DateOrders)'].dt.dayofweek
    df['order_month'] = df['order date (DateOrders)'].dt.month

    # --- 4. Encode Categorical Variables ---
    le = LabelEncoder()
    cat_cols = [
        'Customer Segment', 'Market', 'Order Status', 'Delivery Status',
        'Shipping Mode', 'Category Name', 'Type', 'Department Name'
    ]
    for col in cat_cols:
        if col in df.columns:
            # Fill NaN before encoding
            df[col] = df[col].astype(str).fillna("Unknown")
            df[f'{col}_encoded'] = le.fit_transform(df[col])

    # --- 5. Handle Missing & Outliers in Numerical Columns ---
    num_cols = ['Benefit per order', 'Sales', 'Order Item Profit Ratio', 'Sales per customer']
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            low, high = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=low, upper=high)

    # --- 6. Compute City-Level Risk ---
    risk_cols = ['Order City', 'Order State']
    if all(col in df.columns for col in risk_cols):
        city_risk = df.groupby(risk_cols)['Late_delivery_risk'].mean().reset_index()
        city_risk.columns = ['Order City', 'Order State', 'city_late_risk']
        df = df.merge(city_risk, on=['Order City', 'Order State'], how='left')
    else:
        df['city_late_risk'] = df['Late_delivery_risk']  # fallback

    # --- 7. Distance to Miami (Example for routing)
    MIAMI_LAT, MIAMI_LON = 25.77437019, -80.19377899

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['distance_to_miami'] = df.apply(
            lambda row: haversine(row['Latitude'], row['Longitude'], MIAMI_LAT, MIAMI_LON),
            axis=1
        )
    else:
        df['distance_to_miami'] = np.nan

    # --- 8. Save Cleaned Data ---
    df.to_csv(output_path, index=False)
    print(f"✅ Supply chain data cleaned and saved to {output_path}")
    return df


def clean_access_logs(input_path, output_path):
    print("🔍 Loading access logs...")
    try:
        columns = [
            'product_name', 'category', 'timestamp_str', 'month', 'hour',
            'department', 'ip', 'action_url'
        ]
        df = pd.read_csv(input_path, names=columns, delimiter=',', on_bad_lines='skip', dtype=str)

        # Clean strings
        for col in ['product_name', 'category', 'department', 'action_url']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp_str'], format='%m/%d/%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # ✅ Improved action extraction
        def extract_action(url):
            url = str(url).lower()
            if 'add_to_cart' in url or '/add_to_cart' in url:
                return 'add_to_cart'
            elif '/checkout' in url:
                return 'checkout'
            elif '/cart' in url:
                return 'view_cart'
            else:
                return 'view'  # Default for product page views

        df['action'] = df['action_url'].apply(extract_action)

        # Keep all actions (don't filter out 'view')
        # No df = df[df['action'].isin(...)] filter

        # Clean product name
        df['product_name'] = df['product_name'].apply(
            lambda x: x.split(',')[0].strip() if ',' in str(x) else str(x).strip()
        )

        # Save
        df.to_csv(output_path, index=False)
        print(f"✅ Access logs cleaned → {len(df)} events")
        print(f"🔍 Action distribution:\n{df['action'].value_counts()}")
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()


# --- Main Execution ---
if __name__ == "__main__":
    # Get the project root (two levels up from this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # ../..
    
    raw_data_dir = os.path.join(project_root, "data", "raw")
    preprocessed_dir = os.path.join(project_root, "data", "preprocessed")

    # Debug: Print paths
    print(f"📄 Project Root: {project_root}")
    print(f"🔍 Raw Data Dir: {raw_data_dir}")
    
    input_path = os.path.join(raw_data_dir, "DataCoSupplyChainDataset.csv")
    print(f"🔍 Looking for file at: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ File not found! Please check:")
        print(f"   - Is the file named exactly 'DataCoSupplyChainDataset.csv'?")
        print(f"   - Is it in '{raw_data_dir}'?")
        print(f"   - Did you unzip it if it was compressed?")
    else:
        # Create output directory
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Clean supply chain data
        clean_supply_chain_data(
            input_path,
            os.path.join(preprocessed_dir, "cleaned_supply_chain.csv")
        )

        # Clean access logs
        logs_input = os.path.join(raw_data_dir, "tokenized_access_logs.csv")
        if os.path.exists(logs_input):
            clean_access_logs(
                logs_input,
                os.path.join(preprocessed_dir, "cleaned_access_logs.csv")
            )
        else:
            print(f"⚠️  Access logs not found at {logs_input}")

    print("🎉 All preprocessing complete! Ready for modeling.")
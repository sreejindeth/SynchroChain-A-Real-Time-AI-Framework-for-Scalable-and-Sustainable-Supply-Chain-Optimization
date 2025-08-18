# src/preprocessing/preprocess_data.py
"""
Phase 1: Preprocessing Script for SynchroChain
Transforms raw DataCo data into structured inputs for AI components.
- Handles encoding issues robustly.
- Ensures correct naming for 'product_name'.
- Produces temporal splits for train/validation/test.
- Includes feature engineering for historical delay rates.
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np

def load_data(file_path, is_access_log=False):
    """Loads the raw CSV, handling encoding issues robustly."""
    try:
        if is_access_log:
            # --- CORRECTION: Handle access logs correctly ---
            # The provided tokenized_access_logs.csv seems to be the one WITH data.
            # The Pasted_Text_1755518213784.txt was empty.
            # So, we load it directly, expecting data.
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path} using default encoding.")
        else:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Loaded {file_path} using UTF-8 encoding.")
    except (UnicodeDecodeError, pd.errors.ParserError):
        # If first attempt fails, fallback to ISO-8859-1
        print(f"UTF-8 failed for {file_path}, trying 'ISO-8859-1'")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df

# src/preprocessing/preprocess_data.py (Inside the preprocess_access_logs function)

def preprocess_access_logs(df_access_raw):
    """
    Cleans and structures the tokenized_access_logs.csv data.
    - Renames columns as needed
    - Ensures session_id exists, generating it if missing
    - Handles feature derivation for action and navigation_depth
    """
    print("Preprocessing access logs...")
    df_access = df_access_raw.copy()
    print(f"  Initial access log columns: {list(df_access.columns)}")

    # Rename for consistency
    for col_map in [('Product', 'product_name'), ('Date', 'timestamp'), ('ip', 'user_id')]:
        src, tgt = col_map
        if src in df_access.columns and tgt not in df_access.columns:
            df_access.rename(columns={src: tgt}, inplace=True)
            print(f"  Renamed '{src}' to '{tgt}'.")

    print(f"  Available columns: {df_access.columns.tolist()}")

    # Generate session_id if missing
    if 'session_id' not in df_access.columns:
        print("  'session_id' column not found. Generating dummy session_id per row.")
        df_access['session_id'] = df_access.index  # Each row gets a unique session_id

    # Parse timestamps
    df_access['timestamp'] = pd.to_datetime(df_access['timestamp'], errors='coerce')
    print(f"  Parsed timestamps. Invalid dates: {df_access['timestamp'].isna().sum()}")

    # Derive 'action' if missing
    if 'action' not in df_access.columns:
        if 'url' in df_access.columns:
            print("  'action' not found. Deriving from 'url'.")
            df_access['action'] = df_access['url'].apply(lambda x: 'add_to_cart' if 'add_to_cart' in str(x) else 'view')
        else:
            print("  'action' and 'url' not found. Defaulting to 'view'.")
            df_access['action'] = 'view'

    # Derive navigation_depth if missing
    if 'navigation_depth' not in df_access.columns:
        print("  'navigation_depth' not found. Deriving as count per session_id.")
        session_counts = df_access.groupby('session_id').size()
        df_access = df_access.merge(
            session_counts.rename('navigation_depth'),
            left_on='session_id',
            right_index=True,
            how='left'
        )
        df_access['navigation_depth'].fillna(0, inplace=True)
    else:
        df_access['navigation_depth'] = pd.to_numeric(df_access['navigation_depth'], errors='coerce').fillna(0)

    # Ensure all required columns exist before selection
    required_columns = ['session_id', 'user_id', 'product_name', 'action', 'timestamp', 'navigation_depth']
    if 'Product Card Id' in df_access.columns:
        required_columns.append('Product Card Id')

    missing_cols = set(required_columns) - set(df_access.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in access logs preprocessing: {missing_cols}")

    df_processed_access = df_access[required_columns].copy()
    df_processed_access.sort_values(by=['user_id', 'timestamp'], inplace=True)
    df_processed_access.reset_index(drop=True, inplace=True)

    print(f"  Final access log shape: {df_processed_access.shape}")
    print(f"  Sample of processed access logs:\n{df_processed_access.head()}")
    return df_processed_access


def preprocess_gnn_data(df_orders_raw, df_access_processed, start_date=None, end_date=None):
    """
    Extracts nodes and edges for the GNN using both datasets, filtered by date.
    Nodes: Customer (IP), Product, Warehouse (Order Region).
    Edges: Customer -> Product (from access logs), Product -> Warehouse (from orders).
    Features: Derived from transactional data within the date range, including engineered features.
    """
    print("Extracting GNN data from orders and linking to access logs...")
    df_orders = df_orders_raw.copy()

    # --- Apply Temporal Filter ---
    # Ensure 'order date (DateOrders)' is datetime
    df_orders['order date (DateOrders)'] = pd.to_datetime(df_orders['order date (DateOrders)'], errors='coerce')
    
    if start_date is not None:
        df_orders = df_orders[df_orders['order date (DateOrders)'] >= start_date]
    if end_date is not None:
        df_orders = df_orders[df_orders['order date (DateOrders)'] <= end_date]
    print(f"  Filtered orders to date range: {start_date} to {end_date}. Remaining orders: {len(df_orders)}")
    # --- END Temporal Filter ---

    # --- Link Access Logs to Order Data ---
    # Note: df_access_processed should also ideally be filtered by the same date range
    # if it contained timestamps. However, the sample doesn't show a direct timestamp
    # for access logs linked to order dates. We'll proceed with the current logic,
    # assuming df_access_processed is relevant for the time period of df_orders.
    # A more precise method would filter df_access_processed based on interaction
    # timestamps falling within the order date range, if such timestamps existed.
    
    access_products = df_access_processed[['product_name']].drop_duplicates()
    order_product_mapping = df_orders[['Product Name', 'Product Card Id']].drop_duplicates()
    order_product_mapping.rename(columns={'Product Name': 'product_name'}, inplace=True)
    
    df_access_with_id = df_access_processed.merge(order_product_mapping, on='product_name', how='left')
    missing_ids = df_access_with_id['Product Card Id'].isna().sum()
    print(f"  Access log entries without matching Product Card Id: {missing_ids} (out of {len(df_access_with_id)})")
    df_access_valid = df_access_with_id.dropna(subset=['Product Card Id']).copy()
    df_access_valid['Product Card Id'] = df_access_valid['Product Card Id'].astype(int)

    # --- Nodes ---
    product_ids_in_access = df_access_valid['Product Card Id'].unique()
    # Ensure products considered are only those relevant to the filtered orders
    product_agg = df_orders[df_orders['Product Card Id'].isin(product_ids_in_access)].groupby('Product Card Id').agg({
        'Product Price': 'mean',
        'Category Name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).reset_index()
    product_agg.rename(columns={'Product Card Id': 'node_id'}, inplace=True)
    product_agg['node_type'] = 'product'

    customer_agg = df_access_valid[['user_id']].drop_duplicates()
    customer_agg.rename(columns={'user_id': 'node_id'}, inplace=True)
    customer_agg['node_type'] = 'customer'
    customer_agg['location'] = 'Unknown' # Placeholder

    warehouse_agg = df_orders[['Order Region']].drop_duplicates()
    warehouse_agg.rename(columns={'Order Region': 'node_id'}, inplace=True)
    warehouse_agg['node_type'] = 'warehouse_proxy'

    nodes_list = [
        product_agg[['node_id', 'node_type', 'Product Price', 'Category Name']],
        customer_agg[['node_id', 'node_type']],
        warehouse_agg[['node_id', 'node_type']]
    ]
    gnn_nodes = pd.concat(nodes_list, ignore_index=True, sort=False)
    
    # --- Edges ---
    # Filter customer-product edges based on the valid access logs (which are based on filtered orders)
    edges_customer_product = df_access_valid[['user_id', 'Product Card Id']].drop_duplicates()
    edges_customer_product.columns = ['source_node_id', 'target_node_id']
    edges_customer_product['edge_type'] = 'customer_interacted_with_product'

    # Filter product-warehouse edges based on the filtered orders and valid nodes
    valid_product_node_ids = set(product_agg['node_id'])
    valid_warehouse_node_ids = set(warehouse_agg['node_id'])
    edges_product_warehouse = df_orders[
        df_orders['Product Card Id'].isin(valid_product_node_ids) &
        df_orders['Order Region'].isin(valid_warehouse_node_ids)
    ][['Product Card Id', 'Order Region']].drop_duplicates()
    edges_product_warehouse.columns = ['source_node_id', 'target_node_id']
    edges_product_warehouse['edge_type'] = 'product_shipped_from_region'

    # Combine all edges
    gnn_edges = pd.concat([edges_customer_product, edges_product_warehouse], ignore_index=True, sort=False)

    # --- Node Features (Derived from Filtered Order Data) ---
    print("Simulating GNN node features from order data...")
    # Recalculate features based only on the filtered df_orders
    
    # --- FEATURE ENGINEERING: Historical Delay Rates ---
    print("Calculating historical delay rates for features...")

    # Ensure 'is_late' column exists for calculations
    if 'is_late' not in df_orders.columns:
        df_orders['is_late'] = df_orders['Delivery Status'].str.contains('Late', case=False).fillna(False)
        print("  Created 'is_late' column from 'Delivery Status'.")

    # Sort df_orders by date to ensure correct historical calculation
    df_orders_sorted = df_orders.sort_values(by='order date (DateOrders)').reset_index(drop=True)
    print("  Sorted orders by date for historical calculations.")

    # Function to calculate historical delay rate up to a given date for a specific group
    def calc_delay_rate(group):
        # group is a DataFrame sorted by date for a specific product or region
        group = group.copy()
        group['cum_late'] = group['is_late'].cumsum()
        group['cum_count'] = range(1, len(group) + 1)
        # Delay rate = (late orders so far) / (total orders so far)
        # Use shift(1) to get the rate *before* the current order is known/processed
        group['historical_delay_rate'] = group['cum_late'].shift(1) / group['cum_count'].shift(1)
        # Fill NaN for the first order (no history) with 0 or a default (using 0 for simplicity)
        group['historical_delay_rate'].fillna(0, inplace=True)
        return group

    # --- 1. Product Historical Delay Rate ---
    try:
        df_product_hist_delay = df_orders_sorted.groupby('Product Card Id', group_keys=False).apply(calc_delay_rate)
        # Get the last known historical delay rate for each product within this time window
        product_delay_rates = df_product_hist_delay.groupby('Product Card Id')['historical_delay_rate'].last().reset_index()
        product_delay_rates.rename(columns={'historical_delay_rate': 'product_historical_delay_rate'}, inplace=True)
        print(f"  Calculated product historical delay rates. Entries: {len(product_delay_rates)}")
    except Exception as e:
        print(f"  Warning: Failed to calculate product_historical_delay_rate: {e}")
        # Create an empty DataFrame with the correct columns if calculation fails
        product_delay_rates = pd.DataFrame(columns=['Product Card Id', 'product_historical_delay_rate'])
        
    # --- 2. Region Historical Delay Rate ---
    try:
        df_region_hist_delay = df_orders_sorted.groupby('Order Region', group_keys=False).apply(calc_delay_rate)
        # Get the last known historical delay rate for each region within this time window
        region_delay_rates = df_region_hist_delay.groupby('Order Region')['historical_delay_rate'].last().reset_index()
        region_delay_rates.rename(columns={'historical_delay_rate': 'region_historical_delay_rate'}, inplace=True)
        print(f"  Calculated region historical delay rates. Entries: {len(region_delay_rates)}")
    except Exception as e:
        print(f"  Warning: Failed to calculate region_historical_delay_rate: {e}")
        # Create an empty DataFrame with the correct columns if calculation fails
        region_delay_rates = pd.DataFrame(columns=['Order Region', 'region_historical_delay_rate'])

    # --- END FEATURE ENGINEERING ---
    
    
    # --- Original Features (from previous preprocessing) ---
    inv_proxy = df_orders.groupby(['Order Region', 'Product Card Id']).size().reset_index(name='order_count')
    if not inv_proxy.empty:
        max_count = inv_proxy['order_count'].max()
        inv_proxy['inventory_level'] = 1 - (inv_proxy['order_count'] / max_count)
        warehouse_features = inv_proxy.groupby('Order Region')['inventory_level'].mean().reset_index()
        warehouse_features.rename(columns={'Order Region': 'node_id'}, inplace=True)
    else:
        print("  Warning: No order data available for inventory calculation in this time range.")
        warehouse_features = pd.DataFrame(columns=['node_id', 'inventory_level'])

    # df_orders['is_late'] is already calculated above
    delay_risk = df_orders.groupby('Order Region')['is_late'].mean().reset_index()
    delay_risk.rename(columns={'Order Region': 'node_id', 'is_late': 'delay_risk'}, inplace=True)
    
    # Handle case where delay_risk might be empty
    if delay_risk.empty:
         print("  Warning: No order data available for delay risk calculation in this time range.")
         delay_risk = pd.DataFrame(columns=['node_id', 'delay_risk'])

    shipping_costs = {'Same Day': 1.5, 'First Class': 1.2, 'Second Class': 0.8, 'Standard Class': 0.5}
    df_orders['carbon_cost'] = df_orders['Shipping Mode'].map(shipping_costs).fillna(0.6)
    carbon_cost = df_orders.groupby('Order Region')['carbon_cost'].mean().reset_index()
    carbon_cost.rename(columns={'Order Region': 'node_id'}, inplace=True)
    
    # Handle case where carbon_cost might be empty
    if carbon_cost.empty:
         print("  Warning: No order data available for carbon cost calculation in this time range.")
         carbon_cost = pd.DataFrame(columns=['node_id', 'carbon_cost'])

    supplier_reliability = warehouse_agg[['node_id']].copy()
    supplier_reliability['supplier_reliability'] = 0.85
    # --- End Original Features ---

    # --- Merge Features onto Nodes ---
    # Merge original features
    gnn_nodes = gnn_nodes.merge(warehouse_features, on='node_id', how='left')
    gnn_nodes = gnn_nodes.merge(delay_risk, on='node_id', how='left')
    gnn_nodes = gnn_nodes.merge(carbon_cost, on='node_id', how='left')
    gnn_nodes = gnn_nodes.merge(supplier_reliability, on='node_id', how='left')
    
    # --- Merge NEW Engineered Features ---
    # Rename keys in feature DataFrames to 'node_id' for merging
    # Product features: node_id in gnn_nodes corresponds to Product Card Id for product nodes
    if not product_delay_rates.empty:
        product_delay_rates_renamed = product_delay_rates.rename(columns={'Product Card Id': 'node_id'})
        gnn_nodes = gnn_nodes.merge(product_delay_rates_renamed, on='node_id', how='left')
    else:
        gnn_nodes['product_historical_delay_rate'] = np.nan
        
    # Region features: node_id in gnn_nodes corresponds to Order Region for warehouse nodes
    if not region_delay_rates.empty:
        region_delay_rates_renamed = region_delay_rates.rename(columns={'Order Region': 'node_id'})
        gnn_nodes = gnn_nodes.merge(region_delay_rates_renamed, on='node_id', how='left')
    else:
        gnn_nodes['region_historical_delay_rate'] = np.nan
    # --- End Merge NEW Features ---

    # Fill NaNs, including for cases where merge resulted in NaN due to empty feature DFs or no match
    # Use global defaults calculated from the current df_orders slice if possible, or simple defaults.
    default_inventory = 0.5
    default_delay_risk = 0.1 # Could be df_orders['is_late'].mean() if not empty
    default_carbon_cost = 0.5
    default_supplier_reliability = 0.85
    default_product_hist_delay = 0.1 # Or df_orders['is_late'].mean()
    default_region_hist_delay = 0.1  # Or df_orders.groupby('Order Region')['is_late'].mean().mean()

    gnn_nodes.fillna({
        'inventory_level': default_inventory,
        'delay_risk': default_delay_risk,
        'carbon_cost': default_carbon_cost,
        'supplier_reliability': default_supplier_reliability,
        'product_historical_delay_rate': default_product_hist_delay,
        'region_historical_delay_rate': default_region_hist_delay,
        'Product Price': 0.0,
        'Category Name': 'Unknown'
    }, inplace=True)
    print("  Filled NaNs in node features with defaults.")
    # --- End Merge Features ---

    # --- GNN Data Consistency Check ---
    print(f"\n--- GNN Data Consistency Check ---")
    unique_node_ids = set(gnn_nodes['node_id'])
    source_node_ids_in_edges = set(gnn_edges['source_node_id'])
    target_node_ids_in_edges = set(gnn_edges['target_node_id'])
    all_node_ids_in_edges = source_node_ids_in_edges.union(target_node_ids_in_edges)

    unused_nodes = unique_node_ids - all_node_ids_in_edges
    print(f"Nodes defined but not connected by any edge: {len(unused_nodes)}")

    undefined_source_nodes = source_node_ids_in_edges - unique_node_ids
    undefined_target_nodes = target_node_ids_in_edges - unique_node_ids
    print(f"Edges with undefined source nodes: {len(undefined_source_nodes)}")
    print(f"Edges with undefined target nodes: {len(undefined_target_nodes)}")
    if undefined_source_nodes:
        print(f"Sample undefined source nodes: {list(undefined_source_nodes)[:10]}")
    if undefined_target_nodes:
        print(f"Sample undefined target nodes: {list(undefined_target_nodes)[:10]}")
    print("--- End Consistency Check ---\n")

    print(f"  Created {len(gnn_nodes)} GNN nodes and {len(gnn_edges)} GNN edges for the specified time range.")
    # Return df_access_valid for potential use in intent model or further linking
    return gnn_nodes, gnn_edges, df_access_valid 

def save_processed_data(access_logs_df, gnn_nodes_df, gnn_edges_df, output_dir='data/processed'):
    """Saves the processed dataframes to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    access_logs_df.to_csv(os.path.join(output_dir, 'processed_access_logs.csv'), index=False)
    gnn_nodes_df.to_csv(os.path.join(output_dir, 'gnn_nodes.csv'), index=False)
    gnn_edges_df.to_csv(os.path.join(output_dir, 'gnn_edges.csv'), index=False)
    print(f"Processed data saved to {output_dir}")

def main():
    """Main preprocessing function with temporal splits."""
    print("--- Starting SynchroChain Preprocessing (Phase 1 - Temporal Splits with Features) ---")
    start_time = datetime.now()
    
    # 1. Load Raw Data
    # Adjusted paths based on standard structure
    raw_access_path = 'data/raw/tokenized_access_logs.csv' # This should be the one WITH data now
    raw_orders_path = 'data/raw/DataCoSupplyChainDataset.csv' 
    
    df_raw_access = load_data(raw_access_path, is_access_log=True)
    df_raw_orders = load_data(raw_orders_path, is_access_log=False)
    print(f"Raw access data loaded: {df_raw_access.shape}")
    print(f"Raw orders data loaded: {df_raw_orders.shape}")

    # 2. Define Temporal Splits
    # Inspect date range in df_raw_orders to define splits
    df_raw_orders['order date (DateOrders)'] = pd.to_datetime(df_raw_orders['order date (DateOrders)'], errors='coerce')
    print(f"Order data date range: {df_raw_orders['order date (DateOrders)'].min()} to {df_raw_orders['order date (DateOrders)'].max()}")
    
    # Example Split (Adjust as needed based on data distribution)
    # Let's assume data goes from early 2015 to mid 2018 based on snippets
    splits = {
        'train': (None, '2017-06-30'),
        'val': ('2017-07-01', '2017-09-30'),
        'test': ('2017-10-01', None) # Or a specific end date like '2018-03-31'
    }
    
    output_base_dir = 'data/processed/temporal_splits'
    os.makedirs(output_base_dir, exist_ok=True)

    # 3. Process Data for Each Split
    # Note: For GNN, we filter orders. Access logs are not directly timestamped with order dates.
    # We will process GNN data for each time window based on orders.
    # The access logs used will be those that could be linked to products relevant
    # in that time window (via df_access_with_id logic inside preprocess_gnn_data).
    # This is a simplification. A more advanced approach would filter access logs
    # based on user activity dates if such dates were available and linked.

    for split_name, (start_date, end_date) in splits.items():
        print(f"\n--- Creating {split_name.upper()} data split ---")
        # Convert string dates to datetime for filtering
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        # Preprocess access logs once, before any date filtering on orders
        # This is a simplification; ideally access logs would also be date-filtered.
        df_processed_access = preprocess_access_logs(df_raw_access)
        
        # Preprocess GNN data for this time window
        # df_raw_access is passed as-is; filtering logic is inside preprocess_gnn_data
        # based on linking to filtered df_orders.
        df_gnn_nodes, df_gnn_edges, df_linked_access = preprocess_gnn_data(
            df_raw_orders, df_processed_access, start_dt, end_dt
        )
        
        # Save to split-specific directory
        split_output_dir = os.path.join(output_base_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        # Save the access logs that were successfully linked for this period
        save_processed_data(df_linked_access, df_gnn_nodes, df_gnn_edges, output_dir=split_output_dir)
        print(f"--- {split_name.upper()} data saved to {split_output_dir} ---")

    end_time = datetime.now()
    print(f"\n--- All temporal splits preprocessing complete ({end_time - start_time}) ---")

if __name__ == '__main__':
    main()
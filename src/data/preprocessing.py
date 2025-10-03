# src/data/preprocessing.py
"""
Enhanced Data Preprocessing Pipeline for SynchroChain
Handles data cleaning, feature engineering, and temporal splits
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_path = self.data_config['raw_data_path']
        self.processed_path = self.data_config['processed_data_path']
        self.splits_path = self.data_config['temporal_splits_path']
        
        # Create directories if they don't exist
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.splits_path, exist_ok=True)
    
    def load_data(self, file_path: str, is_access_log: bool = False) -> pd.DataFrame:
        """Load CSV data with robust encoding handling."""
        try:
            if is_access_log:
                df = pd.read_csv(file_path)
                print(f"Loaded {file_path} using default encoding.")
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
                print(f"Loaded {file_path} using UTF-8 encoding.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            print(f"UTF-8 failed for {file_path}, trying 'ISO-8859-1'")
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    
    def preprocess_access_logs(self, df_access_raw: pd.DataFrame, 
                             df_orders_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and structure access logs data."""
        print("Preprocessing access logs...")
        df_access = df_access_raw.copy()
        
        # Standardize column names
        column_renames = {
            'Product': 'product_name',
            'Date': 'timestamp',
            'ip': 'user_id'
        }
        
        cols_to_rename = {k: v for k, v in column_renames.items() 
                         if k in df_access.columns and v not in df_access.columns}
        if cols_to_rename:
            df_access.rename(columns=cols_to_rename, inplace=True)
            print(f"Renamed columns: {cols_to_rename}")
        
        # Ensure required columns exist
        required_columns = ['product_name', 'user_id', 'timestamp']
        missing_required = [col for col in required_columns if col not in df_access.columns]
        if missing_required:
            raise KeyError(f"Missing required columns: {missing_required}")
        
        # Parse timestamps
        df_access['timestamp'] = pd.to_datetime(df_access['timestamp'], errors='coerce')
        invalid_timestamps = df_access['timestamp'].isna().sum()
        print(f"Invalid timestamps: {invalid_timestamps}")
        
        # Derive action column if missing
        if 'action' not in df_access.columns:
            if 'url' in df_access.columns:
                df_access['action'] = df_access['url'].apply(
                    lambda x: 'add_to_cart' if 'add_to_cart' in str(x) else 'view'
                )
            else:
                df_access['action'] = 'view'
        
        # Generate session_id if missing
        if 'session_id' not in df_access.columns:
            df_access['temp_session_key'] = (
                df_access['user_id'].astype(str) + "_" +
                df_access['timestamp'].dt.strftime('%Y-%m-%d_%H')
            )
            df_access['session_id'], _ = pd.factorize(df_access['temp_session_key'])
            df_access.drop(columns=['temp_session_key'], inplace=True)
        
        # Generate navigation_depth
        if 'navigation_depth' not in df_access.columns:
            session_counts = df_access.groupby('session_id').size()
            df_access = df_access.merge(
                session_counts.rename('navigation_depth'),
                left_on='session_id',
                right_index=True,
                how='left'
            )
        
        # Add Category Name from orders data
        if df_orders_raw is not None:
            try:
                product_category_map = df_orders_raw[['Product Name', 'Category Name']].drop_duplicates()
                product_category_map.rename(columns={'Product Name': 'product_name'}, inplace=True)
                df_access = df_access.merge(product_category_map, on='product_name', how='left')
                print(f"Added Category Name to access logs")
            except Exception as e:
                print(f"Warning: Could not add Category Name: {e}")
        
        # Sort and clean
        df_access.sort_values(by=['user_id', 'timestamp'], inplace=True)
        df_access.reset_index(drop=True, inplace=True)
        
        print(f"Final access log shape: {df_access.shape}")
        return df_access
    
    def preprocess_gnn_data(self, df_orders_raw: pd.DataFrame, 
                           df_access_processed: pd.DataFrame,
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract nodes and edges for GNN."""
        print("Extracting GNN data...")
        df_orders = df_orders_raw.copy()
        
        # Apply temporal filter
        df_orders['order date (DateOrders)'] = pd.to_datetime(df_orders['order date (DateOrders)'], errors='coerce')
        
        if start_date is not None:
            df_orders = df_orders[df_orders['order date (DateOrders)'] >= start_date]
        if end_date is not None:
            df_orders = df_orders[df_orders['order date (DateOrders)'] <= end_date]
        
        # Link access logs to orders
        order_product_mapping = df_orders[['Product Name', 'Product Card Id']].drop_duplicates()
        order_product_mapping.rename(columns={'Product Name': 'product_name'}, inplace=True)
        
        df_access_with_id = df_access_processed.merge(order_product_mapping, on='product_name', how='left')
        df_access_valid = df_access_with_id.dropna(subset=['Product Card Id']).copy()
        df_access_valid['Product Card Id'] = df_access_valid['Product Card Id'].astype(int)
        
        # Create nodes
        product_ids_in_access = df_access_valid['Product Card Id'].unique()
        product_agg = df_orders[df_orders['Product Card Id'].isin(product_ids_in_access)].groupby('Product Card Id').agg({
            'Product Price': 'mean',
            'Category Name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        product_agg.rename(columns={'Product Card Id': 'node_id'}, inplace=True)
        product_agg['node_type'] = 'product'
        
        customer_agg = df_access_valid[['user_id']].drop_duplicates()
        customer_agg.rename(columns={'user_id': 'node_id'}, inplace=True)
        customer_agg['node_type'] = 'customer'
        
        warehouse_agg = df_orders[['Order Region']].drop_duplicates()
        warehouse_agg.rename(columns={'Order Region': 'node_id'}, inplace=True)
        warehouse_agg['node_type'] = 'warehouse_proxy'
        
        # Combine nodes
        nodes_list = [
            product_agg[['node_id', 'node_type', 'Product Price', 'Category Name']],
            customer_agg[['node_id', 'node_type']],
            warehouse_agg[['node_id', 'node_type']]
        ]
        gnn_nodes = pd.concat(nodes_list, ignore_index=True, sort=False)
        
        # Create edges
        edges_customer_product = df_access_valid[['user_id', 'Product Card Id']].drop_duplicates()
        edges_customer_product.columns = ['source_node_id', 'target_node_id']
        edges_customer_product['edge_type'] = 'customer_interacted_with_product'
        
        valid_product_node_ids = set(product_agg['node_id'])
        valid_warehouse_node_ids = set(warehouse_agg['node_id'])
        edges_product_warehouse = df_orders[
            df_orders['Product Card Id'].isin(valid_product_node_ids) &
            df_orders['Order Region'].isin(valid_warehouse_node_ids)
        ][['Product Card Id', 'Order Region']].drop_duplicates()
        edges_product_warehouse.columns = ['source_node_id', 'target_node_id']
        edges_product_warehouse['edge_type'] = 'product_shipped_from_region'
        
        gnn_edges = pd.concat([edges_customer_product, edges_product_warehouse], ignore_index=True, sort=False)
        
        # Add features
        gnn_nodes = self._add_node_features(gnn_nodes, df_orders)
        
        print(f"Created {len(gnn_nodes)} nodes and {len(gnn_edges)} edges")
        return gnn_nodes, gnn_edges, df_access_valid
    
    def _add_node_features(self, gnn_nodes: pd.DataFrame, df_orders: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to nodes."""
        print("Adding node features...")
        
        # Calculate historical delay rates
        df_orders['is_late'] = df_orders['Delivery Status'].str.contains('Late', case=False).fillna(False)
        df_orders_sorted = df_orders.sort_values(by='order date (DateOrders)').reset_index(drop=True)
        
        # Product historical delay rate
        def calc_delay_rate(group):
            group = group.copy()
            group['cum_late'] = group['is_late'].cumsum()
            group['cum_count'] = range(1, len(group) + 1)
            group['historical_delay_rate'] = group['cum_late'].shift(1) / group['cum_count'].shift(1)
            group['historical_delay_rate'] = group['historical_delay_rate'].fillna(0)
            return group
        
        try:
            df_product_hist_delay = df_orders_sorted.groupby('Product Card Id', group_keys=False).apply(calc_delay_rate)
            product_delay_rates = df_product_hist_delay.groupby('Product Card Id')['historical_delay_rate'].last().reset_index()
            product_delay_rates.rename(columns={'historical_delay_rate': 'product_historical_delay_rate'}, inplace=True)
        except Exception as e:
            print(f"Warning: Could not calculate product delay rates: {e}")
            product_delay_rates = pd.DataFrame(columns=['Product Card Id', 'product_historical_delay_rate'])
        
        # Region historical delay rate
        try:
            df_region_hist_delay = df_orders_sorted.groupby('Order Region', group_keys=False).apply(calc_delay_rate)
            region_delay_rates = df_region_hist_delay.groupby('Order Region')['historical_delay_rate'].last().reset_index()
            region_delay_rates.rename(columns={'historical_delay_rate': 'region_historical_delay_rate'}, inplace=True)
        except Exception as e:
            print(f"Warning: Could not calculate region delay rates: {e}")
            region_delay_rates = pd.DataFrame(columns=['Order Region', 'region_historical_delay_rate'])
        
        # Other features
        inv_proxy = df_orders.groupby(['Order Region', 'Product Card Id']).size().reset_index(name='order_count')
        if not inv_proxy.empty:
            max_count = inv_proxy['order_count'].max()
            inv_proxy['inventory_level'] = 1 - (inv_proxy['order_count'] / max_count)
            warehouse_features = inv_proxy.groupby('Order Region')['inventory_level'].mean().reset_index()
            warehouse_features.rename(columns={'Order Region': 'node_id'}, inplace=True)
        else:
            warehouse_features = pd.DataFrame(columns=['node_id', 'inventory_level'])
        
        delay_risk = df_orders.groupby('Order Region')['is_late'].mean().reset_index()
        delay_risk.rename(columns={'Order Region': 'node_id', 'is_late': 'delay_risk'}, inplace=True)
        
        shipping_costs = {'Same Day': 1.5, 'First Class': 1.2, 'Second Class': 0.8, 'Standard Class': 0.5}
        df_orders['carbon_cost'] = df_orders['Shipping Mode'].map(shipping_costs).fillna(0.6)
        carbon_cost = df_orders.groupby('Order Region')['carbon_cost'].mean().reset_index()
        carbon_cost.rename(columns={'Order Region': 'node_id'}, inplace=True)
        
        supplier_reliability = pd.DataFrame({'node_id': df_orders['Order Region'].unique()})
        supplier_reliability['supplier_reliability'] = 0.85
        
        # Merge features
        gnn_nodes = gnn_nodes.merge(warehouse_features, on='node_id', how='left')
        gnn_nodes = gnn_nodes.merge(delay_risk, on='node_id', how='left')
        gnn_nodes = gnn_nodes.merge(carbon_cost, on='node_id', how='left')
        gnn_nodes = gnn_nodes.merge(supplier_reliability, on='node_id', how='left')
        
        if not product_delay_rates.empty:
            product_delay_rates_renamed = product_delay_rates.rename(columns={'Product Card Id': 'node_id'})
            gnn_nodes = gnn_nodes.merge(product_delay_rates_renamed, on='node_id', how='left')
        else:
            gnn_nodes['product_historical_delay_rate'] = np.nan
        
        if not region_delay_rates.empty:
            region_delay_rates_renamed = region_delay_rates.rename(columns={'Order Region': 'node_id'})
            gnn_nodes = gnn_nodes.merge(region_delay_rates_renamed, on='node_id', how='left')
        else:
            gnn_nodes['region_historical_delay_rate'] = np.nan
        
        # Fill NaNs
        gnn_nodes.fillna({
            'inventory_level': 0.5,
            'delay_risk': 0.1,
            'carbon_cost': 0.5,
            'supplier_reliability': 0.85,
            'product_historical_delay_rate': 0.1,
            'region_historical_delay_rate': 0.1,
            'Product Price': 0.0,
            'Category Name': 'Unknown'
        }, inplace=True)
        
        return gnn_nodes
    
    def create_temporal_splits(self):
        """Create train/validation/test splits."""
        print("Creating temporal splits...")
        
        # Load raw data
        raw_access_path = os.path.join(self.raw_path, 'tokenized_access_logs.csv')
        raw_orders_path = os.path.join(self.raw_path, 'DataCoSupplyChainDataset.csv')
        
        df_raw_access = self.load_data(raw_access_path, is_access_log=True)
        df_raw_orders = self.load_data(raw_orders_path, is_access_log=False)
        
        # Define temporal splits
        splits = {
            'train': (None, '2017-06-30'),
            'val': ('2017-07-01', '2017-09-30'),
            'test': ('2017-10-01', None)
        }
        
        for split_name, (start_date, end_date) in splits.items():
            print(f"\nCreating {split_name.upper()} split...")
            
            # Preprocess access logs
            df_processed_access = self.preprocess_access_logs(df_raw_access, df_raw_orders)
            
            # Preprocess GNN data
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            
            df_gnn_nodes, df_gnn_edges, df_linked_access = self.preprocess_gnn_data(
                df_raw_orders, df_processed_access, start_dt, end_dt
            )
            
            # Save split data
            split_output_dir = os.path.join(self.splits_path, split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            
            df_linked_access.to_csv(os.path.join(split_output_dir, 'processed_access_logs.csv'), index=False)
            df_gnn_nodes.to_csv(os.path.join(split_output_dir, 'gnn_nodes.csv'), index=False)
            df_gnn_edges.to_csv(os.path.join(split_output_dir, 'gnn_edges.csv'), index=False)
            
            print(f"Saved {split_name} data to {split_output_dir}")
    
    def run(self):
        """Run the complete preprocessing pipeline."""
        print("Starting data preprocessing pipeline...")
        start_time = datetime.now()
        
        self.create_temporal_splits()
        
        end_time = datetime.now()
        print(f"Preprocessing complete in {end_time - start_time}")

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.run()
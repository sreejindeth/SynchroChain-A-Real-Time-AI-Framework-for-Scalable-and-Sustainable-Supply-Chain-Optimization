# src/models/gnn/enhanced_graph_builder.py
"""
Enhanced Graph Builder for SynchroChain Supply Chain GNN
Creates comprehensive graph with customers, products, warehouses, and suppliers
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Tuple, Dict, List, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

class EnhancedGraphBuilder:
    """
    Enhanced graph builder for supply chain optimization.
    Creates nodes: customers, products, warehouses, suppliers
    Creates edges: customer→product, product→warehouse, warehouse→supplier
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the graph builder."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_config = self.config['data']
        self.processed_path = self.data_config['processed_data_path']
        
        # Initialize encoders
        self.node_type_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        self.shipping_mode_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        print("[INIT] Enhanced Graph Builder initialized")
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed datasets."""
        print("[LOAD] Loading processed datasets...")
        
        # Load processed access logs (has Product Card Id mapping)
        access_logs_path = os.path.join(self.data_config['processed_data_path'], 'processed_access_logs.csv')
        df_access = pd.read_csv(access_logs_path)
        
        # Load supply chain data
        supply_chain_path = os.path.join(self.data_config['raw_data_path'], 'DataCoSupplyChainDataset.csv')
        df_supply = pd.read_csv(supply_chain_path, encoding='ISO-8859-1')
        
        # Load description data (for supplier info)
        desc_path = os.path.join(self.data_config['raw_data_path'], 'DescriptionDataCoSupplyChain.csv')
        df_desc = pd.read_csv(desc_path, encoding='ISO-8859-1')
        
        print(f"[LOAD] Loaded {len(df_access)} access logs, {len(df_supply)} supply records, {len(df_desc)} descriptions")
        return df_access, df_supply, df_desc
        
    def create_enhanced_nodes(self, df_access: pd.DataFrame, df_supply: pd.DataFrame, df_desc: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced nodes with comprehensive features."""
        print("[NODES] Creating enhanced node structure...")
        
        all_nodes = []
        
        # 1. PRODUCT NODES
        print("[NODES] Creating product nodes...")
        product_features = df_supply.groupby('Product Card Id').agg({
            'Product Price': 'mean',
            'Category Name': 'first',
            'Order Item Quantity': 'mean',
            'Order Item Total': 'mean',
            'Days for shipping (real)': 'mean',
            'Days for shipment (scheduled)': 'mean',
            'Benefit per order': 'mean',
            'Sales per customer': 'mean'
        }).reset_index()
        
        # Add historical delay rate for products
        product_features['product_historical_delay_rate'] = (
            product_features['Days for shipping (real)'] / 
            (product_features['Days for shipment (scheduled)'] + 1e-6)
        ).clip(0, 2)  # Cap at 2x scheduled time
        
        # Add product popularity from access logs
        product_popularity = df_access.groupby('Product Card Id').size().reset_index(name='interaction_count')
        product_features = product_features.merge(product_popularity, on='Product Card Id', how='left')
        product_features['interaction_count'] = product_features['interaction_count'].fillna(0)
        
        # Create product nodes
        product_nodes = pd.DataFrame({
            'node_id': product_features['Product Card Id'],
            'node_type': 'product',
            'product_price': product_features['Product Price'],
            'category_name': product_features['Category Name'],
            'avg_quantity': product_features['Order Item Quantity'],
            'avg_total': product_features['Order Item Total'],
            'historical_delay_rate': product_features['product_historical_delay_rate'],
            'benefit_per_order': product_features['Benefit per order'],
            'sales_per_customer': product_features['Sales per customer'],
            'popularity_score': product_features['interaction_count'],
            'inventory_level': 0.5,  # Default inventory
            'delay_risk': 0.1,       # Default delay risk
            'carbon_cost': 0.5,      # Default carbon cost
            'supplier_reliability': 0.85  # Default reliability
        })
        all_nodes.append(product_nodes)
        
        # 2. CUSTOMER NODES
        print("[NODES] Creating customer nodes...")
        customer_features = df_access.groupby('user_id').agg({
            'Product Card Id': 'count',  # Total interactions
            'action': lambda x: (x == 'purchase').sum() if 'purchase' in x.values else x.count(),  # Purchase count
            'timestamp': ['min', 'max']  # Activity period
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['user_id', 'total_interactions', 'purchase_count', 'first_activity', 'last_activity']
        
        # Calculate customer features
        customer_features['purchase_rate'] = (
            customer_features['purchase_count'] / 
            (customer_features['total_interactions'] + 1e-6)
        )
        customer_features['activity_days'] = (
            pd.to_datetime(customer_features['last_activity']) - 
            pd.to_datetime(customer_features['first_activity'])
        ).dt.days + 1
        
        # Create customer nodes
        customer_nodes = pd.DataFrame({
            'node_id': customer_features['user_id'],
            'node_type': 'customer',
            'total_interactions': customer_features['total_interactions'],
            'purchase_count': customer_features['purchase_count'],
            'purchase_rate': customer_features['purchase_rate'],
            'activity_days': customer_features['activity_days'],
            'customer_value_score': customer_features['purchase_count'] * customer_features['purchase_rate'],
            'loyalty_score': np.minimum(customer_features['activity_days'] / 365, 1.0),  # Normalized to 1 year
            'inventory_level': 0.0,      # Not applicable
            'delay_risk': 0.0,           # Not applicable
            'carbon_cost': 0.0,          # Not applicable
            'supplier_reliability': 0.0   # Not applicable
        })
        all_nodes.append(customer_nodes)
        
        # 3. WAREHOUSE/REGION NODES
        print("[NODES] Creating warehouse nodes...")
        warehouse_features = df_supply.groupby('Order Region').agg({
            'Order Item Quantity': 'sum',  # Total volume
            'Days for shipping (real)': 'mean',  # Avg shipping time
            'Days for shipment (scheduled)': 'mean',  # Scheduled time
            'Shipping Mode': lambda x: x.mode().iloc[0] if not x.empty else 'Standard Class',  # Most common shipping
            'Order Item Total': 'sum',  # Total value
            'Product Card Id': 'nunique'  # Product variety
        }).reset_index()
        
        # Calculate warehouse features
        warehouse_features['region_historical_delay_rate'] = (
            warehouse_features['Days for shipping (real)'] / 
            (warehouse_features['Days for shipment (scheduled)'] + 1e-6)
        ).clip(0, 2)
        
        warehouse_features['utilization_rate'] = np.minimum(
            warehouse_features['Order Item Quantity'] / warehouse_features['Order Item Quantity'].max(), 1.0
        )
        
        # Create warehouse nodes
        warehouse_nodes = pd.DataFrame({
            'node_id': warehouse_features['Order Region'],
            'node_type': 'warehouse',
            'total_volume': warehouse_features['Order Item Quantity'],
            'avg_shipping_time': warehouse_features['Days for shipping (real)'],
            'region_delay_rate': warehouse_features['region_historical_delay_rate'],
            'primary_shipping_mode': warehouse_features['Shipping Mode'],
            'total_value': warehouse_features['Order Item Total'],
            'product_variety': warehouse_features['Product Card Id'],
            'utilization_rate': warehouse_features['utilization_rate'],
            'inventory_level': warehouse_features['utilization_rate'],  # Use utilization as proxy
            'delay_risk': warehouse_features['region_historical_delay_rate'] / 2,  # Normalized
            'carbon_cost': 0.3 + 0.4 * warehouse_features['utilization_rate'],  # Higher utilization = higher carbon
            'supplier_reliability': 0.9 - 0.2 * warehouse_features['region_historical_delay_rate']  # Lower delay = higher reliability
        })
        all_nodes.append(warehouse_nodes)
        
        # 4. SUPPLIER NODES (derived from supply chain data)
        print("[NODES] Creating supplier nodes...")
        # Create synthetic suppliers based on categories and regions
        supplier_data = df_supply.groupby(['Category Name', 'Order Region']).agg({
            'Product Price': 'mean',
            'Days for shipping (real)': 'mean',
            'Order Item Quantity': 'sum',
            'Benefit per order': 'mean'
        }).reset_index()
        
        supplier_data['supplier_id'] = supplier_data['Category Name'] + '_' + supplier_data['Order Region']
        
        supplier_nodes = pd.DataFrame({
            'node_id': supplier_data['supplier_id'],
            'node_type': 'supplier',
            'category_specialization': supplier_data['Category Name'],
            'primary_region': supplier_data['Order Region'],
            'avg_product_price': supplier_data['Product Price'],
            'avg_delivery_time': supplier_data['Days for shipping (real)'],
            'total_supply_volume': supplier_data['Order Item Quantity'],
            'profit_margin': supplier_data['Benefit per order'],
            'inventory_level': 0.7,  # Default high inventory for suppliers
            'delay_risk': np.minimum(supplier_data['Days for shipping (real)'] / 10, 1.0),  # Normalized
            'carbon_cost': 0.2 + 0.3 * (supplier_data['Days for shipping (real)'] / 10),  # Distance proxy
            'supplier_reliability': 0.95 - 0.15 * np.minimum(supplier_data['Days for shipping (real)'] / 10, 1.0)
        })
        all_nodes.append(supplier_nodes)
        
        # Combine all nodes
        all_nodes_df = pd.concat(all_nodes, ignore_index=True, sort=False)
        
        # Fill missing values with appropriate defaults
        numeric_columns = all_nodes_df.select_dtypes(include=[np.number]).columns
        all_nodes_df[numeric_columns] = all_nodes_df[numeric_columns].fillna(0)
        
        categorical_columns = all_nodes_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['node_id', 'node_type']:
                all_nodes_df[col] = all_nodes_df[col].fillna('Unknown')
        
        print(f"[NODES] Created {len(all_nodes_df)} total nodes:")
        print(f"        - Products: {len(product_nodes)}")
        print(f"        - Customers: {len(customer_nodes)}")
        print(f"        - Warehouses: {len(warehouse_nodes)}")
        print(f"        - Suppliers: {len(supplier_nodes)}")
        
        return all_nodes_df
        
    def create_enhanced_edges(self, df_access: pd.DataFrame, df_supply: pd.DataFrame, 
                            nodes_df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced edges with features."""
        print("[EDGES] Creating enhanced edge structure...")
        
        all_edges = []
        
        # Get node mappings
        product_nodes = set(nodes_df[nodes_df['node_type'] == 'product']['node_id'])
        customer_nodes = set(nodes_df[nodes_df['node_type'] == 'customer']['node_id'])
        warehouse_nodes = set(nodes_df[nodes_df['node_type'] == 'warehouse']['node_id'])
        supplier_nodes = set(nodes_df[nodes_df['node_type'] == 'supplier']['node_id'])
        
        # 1. CUSTOMER → PRODUCT EDGES
        print("[EDGES] Creating customer→product edges...")
        customer_product_edges = df_access.groupby(['user_id', 'Product Card Id']).agg({
            'action': 'count',  # Interaction frequency
            'timestamp': ['min', 'max']  # Interaction period
        }).reset_index()
        
        # Flatten columns
        customer_product_edges.columns = ['source_node_id', 'target_node_id', 'interaction_frequency', 'first_interaction', 'last_interaction']
        
        # Filter valid edges
        valid_cp_edges = customer_product_edges[
            customer_product_edges['source_node_id'].isin(customer_nodes) &
            customer_product_edges['target_node_id'].isin(product_nodes)
        ].copy()
        
        valid_cp_edges['edge_type'] = 'customer_product'
        valid_cp_edges['edge_weight'] = np.log1p(valid_cp_edges['interaction_frequency'])  # Log-scaled weight
        valid_cp_edges['recency_score'] = 1.0  # Could be calculated based on timestamp
        valid_cp_edges['shipping_mode'] = 'Standard Class'  # Default
        valid_cp_edges['transit_time'] = 3.0  # Default days
        
        all_edges.append(valid_cp_edges[['source_node_id', 'target_node_id', 'edge_type', 
                                       'interaction_frequency', 'edge_weight', 'recency_score',
                                       'shipping_mode', 'transit_time']])
        
        # 2. PRODUCT → WAREHOUSE EDGES
        print("[EDGES] Creating product→warehouse edges...")
        product_warehouse_edges = df_supply.groupby(['Product Card Id', 'Order Region']).agg({
            'Order Item Quantity': 'sum',  # Total quantity shipped
            'Days for shipping (real)': 'mean',  # Avg shipping time
            'Shipping Mode': lambda x: x.mode().iloc[0] if not x.empty else 'Standard Class',  # Most common mode
            'Order Item Total': 'sum'  # Total value
        }).reset_index()
        
        # Filter valid edges
        valid_pw_edges = product_warehouse_edges[
            product_warehouse_edges['Product Card Id'].isin(product_nodes) &
            product_warehouse_edges['Order Region'].isin(warehouse_nodes)
        ].copy()
        
        valid_pw_edges['source_node_id'] = valid_pw_edges['Product Card Id']
        valid_pw_edges['target_node_id'] = valid_pw_edges['Order Region']
        valid_pw_edges['edge_type'] = 'product_warehouse'
        valid_pw_edges['interaction_frequency'] = valid_pw_edges['Order Item Quantity']
        valid_pw_edges['edge_weight'] = np.log1p(valid_pw_edges['Order Item Total'])
        valid_pw_edges['recency_score'] = 1.0
        valid_pw_edges['shipping_mode'] = valid_pw_edges['Shipping Mode']
        valid_pw_edges['transit_time'] = valid_pw_edges['Days for shipping (real)']
        
        all_edges.append(valid_pw_edges[['source_node_id', 'target_node_id', 'edge_type',
                                       'interaction_frequency', 'edge_weight', 'recency_score',
                                       'shipping_mode', 'transit_time']])
        
        # 3. WAREHOUSE → SUPPLIER EDGES
        print("[EDGES] Creating warehouse→supplier edges...")
        # Create synthetic warehouse→supplier connections based on category and region
        warehouse_supplier_data = []
        
        for warehouse in warehouse_nodes:
            # Find products in this warehouse
            warehouse_products = df_supply[df_supply['Order Region'] == warehouse]['Category Name'].unique()
            
            for category in warehouse_products:
                supplier_id = f"{category}_{warehouse}"
                if supplier_id in supplier_nodes:
                    # Calculate connection strength
                    volume = df_supply[
                        (df_supply['Order Region'] == warehouse) & 
                        (df_supply['Category Name'] == category)
                    ]['Order Item Quantity'].sum()
                    
                    avg_time = df_supply[
                        (df_supply['Order Region'] == warehouse) & 
                        (df_supply['Category Name'] == category)
                    ]['Days for shipping (real)'].mean()
                    
                    warehouse_supplier_data.append({
                        'source_node_id': warehouse,
                        'target_node_id': supplier_id,
                        'edge_type': 'warehouse_supplier',
                        'interaction_frequency': volume,
                        'edge_weight': np.log1p(volume),
                        'recency_score': 1.0,
                        'shipping_mode': 'First Class',  # Supplier to warehouse typically faster
                        'transit_time': avg_time * 0.7  # Assume supplier delivery is faster
                    })
        
        if warehouse_supplier_data:
            warehouse_supplier_edges = pd.DataFrame(warehouse_supplier_data)
            all_edges.append(warehouse_supplier_edges)
        
        # Combine all edges
        all_edges_df = pd.concat(all_edges, ignore_index=True, sort=False)
        
        # Fill missing values
        numeric_columns = all_edges_df.select_dtypes(include=[np.number]).columns
        all_edges_df[numeric_columns] = all_edges_df[numeric_columns].fillna(0)
        
        all_edges_df['shipping_mode'] = all_edges_df['shipping_mode'].fillna('Standard Class')
        
        print(f"[EDGES] Created {len(all_edges_df)} total edges:")
        edge_counts = all_edges_df['edge_type'].value_counts()
        for edge_type, count in edge_counts.items():
            print(f"        - {edge_type}: {count}")
        
        return all_edges_df
        
    def prepare_node_features(self, nodes_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare node features for GNN."""
        print("[FEATURES] Preparing node features...")
        
        # Define feature columns for each node type
        base_features = ['inventory_level', 'delay_risk', 'carbon_cost', 'supplier_reliability']
        
        # Encode categorical features
        nodes_df['node_type_encoded'] = self.node_type_encoder.fit_transform(nodes_df['node_type'])
        
        # Handle category encoding (only for products)
        category_col = 'category_name' if 'category_name' in nodes_df.columns else 'category_specialization'
        if category_col in nodes_df.columns:
            # Fill missing categories
            nodes_df[category_col] = nodes_df[category_col].fillna('Unknown')
            nodes_df['category_encoded'] = self.category_encoder.fit_transform(nodes_df[category_col])
        else:
            nodes_df['category_encoded'] = 0
            
        # Create feature matrix
        feature_columns = base_features + ['node_type_encoded', 'category_encoded']
        
        # Add type-specific features
        type_specific_features = []
        
        # Product features
        if 'product_price' in nodes_df.columns:
            nodes_df['product_price'] = nodes_df['product_price'].fillna(0)
            type_specific_features.append('product_price')
            
        if 'historical_delay_rate' in nodes_df.columns:
            nodes_df['historical_delay_rate'] = nodes_df['historical_delay_rate'].fillna(0)
            type_specific_features.append('historical_delay_rate')
            
        # Customer features
        if 'purchase_rate' in nodes_df.columns:
            nodes_df['purchase_rate'] = nodes_df['purchase_rate'].fillna(0)
            type_specific_features.append('purchase_rate')
            
        if 'customer_value_score' in nodes_df.columns:
            nodes_df['customer_value_score'] = nodes_df['customer_value_score'].fillna(0)
            type_specific_features.append('customer_value_score')
            
        # Warehouse features
        if 'utilization_rate' in nodes_df.columns:
            nodes_df['utilization_rate'] = nodes_df['utilization_rate'].fillna(0)
            type_specific_features.append('utilization_rate')
            
        # Add type-specific features to feature columns
        feature_columns.extend(type_specific_features)
        
        # Create feature matrix
        feature_matrix = nodes_df[feature_columns].values.astype(np.float32)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Create node mapping
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'])}
        
        feature_info = {
            'feature_columns': feature_columns,
            'num_features': len(feature_columns),
            'node_id_to_idx': node_id_to_idx,
            'node_type_encoder': self.node_type_encoder,
            'category_encoder': self.category_encoder,
            'scaler': self.scaler
        }
        
        print(f"[FEATURES] Created feature matrix: {feature_matrix_scaled.shape}")
        print(f"[FEATURES] Feature columns: {feature_columns}")
        
        return feature_matrix_scaled, feature_info
        
    def prepare_edge_features(self, edges_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare edge features and edge index."""
        print("[FEATURES] Preparing edge features...")
        
        # Encode shipping modes
        edges_df['shipping_mode_encoded'] = self.shipping_mode_encoder.fit_transform(edges_df['shipping_mode'])
        
        # Create edge feature matrix
        edge_feature_columns = ['interaction_frequency', 'edge_weight', 'recency_score', 
                              'transit_time', 'shipping_mode_encoded']
        
        edge_features = edges_df[edge_feature_columns].values.astype(np.float32)
        
        # Create edge index (will be mapped to node indices later)
        edge_index = edges_df[['source_node_id', 'target_node_id']].values
        
        print(f"[FEATURES] Created edge features: {edge_features.shape}")
        print(f"[FEATURES] Edge feature columns: {edge_feature_columns}")
        
        return edge_features, edge_index
        
    def create_targets(self, nodes_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create target variables for multi-task learning."""
        print("[TARGETS] Creating target variables...")
        
        targets = {}
        
        # 1. Delay Risk (regression, 0-1)
        targets['delay_risk'] = nodes_df['delay_risk'].values.astype(np.float32)
        
        # 2. Inventory Shortfall (binary classification)
        # Define shortfall as inventory < 0.3
        targets['inventory_shortfall'] = (nodes_df['inventory_level'] < 0.3).astype(np.float32)
        
        # 3. Carbon Cost (regression)
        targets['carbon_cost'] = nodes_df['carbon_cost'].values.astype(np.float32)
        
        print(f"[TARGETS] Created targets:")
        for target_name, target_values in targets.items():
            print(f"          - {target_name}: {target_values.shape}, range: [{target_values.min():.3f}, {target_values.max():.3f}]")
            
        return targets
        
    def build_enhanced_graph(self, split_name: str = 'train') -> Tuple[Data, Dict, Dict]:
        """Build the complete enhanced graph."""
        print(f"[BUILD] Building enhanced graph for {split_name} split...")
        
        # Load data
        df_access, df_supply, df_desc = self.load_raw_data()
        
        # Create nodes and edges
        nodes_df = self.create_enhanced_nodes(df_access, df_supply, df_desc)
        edges_df = self.create_enhanced_edges(df_access, df_supply, nodes_df)
        
        # Prepare features
        node_features, feature_info = self.prepare_node_features(nodes_df)
        edge_features, edge_index_raw = self.prepare_edge_features(edges_df)
        
        # Map edge indices to node indices
        node_id_to_idx = feature_info['node_id_to_idx']
        
        valid_edges = []
        valid_edge_features = []
        
        for i, (src_id, tgt_id) in enumerate(edge_index_raw):
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]
                valid_edges.append([src_idx, tgt_idx])
                valid_edge_features.append(edge_features[i])
                
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(valid_edge_features, dtype=torch.float32) if valid_edge_features else None
        
        # Create targets
        targets = self.create_targets(nodes_df)
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        print(f"[BUILD] Enhanced graph created:")
        print(f"        - Nodes: {data.x.shape[0]} (features: {data.x.shape[1]})")
        print(f"        - Edges: {data.edge_index.shape[1]} (features: {edge_attr.shape[1] if edge_attr is not None else 0})")
        print(f"        - Targets: {len(targets)} tasks")
        
        return data, targets, feature_info

def main():
    """Test the enhanced graph builder."""
    builder = EnhancedGraphBuilder()
    data, targets, feature_info = builder.build_enhanced_graph()
    
    print("\n[TEST] Enhanced graph building completed successfully!")
    return data, targets, feature_info

if __name__ == '__main__':
    main()

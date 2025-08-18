# src/models/run_rl_simulation.py
"""
Script to run a simulation/demo of the RL agent using trained Intent and GNN models.
This demonstrates the full flow: Access Log -> Intent/Urgency -> Supply State -> Decision.
"""
import pandas as pd
import torch
import numpy as np
import os
import sys

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import models and agent
from models.intent_transformer import IntentTransformer
from models.supply_gnn import SupplyGNN
from models.rl_agent import RLAgent, State

# Import data loading utilities (you might want to put these in a common utils file)
# For now, we'll recreate the essential parts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

def load_trained_intent_model(model_path, vocab_size):
    """Loads the trained Intent Transformer model."""
    model = IntentTransformer(vocab_size=vocab_size) # Ensure params match training
    model.load_state_dict(torch.load(model_path, map_location='cpu')) # Load to CPU for demo
    model.eval()
    print(f"Intent Transformer model loaded from {model_path}")
    return model

def load_trained_gnn_model(model_path):
    """Loads the trained Supply GNN model and associated objects."""
    checkpoint = torch.load(model_path, map_location='cpu')
    # We need num_node_features to init the model. Let's get it from the checkpoint data
    # This is a simplification. In practice, you'd save this config or derive it.
    # Let's assume we know it or can load the scaler/encoder to infer it.
    # For demo, we'll hardcode or re-derive. Let's load scaler to get num features.
    scaler = checkpoint['scaler']
    num_node_features = scaler.mean_.shape[0] # Number of features used for scaling
    
    model = SupplyGNN(num_node_features=num_node_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Supply GNN model loaded from {model_path}")
    return model, scaler, checkpoint['le_node_type'], checkpoint['node_id_to_idx']

def prepare_sample_access_sequence(df_access, product_to_idx, sequence_length=10):
    """Prepares a sample sequence from access logs for the Transformer."""
    # Get the first user's sequence for demo
    first_user = df_access['user_id'].iloc[0]
    user_group = df_access[df_access['user_id'] == first_user].head(sequence_length)
    
    product_ids = user_group['Product Card Id'].tolist()
    # Pad or truncate to sequence_length
    if len(product_ids) < sequence_length:
        product_ids.extend([0] * (sequence_length - len(product_ids))) # 0 for padding
    elif len(product_ids) > sequence_length:
        product_ids = product_ids[:sequence_length]
    
    # Map to indices
    sequence_indices = [product_to_idx.get(pid, 0) for pid in product_ids]
    return torch.tensor([sequence_indices], dtype=torch.long) # Add batch dim

def get_gnn_state_for_product(model, data, node_id_to_idx, product_id, scaler):
    """Gets the GNN embedding and features for a specific product node."""
    # Find the node index for the product
    node_idx = node_id_to_idx.get(product_id)
    if node_idx is None:
        print(f"Warning: Product ID {product_id} not found in GNN graph.")
        # Return dummy state
        return np.random.rand(16), {'WH1': {'risk': 0.5, 'inventory': 0.5, 'carbon': 0.5}} 

    # Run GNN
    with torch.no_grad():
        node_embeddings, node_risks, node_utils = model(data)
    
    # Get embedding and risk for this product node
    product_embedding = node_embeddings[node_idx].cpu().numpy()
    product_risk = node_risks[node_idx].item()
    
    # Simulate warehouse states (in a full system, you'd link product to warehouses)
    # For demo, let's assume fixed warehouses with dummy states derived from graph data
    # Or fetch them based on edges. Let's simplify.
    warehouse_states = {
        'WH_Latam': {'risk': 0.2, 'inventory': 0.8, 'carbon_cost': 0.7},
        'WH_Africa': {'risk': 0.4, 'inventory': 0.6, 'carbon_cost': 0.9},
        'WH_USCA': {'risk': 0.1, 'inventory': 0.9, 'carbon_cost': 0.4},
    }
    # In a real scenario, you'd look up warehouses connected to this product node
    # and get their specific embeddings/risks from the GNN output.
    
    return product_embedding, warehouse_states

def main():
    """Main simulation function."""
    print("--- Running RL Simulation with Trained Models (Phase 2 Demo) ---")
    
    # --- 1. Load Sample Data for Context ---
    access_logs_file = 'data/processed/processed_access_logs.csv' # Use the linked one
    df_access = pd.read_csv(access_logs_file)
    df_access.dropna(subset=['Product Card Id'], inplace=True)
    df_access['Product Card Id'] = df_access['Product Card Id'].astype(int)
    unique_products = df_access['Product Card Id'].unique()
    product_to_idx = {pid: idx+1 for idx, pid in enumerate(unique_products)}
    product_to_idx[0] = 0 # Add padding token
    vocab_size = len(product_to_idx)
    
    # --- 2. Load Trained Models ---
    intent_model_path = 'models/saved/intent_transformer.pth'
    gnn_model_path = 'models/saved/supply_gnn.pth'
    
    if not os.path.exists(intent_model_path) or not os.path.exists(gnn_model_path):
        print("Error: Trained models not found. Please run training scripts first.")
        print(f"  Intent Model Path: {intent_model_path}")
        print(f"  GNN Model Path: {gnn_model_path}")
        return

    intent_model = load_trained_intent_model(intent_model_path, vocab_size)
    gnn_model, scaler, le_node_type, node_id_to_idx = load_trained_gnn_model(gnn_model_path)

    # --- 3. Prepare Sample Input (Sequence) ---
    sample_sequence = prepare_sample_access_sequence(df_access, product_to_idx, sequence_length=10)
    print(f"Sample input sequence for Transformer: {sample_sequence.squeeze().tolist()}")

    # --- 4. Get Intent/Urgency from Transformer ---
    with torch.no_grad():
        intent_score, urgency_score, _ = intent_model(sample_sequence)
    intent_score_val = intent_score.item()
    urgency_score_val = urgency_score.item()
    print(f"Transformer Output -> Intent Score: {intent_score_val:.4f}, Urgency Score: {urgency_score_val:.4f}")

    # --- 5. Get Supply State from GNN ---
    # Assume the last product in the sequence is the one of interest
    last_product_id = df_access[df_access['user_id'] == df_access['user_id'].iloc[0]]['Product Card Id'].iloc[9] # 10th item (0-indexed)
    print(f"Focusing on Product ID: {last_product_id}")
    
    # Load GNN data for state (in a real system, this would be loaded once and updated)
    nodes_df = pd.read_csv('data/processed/gnn_nodes.csv')
    edges_df = pd.read_csv('data/processed/gnn_edges.csv')
    # Recreate data object (simplified)
    nodes_df['node_type_encoded'] = le_node_type.transform(nodes_df['node_type'])
    feature_columns = ['node_type_encoded', 'Product Price', 'inventory_level', 'delay_risk', 'carbon_cost', 'supplier_reliability']
    nodes_df[feature_columns] = nodes_df[feature_columns].fillna(0)
    node_features = scaler.transform(nodes_df[feature_columns])
    x = torch.tensor(node_features, dtype=torch.float)
    
    unique_nodes = pd.concat([edges_df['source_node_id'], edges_df['target_node_id']]).unique()
    temp_node_id_to_idx = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    edges_df['source_idx'] = edges_df['source_node_id'].map(temp_node_id_to_idx)
    edges_df['target_idx'] = edges_df['target_node_id'].map(temp_node_id_to_idx)
    edges_df.dropna(subset=['source_idx', 'target_idx'], inplace=True)
    edge_index = torch.tensor([edges_df['source_idx'].values, edges_df['target_idx'].values], dtype=torch.long)
    gnn_data = Data(x=x, edge_index=edge_index)
    
    product_embedding, warehouse_states = get_gnn_state_for_product(gnn_model, gnn_data, temp_node_id_to_idx, last_product_id, scaler)
    print(f"GNN processed state for Product {last_product_id}.")

    # --- 6. Create State Object for RL Agent ---
    state = State(
        intent_score=intent_score_val,
        urgency_score=urgency_score_val,
        product_embedding=product_embedding,
        warehouse_states=warehouse_states
    )
    print("RL Agent State object created.")

    # --- 7. Instantiate and Run RL Agent ---
    warehouse_ids = list(warehouse_states.keys())
    shipping_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
    rl_agent = RLAgent(warehouse_ids, shipping_modes)
    
    action = rl_agent.act(state)
    print("\n--- RL Agent Decision ---")
    print(f"Fulfill From Warehouse: {action.fulfill_from_warehouse}")
    print(f"Shipping Mode: {action.shipping_mode}")
    print(f"Offer Bundle: {action.offer_bundle}")
    print(f"Offer Sustainability Choice: {action.offer_sustainability_choice}")
    print(f"Apply Dynamic Pricing: {action.apply_dynamic_pricing}")
    if action.apply_dynamic_pricing:
        print(f"  Price Multiplier: {action.dynamic_price_multiplier:.2f}")

    print("\n--- Simulation Complete ---")

if __name__ == '__main__':
    main()
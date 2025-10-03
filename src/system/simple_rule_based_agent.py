# src/system/simple_rule_based_agent.py
"""
Updated Simple Rule-Based Agent for SynchroChain
Fixed imports and improved error handling
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import math
import time
from typing import Dict, Any, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the corrected GNN model
try:
    from src.models.gnn.model import SupplyGNN
    print("SupplyGNN imported successfully")
except ImportError as e:
    print(f"Failed to import SupplyGNN: {e}")
    raise

class SimpleRuleBasedAgent:
    """Simple rule-based agent using GNN predictions."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the agent with configuration."""
        self.project_root = project_root
        self.device = torch.device('cpu')
        
        # Model paths
        self.model_save_path = os.path.join(project_root, 'models', 'saved', 'supply_gnn_final.pth')
        self.scaler_save_path = os.path.join(project_root, 'models', 'saved', 'gnn_scaler.pkl')
        self.le_node_type_save_path = os.path.join(project_root, 'models', 'saved', 'gnn_le_node_type.pkl')
        self.node_id_to_idx_save_path = os.path.join(project_root, 'models', 'saved', 'gnn_node_id_to_idx.pkl')
        
        # Load components
        self.model, self.scaler, self.le_node_type, self.node_id_to_idx = self._load_trained_components()
    
    def _load_trained_components(self) -> Tuple[SupplyGNN, StandardScaler, LabelEncoder, dict]:
        """Load trained GNN components."""
        print("Loading trained GNN components...")
        
        # Check if files exist
        required_files = [self.model_save_path, self.scaler_save_path, 
                         self.le_node_type_save_path, self.node_id_to_idx_save_path]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load scaler and encoders
        with open(self.scaler_save_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(self.le_node_type_save_path, 'rb') as f:
            le_node_type = pickle.load(f)
        with open(self.node_id_to_idx_save_path, 'rb') as f:
            node_id_to_idx = pickle.load(f)
        
        # Initialize model with correct parameters
        model = SupplyGNN(
            num_node_features=7,
            hidden_dim=64,
            output_dim=64,
            num_heads=2,
            num_layers=3,
            dropout=0.1
        )
        
        # Load model weights
        checkpoint = torch.load(self.model_save_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Trained components loaded successfully")
        return model, scaler, le_node_type, node_id_to_idx
    
    def make_decision(self, predicted_delay_risk: float, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make decision based on predicted delay risk."""
        decision = {}
        
        if predicted_delay_risk < 0.3:
            decision['action'] = 'Standard Shipping'
            decision['reason'] = f'Low predicted delay risk ({predicted_delay_risk:.2f}). Standard shipping is sufficient.'
            decision['priority'] = 'Normal'
        elif predicted_delay_risk < 0.6:
            decision['action'] = 'Expedited Shipping'
            decision['reason'] = f'Moderate predicted delay risk ({predicted_delay_risk:.2f}). Expedited shipping recommended.'
            decision['priority'] = 'High'
        else:
            decision['action'] = 'Priority Fulfillment & Alternate Supplier Check'
            decision['reason'] = f'High predicted delay risk ({predicted_delay_risk:.2f}). Highest priority fulfillment initiated.'
            decision['priority'] = 'Critical'
        
        decision['predicted_delay_risk'] = predicted_delay_risk
        return decision
    
    def process_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single event using the GNN.
        
        Args:
            event_data: Dictionary containing event information
            
        Returns:
            Decision made by the rule-based agent
        """
        print(f"\nProcessing event: {event_data}")
        
        try:
            # Prepare node features
            node_type_str = event_data['node_type']
            node_type_encoded = self.le_node_type.transform([node_type_str])[0]
            
            # Extract features
            features = np.array([
                node_type_encoded,
                event_data.get('Product Price', 0.0),
                event_data.get('inventory_level', 0.0),
                event_data.get('carbon_cost', 0.0),
                event_data.get('supplier_reliability', 1.0),
                event_data.get('product_historical_delay_rate', 0.0),
                event_data.get('region_historical_delay_rate', 0.0)
            ]).reshape(1, -1)
            
            # Normalize features
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float)
            
            # Create PyG Data object
            edge_index_empty = torch.empty((2, 0), dtype=torch.long)
            single_node_data = Data(x=features_tensor, edge_index=edge_index_empty)
            
            # Get GNN prediction
            self.model.eval()
            with torch.no_grad():
                _, predicted_delay_risk_tensor, _ = self.model(single_node_data)
                predicted_delay_risk = predicted_delay_risk_tensor.item()
            
            print(f"GNN Predicted Delay Risk: {predicted_delay_risk:.4f}")
            
            # Make decision
            decision = self.make_decision(predicted_delay_risk, product_info=event_data)
            print(f"Rule-Based Agent Decision: {decision}")
            
            return decision
            
        except Exception as e:
            print(f"Error processing event: {e}")
            # Fallback decision
            fallback_decision = {
                'action': 'Standard Shipping',
                'reason': f'Error processing event: {e}. Falling back to standard procedure.',
                'priority': 'Normal',
                'predicted_delay_risk': 0.5
            }
            return fallback_decision
    
    def run_demo(self):
        """Run a demonstration of the rule-based agent."""
        print("=== Simple Rule-Based Agent Demo ===")
        
        # Example events
        events = [
            {
                'node_type': 'product',
                'Product Price': 25.99,
                'inventory_level': 50,
                'carbon_cost': 2.1,
                'supplier_reliability': 0.95,
                'product_historical_delay_rate': 0.05,
                'region_historical_delay_rate': 0.02
            },
            {
                'node_type': 'product',
                'Product Price': 1500.00,
                'inventory_level': 2,
                'carbon_cost': 50.0,
                'supplier_reliability': 0.6,
                'product_historical_delay_rate': 0.4,
                'region_historical_delay_rate': 0.3
            },
            {
                'node_type': 'product',
                'Product Price': 120.50,
                'inventory_level': 15,
                'carbon_cost': 12.0,
                'supplier_reliability': 0.75,
                'product_historical_delay_rate': 0.2,
                'region_historical_delay_rate': 0.15
            }
        ]
        
        for i, event in enumerate(events, 1):
            print(f"\n--- Event {i} ---")
            decision = self.process_event(event)
            print(f"Decision: {decision['action']} (Priority: {decision['priority']})")
            print(f"Reason: {decision['reason']}")

def main():
    """Main function to run the rule-based agent."""
    try:
        agent = SimpleRuleBasedAgent()
        agent.run_demo()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the GNN model is trained and saved correctly.")
        print("Run the preprocessing and training scripts first.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
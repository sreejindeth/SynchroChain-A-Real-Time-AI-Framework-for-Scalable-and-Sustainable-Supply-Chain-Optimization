# src/models/gnn/enhanced_supply_gnn.py
"""
Enhanced Multi-Task Supply Chain GNN for SynchroChain
Supports multiple node types and multi-task predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from typing import Dict, Tuple, Optional, List
import numpy as np

class EnhancedSupplyGNN(nn.Module):
    """
    Enhanced multi-task GNN for supply chain optimization.
    
    Features:
    - Multi-task learning: delay_risk, inventory_shortfall, carbon_cost
    - Node-type aware processing
    - Edge feature integration
    - Attention mechanisms
    """
    
    def __init__(self, 
                 num_node_features: int,
                 num_edge_features: int = 0,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 gnn_type: str = 'gatv2',
                 use_edge_features: bool = True,
                 node_types: List[str] = ['customer', 'product', 'warehouse', 'supplier']):
        
        super(EnhancedSupplyGNN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.use_edge_features = use_edge_features and num_edge_features > 0
        self.node_types = node_types
        
        # Input projection
        self.input_projection = nn.Linear(num_node_features, hidden_dim)
        
        # Edge feature projection (if using edge features)
        if self.use_edge_features:
            self.edge_projection = nn.Linear(num_edge_features, hidden_dim // 4)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == 'gatv2':
                conv = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=hidden_dim // 4 if self.use_edge_features else None,
                    share_weights=True
                )
            elif self.gnn_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == 'sage':
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
                
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Node-type specific processing
        self.node_type_processors = nn.ModuleDict()
        for node_type in node_types:
            self.node_type_processors[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Final projection
        self.final_projection = nn.Linear(hidden_dim, output_dim)
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict()
        
        # 1. Delay Risk (regression, 0-1)
        self.prediction_heads['delay_risk'] = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 2. Inventory Shortfall (binary classification)
        self.prediction_heads['inventory_shortfall'] = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 3. Carbon Cost (regression)
        self.prediction_heads['carbon_cost'] = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()  # Assuming carbon cost is normalized 0-1
        )
        
        # Attention mechanism for multi-task fusion
        self.task_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, data, node_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced GNN.
        
        Args:
            data: PyG Data object with x, edge_index, and optionally edge_attr
            node_types: Tensor of node type indices (optional)
            
        Returns:
            Dictionary containing:
            - node_embeddings: Final node embeddings
            - delay_risk: Delay risk predictions
            - inventory_shortfall: Inventory shortfall predictions  
            - carbon_cost: Carbon cost predictions
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process edge features if available
        if self.use_edge_features and edge_attr is not None:
            edge_attr = self.edge_projection(edge_attr)
        
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            residual = x
            
            # Apply convolution
            if self.gnn_type == 'gatv2' and self.use_edge_features and edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # Batch normalization
            x = bn(x)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if residual.shape == x.shape:
                x = x + residual
        
        # Node-type specific processing (if node types provided)
        if node_types is not None:
            processed_x = torch.zeros_like(x)
            for i, node_type in enumerate(self.node_types):
                mask = (node_types == i)
                if mask.any():
                    processed_x[mask] = self.node_type_processors[node_type](x[mask])
            x = processed_x
        
        # Final projection to get node embeddings
        node_embeddings = self.final_projection(x)
        
        # Apply attention mechanism for better representations
        # Reshape for attention: [batch_size=1, seq_len=num_nodes, embed_dim]
        x_attn = node_embeddings.unsqueeze(0)
        attn_output, _ = self.task_attention(x_attn, x_attn, x_attn)
        enhanced_embeddings = attn_output.squeeze(0)
        
        # Multi-task predictions
        predictions = {}
        predictions['node_embeddings'] = enhanced_embeddings
        
        # Apply prediction heads
        predictions['delay_risk'] = self.prediction_heads['delay_risk'](enhanced_embeddings)
        predictions['inventory_shortfall'] = self.prediction_heads['inventory_shortfall'](enhanced_embeddings)
        predictions['carbon_cost'] = self.prediction_heads['carbon_cost'](enhanced_embeddings)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    task_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            task_weights: Optional weights for each task
            
        Returns:
            Dictionary of losses
        """
        if task_weights is None:
            task_weights = {
                'delay_risk': 1.0,
                'inventory_shortfall': 1.0,
                'carbon_cost': 1.0
            }
        
        losses = {}
        
        # Delay risk loss (MSE)
        if 'delay_risk' in predictions and 'delay_risk' in targets:
            losses['delay_risk'] = F.mse_loss(
                predictions['delay_risk'].squeeze(),
                targets['delay_risk']
            ) * task_weights['delay_risk']
        
        # Inventory shortfall loss (BCE)
        if 'inventory_shortfall' in predictions and 'inventory_shortfall' in targets:
            losses['inventory_shortfall'] = F.binary_cross_entropy(
                predictions['inventory_shortfall'].squeeze(),
                targets['inventory_shortfall']
            ) * task_weights['inventory_shortfall']
        
        # Carbon cost loss (MSE)
        if 'carbon_cost' in predictions and 'carbon_cost' in targets:
            losses['carbon_cost'] = F.mse_loss(
                predictions['carbon_cost'].squeeze(),
                targets['carbon_cost']
            ) * task_weights['carbon_cost']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy for metric computation
        with torch.no_grad():
            # Delay risk metrics (regression)
            if 'delay_risk' in predictions and 'delay_risk' in targets:
                pred_delay = predictions['delay_risk'].squeeze().cpu().numpy()
                true_delay = targets['delay_risk'].cpu().numpy()
                
                mse = np.mean((pred_delay - true_delay) ** 2)
                mae = np.mean(np.abs(pred_delay - true_delay))
                
                # R² score
                ss_res = np.sum((true_delay - pred_delay) ** 2)
                ss_tot = np.sum((true_delay - np.mean(true_delay)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                metrics['delay_risk_mse'] = mse
                metrics['delay_risk_mae'] = mae
                metrics['delay_risk_r2'] = r2
            
            # Inventory shortfall metrics (classification)
            if 'inventory_shortfall' in predictions and 'inventory_shortfall' in targets:
                pred_shortfall = (predictions['inventory_shortfall'].squeeze() > 0.5).cpu().numpy()
                true_shortfall = targets['inventory_shortfall'].cpu().numpy()
                
                accuracy = np.mean(pred_shortfall == true_shortfall)
                
                # Precision, Recall, F1
                tp = np.sum((pred_shortfall == 1) & (true_shortfall == 1))
                fp = np.sum((pred_shortfall == 1) & (true_shortfall == 0))
                fn = np.sum((pred_shortfall == 0) & (true_shortfall == 1))
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics['inventory_shortfall_accuracy'] = accuracy
                metrics['inventory_shortfall_precision'] = precision
                metrics['inventory_shortfall_recall'] = recall
                metrics['inventory_shortfall_f1'] = f1
            
            # Carbon cost metrics (regression)
            if 'carbon_cost' in predictions and 'carbon_cost' in targets:
                pred_carbon = predictions['carbon_cost'].squeeze().cpu().numpy()
                true_carbon = targets['carbon_cost'].cpu().numpy()
                
                mse = np.mean((pred_carbon - true_carbon) ** 2)
                mae = np.mean(np.abs(pred_carbon - true_carbon))
                
                # R² score
                ss_res = np.sum((true_carbon - pred_carbon) ** 2)
                ss_tot = np.sum((true_carbon - np.mean(true_carbon)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                metrics['carbon_cost_mse'] = mse
                metrics['carbon_cost_mae'] = mae
                metrics['carbon_cost_r2'] = r2
        
        return metrics
    
    def __repr__(self):
        """String representation of the model."""
        return (f'EnhancedSupplyGNN(\n'
                f'  num_node_features={self.num_node_features},\n'
                f'  num_edge_features={self.num_edge_features},\n'
                f'  hidden_dim={self.hidden_dim},\n'
                f'  output_dim={self.output_dim},\n'
                f'  num_heads={self.num_heads},\n'
                f'  num_layers={self.num_layers},\n'
                f'  dropout={self.dropout},\n'
                f'  gnn_type={self.gnn_type},\n'
                f'  use_edge_features={self.use_edge_features},\n'
                f'  node_types={self.node_types}\n'
                f')')

# Lightweight version for faster training
class LightweightSupplyGNN(nn.Module):
    """Lightweight version of the supply chain GNN for faster training."""
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64, 
                 output_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super(LightweightSupplyGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Simple GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Prediction heads
        self.delay_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.inventory_shortfall_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.carbon_cost_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        """Forward pass."""
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Predictions
        predictions = {
            'node_embeddings': x,
            'delay_risk': self.delay_risk_head(x),
            'inventory_shortfall': self.inventory_shortfall_head(x),
            'carbon_cost': self.carbon_cost_head(x)
        }
        
        return predictions

def main():
    """Test the enhanced GNN model."""
    # Test parameters
    num_nodes = 1000
    num_edges = 5000
    num_node_features = 10
    num_edge_features = 5
    
    # Create dummy data
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, num_edge_features)
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Test enhanced model
    model = EnhancedSupplyGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        output_dim=64
    )
    
    print("Enhanced Supply GNN Model:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(data)
        
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # Test lightweight model
    lightweight_model = LightweightSupplyGNN(num_node_features=num_node_features)
    print(f"\nLightweight model parameters: {sum(p.numel() for p in lightweight_model.parameters()):,}")
    
    return model, lightweight_model

if __name__ == '__main__':
    main()






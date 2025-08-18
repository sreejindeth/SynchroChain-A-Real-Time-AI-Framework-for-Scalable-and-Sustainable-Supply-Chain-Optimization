# src/models/supply_gnn.py
"""
Supply Chain GNN Model for SynchroChain.
This model takes a PyG Data object representing the supply chain graph
and learns node embeddings to predict attributes like delay_risk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SupplyGNN(torch.nn.Module):
    """
    A GNN model for supply chain analysis using GATv2 layers.
    
    Attributes:
        num_layers (int): The number of GATv2Conv layers.
        dropout (float): Dropout rate used between layers.
        conv1 (GATv2Conv): The first GATv2 convolutional layer.
        convs (torch.nn.ModuleList): List of subsequent GATv2 layers.
        bns (torch.nn.ModuleList): List of BatchNorm layers for hidden states.
        final_projection (nn.Linear): Linear layer to project to final output dimension.
        delay_risk_head (nn.Sequential): MLP head for predicting delay risk.
    """
    def __init__(self, num_node_features, hidden_dim=32, output_dim=16, num_heads=2, num_layers=2, dropout=0.1):
        """
        Initializes the SupplyGNN model.

        Args:
            num_node_features (int): Number of input features per node.
            hidden_dim (int, optional): Hidden dimension size for GAT layers. Defaults to 32.
            output_dim (int, optional): Output dimension size after final projection. Defaults to 16.
            num_heads (int, optional): Number of attention heads in GAT layers. Defaults to 2.
                                     Note: hidden_dim * num_heads is the intermediate size.
            num_layers (int, optional): Number of GATv2Conv layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(SupplyGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # --- GNN Layers ---
        # Define the first GAT layer
        # Input size: num_node_features
        # Output size: hidden_dim * num_heads (because of concatenation)
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True # Concatenate multi-head outputs -> hidden_dim * num_heads
        )
        
        # Container for subsequent GAT layers
        self.convs = torch.nn.ModuleList()
        # Container for BatchNorm layers (applied after conv, before ReLU/activation)
        self.bns = torch.nn.ModuleList()

        # Define subsequent GAT layers (from layer 2 to num_layers)
        # Input size for these: hidden_dim * num_heads (output of previous layer)
        # Output size: hidden_dim * num_heads (assuming consistent hidden size)
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * num_heads, # Input from previous layer
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True # Consistent concatenation
                )
            )
            # BatchNorm for the output of the conv layer
            # Input size for BN: hidden_dim * num_heads
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))

        # --- Final Projection ---
        # Projects the final GNN output (hidden_dim * num_heads) to a smaller embedding space (output_dim)
        self.final_projection = nn.Linear(hidden_dim * num_heads, output_dim)

        # --- Prediction Heads ---
        # Example head for predicting 'delay_risk' (single float)
        self.delay_risk_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), # Apply dropout before the final prediction layer
            nn.Linear(output_dim // 2, 1) # Predict a single scalar value
            # No activation here if the target (delay_risk) can be any real number.
            # If delay_risk is normalized to [0, 1], consider adding a Sigmoid().
        )
        # --- End Prediction Heads ---
        # You can add more heads here for other node attributes (e.g., inventory_level)
        # self.inventory_head = nn.Sequential(...)
        # self.carbon_cost_head = nn.Sequential(...)
        # -------------------------

    def forward(self, data):
        """
        Performs a forward pass through the GNN.

        Args:
            data (torch_geometric.data.Data): A PyG Data object containing:
                - x (Tensor): Node feature matrix of shape [num_nodes, num_node_features].
                - edge_index (LongTensor): Graph connectivity matrix of shape [2, num_edges].

        Returns:
            tuple: A tuple containing:
                - node_embeddings (Tensor): Final node embeddings of shape [num_nodes, output_dim].
                - predicted_delay_risk (Tensor): Predicted delay risk of shape [num_nodes, 1].
                - predicted_utilization (Tensor): Placeholder, currently None.
                  (Can be replaced with output from another head if needed).
        """
        x, edge_index = data.x, data.edge_index

        # --- Apply First GAT Layer ---
        # x: [num_nodes, num_node_features] -> [num_nodes, hidden_dim * num_heads]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Apply Subsequent GAT Layers ---
        # Iterate through the remaining layers defined in self.convs and self.bns
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # conv: [num_nodes, hidden_dim * num_heads] -> [num_nodes, hidden_dim * num_heads]
            x = conv(x, edge_index)
            # bn: [num_nodes, hidden_dim * num_heads] -> [num_nodes, hidden_dim * num_heads]
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Final Projection to Node Embeddings ---
        # x: [num_nodes, hidden_dim * num_heads] -> [num_nodes, output_dim]
        node_embeddings = self.final_projection(x)

        # --- Apply Prediction Heads ---
        # Predict delay risk from the final node embeddings
        # predicted_delay_risk: [num_nodes, output_dim] -> [num_nodes, 1]
        predicted_delay_risk = self.delay_risk_head(node_embeddings)

        # Placeholder for other predictions (e.g., inventory)
        predicted_utilization = None # Or output from self.inventory_head(node_embeddings)

        # Return embeddings and specific predictions
        return node_embeddings, predicted_delay_risk, predicted_utilization

    def __repr__(self):
        """String representation of the model, useful for debugging."""
        return (f'SupplyGNN(\n'
                f'  num_node_features={self.conv1.in_channels},\n'
                f'  hidden_dim={self.conv1.out_channels},\n'
                f'  output_dim={self.final_projection.out_features},\n'
                f'  num_heads={self.conv1.heads},\n'
                f'  num_layers={self.num_layers},\n'
                f'  dropout={self.dropout}\n'
                f')')
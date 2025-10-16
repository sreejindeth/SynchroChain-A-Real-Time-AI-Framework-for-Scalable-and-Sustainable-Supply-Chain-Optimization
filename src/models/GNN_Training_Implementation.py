"""
GNN Training Implementation with Proper Train-Test Splits
Uses temporal splits for delay risk prediction in supply chain
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DelayRiskGNN(nn.Module):
    """Graph Neural Network for delay risk prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, 
                 num_heads: int = 2, dropout: float = 0.1):
        super(DelayRiskGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1 for risk probability
        )
        
    def forward(self, x, edge_index=None):
        """Forward pass through the GNN."""
        # Input projection
        x = self.input_proj(x)
        
        # Graph attention layers
        for layer in self.gat_layers:
            x = layer(x, edge_index)
        
        # Global pooling (mean pooling)
        x = torch.mean(x, dim=1) if len(x.shape) > 2 else x
        
        # Output prediction
        risk_score = self.output_layers(x)
        
        return risk_score

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer implementation."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, dropout: float = 0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        self.attention = nn.MultiheadAttention(out_features, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index=None):
        """Forward pass through attention layer."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection and layer norm
        x = self.norm(x + self.dropout(attn_output))
        
        return x

class GNNTrainer:
    """GNN Trainer with proper train/validation/test splits."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = {}
        self.model = None
        
    def load_temporal_splits(self, data_path: str) -> Tuple[Dict, Dict, Dict]:
        """Load train/val/test data from temporal splits."""
        print("📂 Loading temporal split data...")
        
        splits = {}
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(data_path, split)
            
            # Load GNN data
            nodes_df = pd.read_csv(os.path.join(split_path, 'gnn_nodes.csv'))
            edges_df = pd.read_csv(os.path.join(split_path, 'gnn_edges.csv'))
            logs_df = pd.read_csv(os.path.join(split_path, 'processed_access_logs.csv'))
            
            splits[split] = {
                'nodes': nodes_df,
                'edges': edges_df,
                'logs': logs_df
            }
            
            print(f"   {split.capitalize()}: {len(nodes_df)} nodes, {len(edges_df)} edges, {len(logs_df)} logs")
        
        return splits['train'], splits['val'], splits['test']
    
    def prepare_features(self, train_data: Dict, val_data: Dict, test_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare features and targets for training."""
        print("🔧 Preparing features and targets...")
        
        # Extract features from nodes (supply chain entities)
        def extract_node_features(nodes_df):
            features = []
            for _, row in nodes_df.iterrows():
                # Node features: entity type, location, capacity, etc.
                node_features = [
                    row.get('entity_type', 0),  # 0: supplier, 1: warehouse, 2: retailer
                    row.get('location_risk', 0.5),  # Location-based risk
                    row.get('capacity_utilization', 0.5),  # Current capacity usage
                    row.get('historical_delay_rate', 0.1),  # Past delay frequency
                    row.get('reliability_score', 0.8),  # Supplier reliability
                ]
                features.append(node_features)
            return np.array(features)
        
        # Extract targets from logs (delay information)
        def extract_targets(logs_df):
            targets = []
            for _, row in logs_df.iterrows():
                # Target: delay risk (0-1)
                delay_risk = min(1.0, max(0.0, row.get('delay_risk', 0.1)))
                targets.append(delay_risk)
            return np.array(targets)
        
        # Process each split
        X_train = extract_node_features(train_data['nodes'])
        y_train = extract_targets(train_data['logs'])
        
        X_val = extract_node_features(val_data['nodes'])
        y_val = extract_targets(val_data['logs'])
        
        X_test = extract_node_features(test_data['nodes'])
        y_test = extract_targets(test_data['logs'])
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        print(f"   Train: {X_train_tensor.shape[0]} samples, {X_train_tensor.shape[1]} features")
        print(f"   Val: {X_val_tensor.shape[0]} samples")
        print(f"   Test: {X_test_tensor.shape[0]} samples")
        
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """Create data loaders for training."""
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, X_test, y_test):
        """Train the GNN model with proper validation."""
        print("🚀 Training GNN model...")
        
        # Initialize model
        input_dim = X_test.shape[1]
        self.model = DelayRiskGNN(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/gnn_best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                print(f"   Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= 10:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/gnn_best_model.pth'))
        
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test data."""
        print("📊 Evaluating model on test data...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        print(f"   Test MSE: {mse:.4f}")
        print(f"   Test RMSE: {rmse:.4f}")
        print(f"   Test MAE: {mae:.4f}")
        print(f"   Test R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
    
    def plot_training_progress(self, train_losses, val_losses):
        """Plot training progress."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GNN Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss (Zoomed)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/gnn_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_scalers(self):
        """Save the trained model and scalers."""
        os.makedirs('models', exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), 'models/final_delay_risk_model.pth')
        
        # Save scalers
        with open('models/final_target_scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print("💾 Model and scalers saved successfully!")

def main():
    """Main training function."""
    print("=" * 80)
    print("🧠 GNN TRAINING WITH PROPER TRAIN-TEST SPLITS")
    print("=" * 80)
    
    # Configuration
    config = {
        'hidden_dim': 64,
        'num_layers': 3,
        'num_heads': 2,
        'dropout': 0.1,
        'learning_rate': 0.00033,
        'weight_decay': 0.00086,
        'num_epochs': 20,
        'batch_size': 32
    }
    
    try:
        # Initialize trainer
        trainer = GNNTrainer(config)
        
        # Load temporal splits
        train_data, val_data, test_data = trainer.load_temporal_splits('data/processed/temporal_splits')
        
        # Prepare features
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(
            train_data, val_data, test_data
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, config['batch_size']
        )
        
        # Train model
        train_losses, val_losses = trainer.train_model(train_loader, val_loader, X_test, y_test)
        
        # Evaluate model
        test_results = trainer.evaluate_model(test_loader)
        
        # Plot training progress
        trainer.plot_training_progress(train_losses, val_losses)
        
        # Save model
        trainer.save_model_and_scalers()
        
        print("\n🎉 GNN training completed successfully!")
        print(f"📊 Final Test R²: {test_results['r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
















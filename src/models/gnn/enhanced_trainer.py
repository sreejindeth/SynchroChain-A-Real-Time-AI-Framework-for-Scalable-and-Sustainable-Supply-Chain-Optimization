# src/models/gnn/enhanced_trainer.py
"""
Enhanced GNN Training Pipeline for SynchroChain
Comprehensive training with multi-task learning and evaluation
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pickle
import time
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.models.gnn.enhanced_supply_gnn import EnhancedSupplyGNN, LightweightSupplyGNN
from src.models.gnn.enhanced_graph_builder import EnhancedGraphBuilder

class EnhancedGNNTrainer:
    """
    Enhanced GNN trainer with multi-task learning support.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the trainer."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.gnn_config = self.config['models']['gnn']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.project_root = project_root
        self.models_dir = os.path.join(project_root, 'models', 'saved')
        self.results_dir = os.path.join(project_root, 'results')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training parameters
        self.hidden_dim = self.gnn_config.get('hidden_dim', 128)
        self.num_layers = self.gnn_config.get('num_layers', 3)
        self.num_heads = self.gnn_config.get('num_heads', 4)
        self.dropout = self.gnn_config.get('dropout', 0.2)
        self.learning_rate = self.gnn_config.get('learning_rate', 0.001)
        self.weight_decay = self.gnn_config.get('weight_decay', 0.0005)
        self.num_epochs = self.gnn_config.get('num_epochs', 50)
        self.patience = self.gnn_config.get('patience', 10)
        
        # Task weights for multi-task learning
        self.task_weights = {
            'delay_risk': 1.0,
            'inventory_shortfall': 1.0,
            'carbon_cost': 1.0
        }
        
        print(f"[INIT] Enhanced GNN Trainer initialized")
        print(f"       Device: {self.device}")
        print(f"       Config: {self.gnn_config}")
        
    def prepare_data(self) -> Tuple[Dict[str, Data], Dict[str, Dict], Dict]:
        """Prepare training, validation, and test data."""
        print("[DATA] Preparing enhanced graph data...")
        
        builder = EnhancedGraphBuilder()
        
        # Build graphs for each split
        data_splits = {}
        targets_splits = {}
        
        for split in ['train', 'val', 'test']:
            print(f"[DATA] Building {split} graph...")
            try:
                data, targets, feature_info = builder.build_enhanced_graph(split)
                data_splits[split] = data.to(self.device)
                targets_splits[split] = {k: torch.tensor(v).to(self.device) for k, v in targets.items()}
                
                print(f"[DATA] {split.upper()} - Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
                
            except Exception as e:
                print(f"[WARNING] Could not build {split} graph: {e}")
                continue
        
        # Use train split feature info as reference
        if 'train' not in data_splits:
            raise ValueError("Could not create training data!")
            
        return data_splits, targets_splits, feature_info
        
    def create_model(self, num_node_features: int, num_edge_features: int = 0, 
                    model_type: str = 'enhanced') -> nn.Module:
        """Create the GNN model."""
        print(f"[MODEL] Creating {model_type} GNN model...")
        
        if model_type == 'enhanced':
            model = EnhancedSupplyGNN(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                gnn_type='gatv2',
                use_edge_features=num_edge_features > 0
            )
        elif model_type == 'lightweight':
            model = LightweightSupplyGNN(
                num_node_features=num_node_features,
                hidden_dim=self.hidden_dim // 2,
                output_dim=self.hidden_dim // 2,
                num_layers=max(2, self.num_layers - 1),
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model = model.to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Created {model_type} model with {num_params:,} parameters")
        
        return model
        
    def train_epoch(self, model: nn.Module, data: Data, targets: Dict[str, torch.Tensor],
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data)
        
        # Compute losses
        if hasattr(model, 'compute_loss'):
            losses = model.compute_loss(predictions, targets, self.task_weights)
        else:
            # Fallback for lightweight model
            losses = {}
            losses['delay_risk'] = nn.MSELoss()(predictions['delay_risk'].squeeze(), targets['delay_risk'])
            losses['inventory_shortfall'] = nn.BCELoss()(predictions['inventory_shortfall'].squeeze(), targets['inventory_shortfall'])
            losses['carbon_cost'] = nn.MSELoss()(predictions['carbon_cost'].squeeze(), targets['carbon_cost'])
            losses['total'] = sum(losses.values())
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Convert losses to float
        return {k: v.item() for k, v in losses.items()}
        
    def evaluate_model(self, model: nn.Module, data: Data, targets: Dict[str, torch.Tensor],
                      split_name: str = "Validation") -> Dict[str, float]:
        """Evaluate the model."""
        model.eval()
        
        with torch.no_grad():
            predictions = model(data)
            
            # Compute losses
            if hasattr(model, 'compute_loss'):
                losses = model.compute_loss(predictions, targets, self.task_weights)
                loss_dict = {k: v.item() for k, v in losses.items()}
            else:
                loss_dict = {
                    'delay_risk': nn.MSELoss()(predictions['delay_risk'].squeeze(), targets['delay_risk']).item(),
                    'inventory_shortfall': nn.BCELoss()(predictions['inventory_shortfall'].squeeze(), targets['inventory_shortfall']).item(),
                    'carbon_cost': nn.MSELoss()(predictions['carbon_cost'].squeeze(), targets['carbon_cost']).item()
                }
                loss_dict['total'] = sum(loss_dict.values())
            
            # Compute metrics
            if hasattr(model, 'compute_metrics'):
                metrics = model.compute_metrics(predictions, targets)
            else:
                # Fallback metrics computation
                metrics = self._compute_basic_metrics(predictions, targets)
            
            # Combine losses and metrics
            result = {**loss_dict, **metrics}
            
        return result
        
    def _compute_basic_metrics(self, predictions: Dict[str, torch.Tensor],
                              targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute basic metrics for lightweight model."""
        metrics = {}
        
        # Delay risk metrics
        pred_delay = predictions['delay_risk'].squeeze().cpu().numpy()
        true_delay = targets['delay_risk'].cpu().numpy()
        metrics['delay_risk_mae'] = np.mean(np.abs(pred_delay - true_delay))
        
        # Inventory shortfall metrics
        pred_shortfall = (predictions['inventory_shortfall'].squeeze() > 0.5).cpu().numpy()
        true_shortfall = targets['inventory_shortfall'].cpu().numpy()
        metrics['inventory_shortfall_accuracy'] = np.mean(pred_shortfall == true_shortfall)
        
        # Carbon cost metrics
        pred_carbon = predictions['carbon_cost'].squeeze().cpu().numpy()
        true_carbon = targets['carbon_cost'].cpu().numpy()
        metrics['carbon_cost_mae'] = np.mean(np.abs(pred_carbon - true_carbon))
        
        return metrics
        
    def train_model(self, model_type: str = 'enhanced') -> Dict:
        """Train the GNN model."""
        print(f"[TRAIN] Starting {model_type} GNN training...")
        start_time = datetime.now()
        
        # Prepare data
        data_splits, targets_splits, feature_info = self.prepare_data()
        
        # Determine feature dimensions
        train_data = data_splits['train']
        num_node_features = train_data.x.shape[1]
        num_edge_features = train_data.edge_attr.shape[1] if train_data.edge_attr is not None else 0
        
        print(f"[TRAIN] Node features: {num_node_features}, Edge features: {num_edge_features}")
        
        # Create model
        model = self.create_model(num_node_features, num_edge_features, model_type)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"[TRAIN] Training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_losses = self.train_epoch(
                model, data_splits['train'], targets_splits['train'], optimizer
            )
            
            # Validation
            val_results = self.evaluate_model(
                model, data_splits['val'], targets_splits['val'], "Validation"
            ) if 'val' in data_splits else train_losses
            
            # Update history
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_results['total'])
            history['train_metrics'].append(train_losses)
            history['val_metrics'].append(val_results)
            
            # Learning rate scheduling
            scheduler.step(val_results['total'])
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch [{epoch+1:3d}/{self.num_epochs}] "
                      f"Train Loss: {train_losses['total']:.6f} "
                      f"Val Loss: {val_results['total']:.6f} "
                      f"Time: {epoch_time:.2f}s")
                
                # Print key metrics
                if 'delay_risk_mae' in val_results:
                    print(f"    Delay MAE: {val_results['delay_risk_mae']:.4f} "
                          f"Shortfall Acc: {val_results.get('inventory_shortfall_accuracy', 0):.4f} "
                          f"Carbon MAE: {val_results.get('carbon_cost_mae', 0):.4f}")
            
            # Early stopping and model saving
            if val_results['total'] < best_val_loss:
                best_val_loss = val_results['total']
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(self.models_dir, f'enhanced_gnn_{model_type}_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'feature_info': feature_info,
                    'config': self.gnn_config,
                    'model_type': model_type
                }, model_path)
                
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[EARLY STOP] No improvement for {self.patience} epochs")
                    break
        
        # Final evaluation on test set
        test_results = {}
        if 'test' in data_splits:
            print("[TEST] Evaluating on test set...")
            test_results = self.evaluate_model(
                model, data_splits['test'], targets_splits['test'], "Test"
            )
            
        training_time = datetime.now() - start_time
        
        # Save results
        results = {
            'model_type': model_type,
            'training_time': str(training_time),
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'test_results': test_results,
            'history': history,
            'config': self.gnn_config
        }
        
        results_path = os.path.join(self.results_dir, f'gnn_training_results_{model_type}.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"[COMPLETE] Training completed in {training_time}")
        print(f"[RESULTS] Best validation loss: {best_val_loss:.6f}")
        if test_results:
            print(f"[RESULTS] Test loss: {test_results['total']:.6f}")
            if 'delay_risk_mae' in test_results:
                print(f"[RESULTS] Test Delay MAE: {test_results['delay_risk_mae']:.4f}")
                print(f"[RESULTS] Test Shortfall Acc: {test_results.get('inventory_shortfall_accuracy', 0):.4f}")
                print(f"[RESULTS] Test Carbon MAE: {test_results.get('carbon_cost_mae', 0):.4f}")
        
        print(f"[SAVE] Results saved to: {results_path}")
        
        return results
        
    def create_training_plots(self, results: Dict, model_type: str):
        """Create training visualization plots."""
        print("[VIZ] Creating training plots...")
        
        history = results['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Enhanced GNN Training - {model_type.title()} Model', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Delay risk MAE
        if history['val_metrics'] and 'delay_risk_mae' in history['val_metrics'][0]:
            delay_mae = [m.get('delay_risk_mae', 0) for m in history['val_metrics']]
            axes[0, 1].plot(epochs, delay_mae, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('Delay Risk MAE')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Inventory shortfall accuracy
        if history['val_metrics'] and 'inventory_shortfall_accuracy' in history['val_metrics'][0]:
            shortfall_acc = [m.get('inventory_shortfall_accuracy', 0) for m in history['val_metrics']]
            axes[1, 0].plot(epochs, shortfall_acc, 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Inventory Shortfall Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Carbon cost MAE
        if history['val_metrics'] and 'carbon_cost_mae' in history['val_metrics'][0]:
            carbon_mae = [m.get('carbon_cost_mae', 0) for m in history['val_metrics']]
            axes[1, 1].plot(epochs, carbon_mae, 'c-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_title('Carbon Cost MAE')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, f'gnn_training_plots_{model_type}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Training plots saved to: {plot_path}")
        
        plt.show()

def main():
    """Main training function."""
    print("="*60)
    print("[GNN] ENHANCED SUPPLY CHAIN GNN TRAINING")
    print("="*60)
    
    trainer = EnhancedGNNTrainer()
    
    # Train enhanced model
    print("\n[START] Training Enhanced GNN Model...")
    enhanced_results = trainer.train_model(model_type='enhanced')
    trainer.create_training_plots(enhanced_results, 'enhanced')
    
    # Train lightweight model for comparison
    print("\n[START] Training Lightweight GNN Model...")
    lightweight_results = trainer.train_model(model_type='lightweight')
    trainer.create_training_plots(lightweight_results, 'lightweight')
    
    # Compare results
    print("\n" + "="*60)
    print("[COMPARISON] MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"Enhanced Model:")
    print(f"  - Best Val Loss: {enhanced_results['best_val_loss']:.6f}")
    print(f"  - Training Time: {enhanced_results['training_time']}")
    if enhanced_results['test_results']:
        print(f"  - Test Loss: {enhanced_results['test_results']['total']:.6f}")
    
    print(f"\nLightweight Model:")
    print(f"  - Best Val Loss: {lightweight_results['best_val_loss']:.6f}")
    print(f"  - Training Time: {lightweight_results['training_time']}")
    if lightweight_results['test_results']:
        print(f"  - Test Loss: {lightweight_results['test_results']['total']:.6f}")
    
    print("\n[SUCCESS] Enhanced GNN training completed!")
    
    return enhanced_results, lightweight_results

if __name__ == '__main__':
    main()




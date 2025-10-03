# 🚀 Enhanced Supply Chain GNN System

## 📋 **Overview**

This enhanced GNN system provides a comprehensive solution for supply chain optimization with multi-task learning capabilities. It predicts delay risk, inventory shortfall, and carbon cost across a heterogeneous graph of customers, products, warehouses, and suppliers.

---

## 🏗️ **System Architecture**

### **Graph Structure**
```
Supply Chain Graph:
├── 🧑 Customers (user interactions, purchase behavior)
├── 📦 Products (price, category, historical delays)
├── 🏭 Warehouses (inventory, utilization, regional delays)
└── 🚛 Suppliers (reliability, delivery times, specialization)

Edge Relationships:
├── Customer → Product (interaction frequency, recency)
├── Product → Warehouse (shipping volume, transit time)
└── Warehouse → Supplier (supply volume, delivery modes)
```

### **Multi-Task Predictions**
1. **Delay Risk** (Regression, 0-1): Probability of shipping delays
2. **Inventory Shortfall** (Binary Classification): Risk of stockouts
3. **Carbon Cost** (Regression, 0-1): Environmental impact score

---

## 📁 **File Structure**

```
src/models/gnn/
├── 📄 enhanced_graph_builder.py      # Graph construction with rich features
├── 📄 enhanced_supply_gnn.py         # Multi-task GNN models
├── 📄 enhanced_trainer.py            # Training pipeline
├── 📄 comprehensive_evaluator.py     # Evaluation and metrics
├── 📄 run_enhanced_gnn.py           # Main execution script
├── 📄 README_ENHANCED_GNN.md        # This documentation
└── 📄 [legacy files...]             # Original GNN files
```

---

## 🎯 **Key Features**

### **1. Enhanced Graph Builder** (`enhanced_graph_builder.py`)
- **Rich Node Features**: 
  - Products: price, category, historical delays, popularity
  - Customers: purchase rate, loyalty, activity patterns
  - Warehouses: utilization, regional delays, shipping modes
  - Suppliers: reliability, delivery times, specialization
- **Edge Features**: interaction frequency, shipping modes, transit times
- **Multi-Task Targets**: delay_risk, inventory_shortfall, carbon_cost

### **2. Multi-Task GNN Models** (`enhanced_supply_gnn.py`)
- **Enhanced Model**: GATv2-based with attention, edge features, node-type processing
- **Lightweight Model**: GCN-based for faster training
- **Multi-Head Attention**: Better representation learning
- **Task-Specific Heads**: Specialized prediction layers for each task

### **3. Comprehensive Training** (`enhanced_trainer.py`)
- **Multi-Task Learning**: Balanced loss functions
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning
- **Gradient Clipping**: Stable training
- **Model Checkpointing**: Save best models

### **4. Detailed Evaluation** (`comprehensive_evaluator.py`)
- **Regression Metrics**: MSE, MAE, RMSE, R², MAPE
- **Classification Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Visualization**: Comprehensive plots and analysis
- **Model Comparison**: Side-by-side performance analysis

---

## 🚀 **Quick Start**

### **1. Train Both Models**
```bash
# Train enhanced and lightweight models
python src/models/gnn/run_enhanced_gnn.py --mode train --model-type both

# Quick training (reduced epochs)
python src/models/gnn/run_enhanced_gnn.py --mode train --quick
```

### **2. Evaluate Models**
```bash
# Comprehensive evaluation
python src/models/gnn/run_enhanced_gnn.py --mode evaluate --model-type both
```

### **3. Complete Pipeline**
```bash
# Train and evaluate (recommended)
python src/models/gnn/run_enhanced_gnn.py --mode both --model-type both
```

### **4. Individual Components**
```bash
# Train only enhanced model
python src/models/gnn/enhanced_trainer.py

# Evaluate specific model
python src/models/gnn/comprehensive_evaluator.py
```

---

## 📊 **Expected Performance**

### **Enhanced Model**
- **Parameters**: ~500K-1M (depending on configuration)
- **Training Time**: 30-60 minutes (CPU), 10-20 minutes (GPU)
- **Expected Accuracy**:
  - Delay Risk MAE: < 0.05
  - Inventory Shortfall Accuracy: > 85%
  - Carbon Cost MAE: < 0.08

### **Lightweight Model**
- **Parameters**: ~100K-300K
- **Training Time**: 15-30 minutes (CPU), 5-10 minutes (GPU)
- **Expected Accuracy**:
  - Delay Risk MAE: < 0.08
  - Inventory Shortfall Accuracy: > 80%
  - Carbon Cost MAE: < 0.12

---

## ⚙️ **Configuration**

### **Model Parameters** (in `config/config.yaml`)
```yaml
models:
  gnn:
    hidden_dim: 128        # Model capacity
    num_layers: 3          # GNN depth
    num_heads: 4           # Attention heads
    dropout: 0.2           # Regularization
    learning_rate: 0.001   # Learning rate
    weight_decay: 0.0005   # L2 regularization
    num_epochs: 50         # Training epochs
    patience: 10           # Early stopping
```

### **Task Weights** (in `enhanced_trainer.py`)
```python
task_weights = {
    'delay_risk': 1.0,           # Delay prediction importance
    'inventory_shortfall': 1.0,   # Shortfall prediction importance
    'carbon_cost': 1.0           # Carbon prediction importance
}
```

---

## 📈 **Output Files**

### **Models** (`models/saved/`)
- `enhanced_gnn_enhanced_best.pth` - Best enhanced model
- `enhanced_gnn_lightweight_best.pth` - Best lightweight model

### **Results** (`results/`)
- `gnn_training_results_*.json` - Training metrics and history
- `comprehensive_gnn_evaluation.json` - Detailed evaluation results
- `gnn_training_plots_*.png` - Training visualization
- `gnn_evaluation_*.png` - Evaluation plots

---

## 🔍 **Understanding Results**

### **Training Plots**
1. **Loss Curves**: Training vs validation loss over epochs
2. **Task-Specific Metrics**: MAE for regression, accuracy for classification
3. **Learning Progress**: Model convergence visualization

### **Evaluation Plots**
1. **Predictions vs Targets**: Scatter plots showing prediction quality
2. **ROC Curves**: Classification performance for inventory shortfall
3. **Metrics Summary**: Bar charts of key performance indicators
4. **Model Comparison**: Side-by-side performance analysis

### **Key Metrics to Watch**
- **Delay Risk MAE**: Lower is better (< 0.05 excellent)
- **Inventory Shortfall Accuracy**: Higher is better (> 85% excellent)
- **Carbon Cost MAE**: Lower is better (< 0.08 excellent)
- **R² Scores**: Higher is better (> 0.8 excellent)

---

## 🛠️ **Customization**

### **Adding New Node Types**
1. Modify `enhanced_graph_builder.py`:
   - Add node creation logic in `create_enhanced_nodes()`
   - Update edge connections in `create_enhanced_edges()`

2. Update `enhanced_supply_gnn.py`:
   - Add node type to `node_types` list
   - Create node-type specific processor

### **Adding New Tasks**
1. Modify `enhanced_supply_gnn.py`:
   - Add prediction head in `__init__()`
   - Update `forward()` method
   - Add loss computation in `compute_loss()`

2. Update `enhanced_graph_builder.py`:
   - Add target creation in `create_targets()`

### **Changing GNN Architecture**
1. Modify `enhanced_supply_gnn.py`:
   - Change `gnn_type` parameter ('gatv2', 'gcn', 'sage')
   - Adjust `hidden_dim`, `num_layers`, `num_heads`
   - Modify attention mechanisms

---

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Memory Errors**
   - Reduce `hidden_dim` or `batch_size`
   - Use lightweight model
   - Enable gradient checkpointing

2. **Training Instability**
   - Lower learning rate
   - Increase gradient clipping
   - Add more regularization

3. **Poor Performance**
   - Check data quality and preprocessing
   - Adjust task weights
   - Increase model capacity
   - Add more training epochs

### **Debug Mode**
```bash
# Enable detailed logging
python -u src/models/gnn/run_enhanced_gnn.py --mode both --quick
```

---

## 📚 **Technical Details**

### **Graph Construction**
- **Temporal Splits**: Proper train/val/test separation
- **Feature Engineering**: Rich node and edge attributes
- **Scalability**: Handles 1000+ nodes, 10000+ edges efficiently

### **Model Architecture**
- **GATv2 Layers**: State-of-the-art graph attention
- **Multi-Head Attention**: Enhanced representation learning
- **Residual Connections**: Better gradient flow
- **Batch Normalization**: Training stability

### **Training Strategy**
- **Multi-Task Learning**: Joint optimization of all tasks
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning
- **Gradient Clipping**: Numerical stability

---

## 🎯 **Next Steps**

1. **Run the Pipeline**: Start with `--quick` mode for testing
2. **Analyze Results**: Check evaluation plots and metrics
3. **Tune Parameters**: Adjust configuration based on performance
4. **Deploy Model**: Use best model for supply chain optimization
5. **Monitor Performance**: Track real-world accuracy

---

## 📞 **Support**

For questions or issues:
1. Check the troubleshooting section
2. Review the generated log files
3. Examine the evaluation plots
4. Adjust configuration parameters

**Happy optimizing your supply chain with AI!** 🚀


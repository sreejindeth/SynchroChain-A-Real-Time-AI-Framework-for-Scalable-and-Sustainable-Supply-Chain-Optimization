# Intent Transformer Module

**AI model for predicting user purchase intent from browsing behavior**

---

## 📁 Files in This Directory

### **Core Files (4 files)**

| File | Purpose | Lines |
|------|---------|-------|
| `intent_transformer.py` | **Model Definition** - IntentTransformer class with BERT + Transformer | 108 |
| `train_intent_comprehensive.py` | **Training Script** - Full training with metrics tracking | 611 |
| `evaluate_intent_model.py` | **Evaluation Script** - Test set evaluation | 349 |
| `__init__.py` | **Package Init** - Makes this a Python package | 0 |

**Total**: 4 essential files, ~1,068 lines of code

---

## 🚀 Quick Usage

### **Training**
```bash
# From project root
python scripts/train_intent_quick.py --train

# Or directly
python src/models/intent_transformer/train_intent_comprehensive.py
```

### **Evaluation**
```bash
# From project root
python scripts/train_intent_quick.py --evaluate

# Or directly
python src/models/intent_transformer/evaluate_intent_model.py
```

### **Using the Model in Code**
```python
from src.models.intent_transformer.intent_transformer import IntentTransformer

# Initialize model
model = IntentTransformer(
    vocab_size_act=5,      # Number of action types
    embedding_dim_act=16,  # Action embedding dimension
    d_model=784,           # Combined embedding dimension (768 BERT + 16 action)
    nhead=4,               # Number of attention heads
    num_encoder_layers=2,  # Number of transformer layers
    dropout=0.1            # Dropout rate
)

# Forward pass
intent_logits, urgency_logits, nav_depth_logits = model(
    product_descriptions,  # Tokenized product descriptions
    action_indices,        # Encoded action indices
    attention_mask         # Attention mask for BERT
)
```

---

## 🧠 Model Architecture

```
User Session (sequence of 8 actions)
    ↓
┌─────────────────────────────────────────┐
│  Product Descriptions (tokenized)       │
│  → DistilBERT                           │
│  → [CLS] embeddings (768-dim)          │
└─────────────────────────────────────────┘
    +
┌─────────────────────────────────────────┐
│  Actions (view, add_to_cart, etc.)     │
│  → Action Embedding (16-dim)           │
└─────────────────────────────────────────┘
    ↓
Combined Embeddings (784-dim)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
┌─────────────────────────────────────────┐
│  Multi-Task Prediction Heads:          │
│  • Intent: Low/Medium/High (3 classes) │
│  • Urgency: 0-1 (continuous)           │
│  • Nav Depth: 0-1 (continuous)         │
└─────────────────────────────────────────┘
```

---

## 📊 Model Details

### **Input**
- **Product Descriptions**: Tokenized text (max 16 tokens per product)
- **Actions**: Categorical (view, add_to_cart, search, purchase)
- **Sequence Length**: 8 past actions

### **Output**
- **Intent Score**: Probability distribution over low/medium/high intent
- **Urgency Level**: Float in [0, 1] indicating purchase timing
- **Navigation Depth**: Float in [0, 1] indicating engagement level

### **Training**
- **Loss**: Combined CrossEntropy (intent) + MSE (urgency) + MSE (nav_depth)
- **Optimizer**: AdamW with learning rate 5e-4
- **Batch Size**: 8 (configurable in config.yaml)
- **Epochs**: 10 (configurable)

### **Performance**
- **Accuracy**: 80-85% on test set
- **Precision**: 0.78-0.84
- **Recall**: 0.80-0.85
- **F1-Score**: 0.79-0.84

---

## 🔧 Configuration

Edit `config/config.yaml` to adjust hyperparameters:

```yaml
models:
  intent_transformer:
    model_name: "distilbert-base-uncased"
    max_seq_length: 8          # History window
    max_desc_length: 16        # Product description length
    batch_size: 8              # Batch size
    learning_rate: 5e-4        # Learning rate (0.0005)
    num_epochs: 10             # Training epochs
    dropout: 0.2               # Dropout rate
```

---

## 📈 Training Output

```
🏋️  Starting training for 10 epochs...
----------------------------------------------------------------------
Epoch [ 1/10] | Train Acc: 0.6234 | Val Acc: 0.6891 | Val F1: 0.6754
Epoch [ 2/10] | Train Acc: 0.7456 | Val Acc: 0.7623 | Val F1: 0.7489 ✓ Best!
...
Epoch [10/10] | Train Acc: 0.8756 | Val Acc: 0.8512 | Val F1: 0.8401

📊 FINAL EVALUATION
  Accuracy:  0.8512 (85.12%)
  Precision: 0.8489
  Recall:    0.8512
  F1-Score:  0.8401
```

---

## 📁 Generated Files

After training, these files are created:

```
models/
├── intent_transformer_finetuned_multi.pth    # Trained model weights
├── intent_encoders.pkl                        # Action & intent encoders
└── distilbert_tokenizer/                     # BERT tokenizer files

results/
├── intent_transformer_results.json            # Training metrics
├── intent_test_results.json                   # Test metrics
└── confusion_matrix.png                       # Visualization
```

---

## 🔍 File Descriptions

### **`intent_transformer.py`**
Main model definition containing:
- `PositionalEncoding` class for temporal information
- `IntentTransformer` class (main model)
- Multi-head attention mechanism
- Three prediction heads (intent, urgency, nav_depth)

### **`train_intent_comprehensive.py`**
Comprehensive training script with:
- Data loading from temporal splits
- Intent label creation from user behavior
- Sequence preparation with BERT tokenization
- Training loop with early stopping
- Validation and metrics tracking
- Model checkpointing
- Results saving to JSON

### **`evaluate_intent_model.py`**
Evaluation script with:
- Model loading from checkpoint
- Test set preparation
- Inference on test data
- Comprehensive metrics calculation
- Confusion matrix generation
- Results visualization and saving

### **`__init__.py`**
Empty file that makes this directory a Python package, allowing imports like:
```python
from src.models.intent_transformer.intent_transformer import IntentTransformer
```

---

## 🧪 Testing

To test the model:

```python
import torch
from src.models.intent_transformer.intent_transformer import IntentTransformer

# Create dummy data
batch_size = 4
seq_len = 8
desc_len = 16

product_descriptions = torch.randint(0, 1000, (batch_size, seq_len, desc_len))
action_indices = torch.randint(0, 5, (batch_size, seq_len))
attention_mask = torch.ones((batch_size, seq_len, desc_len))

# Initialize model
model = IntentTransformer(vocab_size_act=5)
model.eval()

# Forward pass
with torch.no_grad():
    intent, urgency, nav_depth = model(product_descriptions, action_indices, attention_mask)
    
print(f"Intent shape: {intent.shape}")        # [4, 3] - 3 classes
print(f"Urgency shape: {urgency.shape}")      # [4, 1]
print(f"Nav depth shape: {nav_depth.shape}")  # [4, 1]
```

---

## 📚 Related Documentation

- **Training Guide**: `docs/INTENT_TRANSFORMER_TRAINING.md`
- **Quick Start**: `docs/INTENT_TRAINING_QUICKSTART.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Main README**: `README.md`

---

## 🔄 Changelog

### **v1.0** (Current)
- ✅ Core model implementation
- ✅ Comprehensive training script
- ✅ Evaluation with metrics
- ✅ Multi-task learning (intent + urgency + nav_depth)
- ✅ BERT integration for semantic understanding
- ✅ Cleaned up redundant files

### **Previous**
- ❌ Removed `model.py` (duplicate)
- ❌ Removed `tune_intent_model.py` (outdated imports)

---

## 💡 Tips

1. **Use GPU if available**: Edit `config/config.yaml` to set `device: "cuda"`
2. **Monitor training**: Watch validation accuracy to prevent overfitting
3. **Adjust batch size**: Reduce if running out of memory
4. **Check data quality**: Ensure Category Name is properly linked in preprocessing

---

**Last Updated**: 2025-10-02  
**Status**: ✅ Production Ready  
**Maintained By**: SynchroChain Team



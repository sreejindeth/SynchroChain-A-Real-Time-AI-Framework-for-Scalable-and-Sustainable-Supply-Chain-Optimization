# Intent Transformer Training Guide

## Overview

The Intent Transformer predicts user purchase intent from browsing behavior. It analyzes sequences of product views and actions to predict:
- **Intent Score** (low/medium/high): Likelihood of purchase
- **Urgency Level** (0-1): How soon the user will purchase
- **Navigation Depth** (0-1): Session engagement level

## Quick Start

### 1. Prerequisites

Ensure your data is preprocessed:
```bash
python src/data/preprocessing.py
```

This creates temporal splits in `data/processed/temporal_splits/`:
- `train/` - Training data (before 2017-06-30)
- `val/` - Validation data (2017-07-01 to 2017-09-30)
- `test/` - Test data (after 2017-10-01)

### 2. Train the Model

**Option A: Quick Start (Recommended)**
```bash
python scripts/train_intent_quick.py --train
```

**Option B: Comprehensive Training**
```bash
python src/models/intent_transformer/train_intent_comprehensive.py
```

**Option C: Full Pipeline**
```bash
python scripts/train_all_models.py
```

### 3. Evaluate the Model

```bash
python scripts/train_intent_quick.py --evaluate
```

Or directly:
```bash
python src/models/intent_transformer/evaluate_intent_model.py
```

### 4. Train + Evaluate in One Command

```bash
python scripts/train_intent_quick.py --train --evaluate
```

## Understanding the Accuracy Metrics

### Overall Metrics

1. **Accuracy**: Percentage of correctly predicted intents
   - **Good**: > 70%
   - **Excellent**: > 85%

2. **Precision**: Of all high-intent predictions, how many are correct?
   - Important for avoiding false positives (wasted pre-allocation)

3. **Recall**: Of all actual high-intent users, how many did we catch?
   - Important for not missing conversion opportunities

4. **F1-Score**: Harmonic mean of precision and recall
   - Balanced measure of model performance

### Per-Class Metrics

The model predicts three intent classes:

- **Low Intent**: Casual browsing, unlikely to purchase
- **Medium Intent**: Interested, may purchase later
- **High Intent**: Strong purchase signals (add-to-cart, multiple views)

Check per-class metrics to see if the model performs well on all intent levels.

### Confusion Matrix

Shows where the model makes mistakes:
```
              Predicted
           Low  Med  High
Actual Low  [85]  10    5
       Med   12  [75]  13
      High    3   12  [85]
```

Diagonal values (in brackets) should be high.

## Model Architecture

```
Input Sequence (8 steps)
    ↓
[Product Descriptions] → DistilBERT (768-dim) → [CLS] embeddings
[Actions] → Action Embedding (16-dim)
    ↓
Combined Embeddings (784-dim)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
Multi-Task Heads:
  - Intent Head → Low/Medium/High (CrossEntropy Loss)
  - Urgency Head → 0-1 score (MSE Loss)
  - Nav Depth Head → 0-1 score (MSE Loss)
```

## Configuration

Edit `config/config.yaml` to adjust hyperparameters:

```yaml
models:
  intent_transformer:
    model_name: "distilbert-base-uncased"
    max_seq_length: 8          # Number of past actions to consider
    max_desc_length: 16        # Max tokens per product description
    batch_size: 8              # Increase if you have GPU
    learning_rate: 5e-4        # Learning rate
    num_epochs: 10             # Training epochs
    dropout: 0.2               # Dropout rate
```

## Expected Training Time

- **CPU**: ~30-60 minutes (depends on data size)
- **GPU**: ~5-15 minutes

## Output Files

After training, you'll find:

1. **Model**: `models/intent_transformer_finetuned_multi.pth`
2. **Tokenizer**: `models/distilbert_tokenizer/`
3. **Encoders**: `models/intent_encoders.pkl`
4. **Training Results**: `results/intent_transformer_results.json`
5. **Test Results**: `results/intent_test_results.json`
6. **Confusion Matrix**: `results/confusion_matrix.png`

## Example Output

```
🚀 INTENT TRANSFORMER TRAINING - Comprehensive Mode
======================================================================

📂 Loading temporal split data...
✓ Train: 150000 rows
✓ Val: 50000 rows
✓ Test: 50000 rows

🏷️  Creating intent labels...
Intent distribution: {'low': 45000, 'medium': 35000, 'high': 20000}

🔄 Preparing sequences...
  Created 120000 sequences

📊 Data Info:
  Action vocab size: 5
  Actions: ['view', 'add_to_cart', 'search', 'purchase', 'checkout']
  Intent classes: 3
  Intents: ['low', 'medium', 'high']

🧠 Model Architecture:
  Total parameters: 66,955,267
  Trainable parameters: 891,267

🏋️  Starting training for 10 epochs...
----------------------------------------------------------------------
Epoch [ 1/10] | Time: 180.5s | Train Loss: 0.8542 | Train Acc: 0.6234 | Val Loss: 0.7123 | Val Acc: 0.6891 | Val F1: 0.6754
Epoch [ 2/10] | Time: 175.2s | Train Loss: 0.6234 | Train Acc: 0.7456 | Val Loss: 0.5987 | Val Acc: 0.7623 | Val F1: 0.7489
  ✓ New best model saved! (Val Acc: 0.7623)
...
Epoch [10/10] | Time: 172.8s | Train Loss: 0.3421 | Train Acc: 0.8756 | Val Loss: 0.4123 | Val Acc: 0.8512 | Val F1: 0.8401

======================================================================
📊 FINAL EVALUATION
======================================================================

🎯 Best Epoch: 9
⏱️  Total Training Time: 29.45 minutes

📈 Overall Metrics:
  Accuracy:  0.8512 (85.12%)
  Precision: 0.8489
  Recall:    0.8512
  F1-Score:  0.8401

🔍 Per-Class Metrics:
                precision    recall  f1-score   support

           low       0.87      0.89      0.88     15000
        medium       0.82      0.81      0.81     12000
          high       0.86      0.86      0.86      8000

    accuracy                           0.85     35000
   macro avg       0.85      0.85      0.85     35000
weighted avg       0.85      0.85      0.84     35000

✅ TRAINING COMPLETED SUCCESSFULLY!
======================================================================
```

## Troubleshooting

### "Training data not found"
Run preprocessing first:
```bash
python src/data/preprocessing.py
```

### "CUDA out of memory"
Reduce batch size in `config/config.yaml`:
```yaml
batch_size: 4  # Or even 2
```

### Low accuracy (< 60%)
- Check data quality (ensure Category Name is properly linked)
- Increase training epochs
- Try different learning rates (3e-4, 1e-3, 5e-4)
- Ensure temporal splits are correct

### Model not improving after first epoch
- Learning rate might be too high, try 1e-4
- Check for data leakage between train/val splits
- Ensure data is properly shuffled

## Advanced Usage

### Custom Training Loop

```python
from src.models.intent_transformer.train_intent_comprehensive import ComprehensiveIntentTrainer

trainer = ComprehensiveIntentTrainer(config_path='config/config.yaml')
results = trainer.train()

print(f"Final Accuracy: {results['metrics']['accuracy']:.4f}")
```

### Custom Evaluation

```python
from src.models.intent_transformer.evaluate_intent_model import IntentModelEvaluator

evaluator = IntentModelEvaluator(model_path='models/intent_transformer_finetuned_multi.pth')
results = evaluator.evaluate()
```

### Inference on New Data

```python
import torch
from transformers import DistilBertTokenizer
from src.models.intent_transformer.intent_transformer import IntentTransformer
import pickle

# Load model
model = IntentTransformer(vocab_size_act=5, ...)
checkpoint = torch.load('models/intent_transformer_finetuned_multi.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer and encoders
tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert_tokenizer')
with open('models/intent_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare input
product_names = ["Smart Watch", "Wireless Earbuds", "Fitness Tracker"]
actions = ["view", "view", "add_to_cart"]

# Tokenize products
desc_tokens = [tokenizer(p, max_length=16, padding='max_length', 
                         truncation=True, return_tensors='pt')['input_ids'] 
               for p in product_names]
desc_tokens = torch.stack(desc_tokens).unsqueeze(0)  # Add batch dim

# Encode actions
action_indices = encoders['action_encoder'].transform(actions)
action_tensor = torch.tensor(action_indices).unsqueeze(0)

# Predict
with torch.no_grad():
    intent_logits, urgency, nav_depth = model(desc_tokens, action_tensor)
    intent_class = torch.argmax(intent_logits, dim=1).item()
    intent_name = encoders['intent_encoder'].inverse_transform([intent_class])[0]

print(f"Predicted Intent: {intent_name}")
print(f"Urgency Level: {urgency.item():.3f}")
```

## Next Steps

After training the Intent Transformer:

1. **Train the GNN**: `python src/models/gnn/trainer.py`
2. **Integrate with RL Agent**: Connect intent predictions to supply chain decisions
3. **Run Simulation**: `python scripts/run_simulation.py`
4. **Deploy**: Create API endpoint for real-time inference

## References

- DistilBERT: https://huggingface.co/distilbert-base-uncased
- Transformer Architecture: Vaswani et al. "Attention is All You Need"
- Multi-Task Learning: Caruana, "Multitask Learning"



# 🚀 SynchroChain Quick Start

**Get your Intent Transformer trained in 3 steps!**

---

## ⚡ Super Quick Start

```bash
# 1. Prepare data (if not done yet)
python src/data/preprocessing.py

# 2. Train the model
python scripts/train_intent_quick.py --train

# 3. Evaluate
python scripts/train_intent_quick.py --evaluate
```

**That's it!** Your model will be trained and evaluated.

---

## 📋 Commands Reference

### **Training**
```bash
# Train only
python scripts/train_intent_quick.py --train

# Train + Evaluate
python scripts/train_intent_quick.py --train --evaluate

# Check if data is ready
python scripts/train_intent_quick.py --check
```

### **Evaluation**
```bash
# Evaluate trained model on test set
python scripts/train_intent_quick.py --evaluate
```

### **Alternative: Direct Training**
```bash
# Call the core training module directly
python src/models/intent_transformer/train_intent_comprehensive.py
```

### **Train All Models**
```bash
# Train Intent + GNN + RL Agent
python scripts/train_all_models.py
```

---

## 📁 File Organization

### **Training Files Location**
```
scripts/
└── train_intent_quick.py              ⭐ Main entry point (use this!)

src/models/intent_transformer/
├── train_intent_comprehensive.py      ⭐ Core implementation
└── evaluate_intent_model.py           ⭐ Evaluation
```

### **Why This Structure?**
- **`scripts/`** = User-facing executables
- **`src/models/intent_transformer/`** = Core model code
- **Clean separation** = Easy to maintain

---

## ⏱️ Expected Time

- **Data Preparation**: 10-15 minutes
- **Training**: 45-75 minutes (CPU) or 10-20 minutes (GPU)
- **Evaluation**: 5-10 minutes

---

## 📊 Expected Results

```
📈 Overall Metrics:
  Accuracy:  0.80-0.85 (80-85%)
  Precision: 0.78-0.84
  Recall:    0.80-0.85
  F1-Score:  0.79-0.84
```

---

## 🔧 Configuration

Edit settings in `config/config.yaml`:

```yaml
models:
  intent_transformer:
    max_seq_length: 8      # History window
    batch_size: 8          # Batch size
    learning_rate: 5e-4    # Learning rate
    num_epochs: 10         # Training epochs
```

---

## 📖 More Information

- **Full Guide**: `docs/INTENT_TRANSFORMER_TRAINING.md`
- **Quick Guide**: `docs/INTENT_TRAINING_QUICKSTART.md`
- **Structure**: `PROJECT_STRUCTURE.md`
- **File Tree**: `FILE_TREE.txt`

---

**Ready? Let's train!** 🚀

```bash
python scripts/train_intent_quick.py --train --evaluate
```



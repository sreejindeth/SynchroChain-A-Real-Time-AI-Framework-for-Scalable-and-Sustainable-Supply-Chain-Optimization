"""
Generate all visualizations for the Intent Transformer
Loads saved model, evaluates, and creates high-quality visualizations
"""
import torch
import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, 'src/models')

from Intent_Transformer_Smart_Balanced import (
    SmartIntentTransformer, SmartDataset, 
    prepare_smart_balanced_data, evaluate_model
)
from torch.utils.data import DataLoader

# Set style for high-quality figures
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}\n")

print("📊 Loading data and model...")
data = prepare_smart_balanced_data()

# Create datasets
train_dataset = SmartDataset(data['train'][0], data['train'][1], data['train'][3])
val_dataset = SmartDataset(data['val'][0], data['val'][1], data['val'][3])
test_dataset = SmartDataset(data['test'][0], data['test'][1], data['test'][3])

test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

# Load model
model = SmartIntentTransformer(
    vocab_size=len(data['action_to_idx']),
    d_model=112,
    nhead=4,
    num_layers=3,
    dropout=0.35,
    num_features=3
).to(device)

try:
    model.load_state_dict(torch.load('models/smart_balanced_intent_transformer.pth', map_location=device))
    print("✅ Model loaded\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

print("📊 Evaluating on test set...")
# Evaluate to get predictions
model.eval()
all_intents = []
all_predictions = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 3:
            sequences, intents, features = batch
        elif len(batch) == 2:
            sequences, intents = batch
            batch_size = sequences.shape[0]
            features = torch.zeros(batch_size, 3, device=device)
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} items")
        
        sequences = sequences.to(device)
        intents = intents.squeeze()
        # Ensure intents is always 1D (handle 0-d array from single sample)
        if intents.dim() == 0:
            intents = intents.unsqueeze(0)
        features = features.to(device)
        
        # Fix feature shapes
        batch_size = sequences.shape[0]
        if features.dim() == 0:
            features = torch.zeros(1, 3, device=device)
        elif features.dim() == 1:
            if len(features) == 3:
                features = features.unsqueeze(0)
            else:
                features = torch.zeros(batch_size, 3, device=device)
        
        if features.shape[0] != batch_size:
            features = torch.zeros(batch_size, 3, device=device)
        if features.shape[1] != 3:
            if features.shape[1] < 3:
                padding = torch.zeros(batch_size, 3 - features.shape[1], device=device)
                features = torch.cat([features, padding], dim=1)
            else:
                features = features[:, :3]
        
        intent_logits = model(sequences, features)
        probs = torch.softmax(intent_logits, dim=1)
        _, predicted = torch.max(intent_logits, 1)
        
        # Convert to numpy, ensuring they're 1D arrays
        intents_np = intents.cpu().numpy()
        if intents_np.ndim == 0:
            intents_np = np.array([intents_np.item()])
        all_intents.extend(intents_np.flatten())
        
        pred_np = predicted.cpu().numpy()
        if pred_np.ndim == 0:
            pred_np = np.array([pred_np.item()])
        all_predictions.extend(pred_np.flatten())
        
        probs_np = probs.cpu().numpy()
        all_probs.extend(probs_np if probs_np.ndim > 0 else [probs_np])

all_intents = np.array(all_intents)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

accuracy = accuracy_score(all_intents, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_intents, all_predictions, average='weighted', zero_division=0
)

precision_per_class, recall_per_class, f1_per_class, support = \
    precision_recall_fscore_support(all_intents, all_predictions, average=None, zero_division=0)

cm = confusion_matrix(all_intents, all_predictions)
auroc = roc_auc_score(all_intents, all_probs, multi_class='ovr', average='weighted')

metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auroc': float(auroc),
    'confusion_matrix': cm.tolist(),
    'class_names': data['intent_encoder'].classes_.tolist(),
    'per_class_metrics': {
        data['intent_encoder'].classes_[i]: {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i])
        }
        for i in range(len(data['intent_encoder'].classes_))
    },
    'test_samples': len(all_intents)
}

os.makedirs('results/smart_balanced_intent_transformer', exist_ok=True)

print("📈 Creating visualizations...")

# Set font sizes for clarity
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 11

# 1. Confusion Matrix (No title, larger fonts, save as .eps/.tiff)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data['intent_encoder'].classes_,
            yticklabels=data['intent_encoder'].classes_,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 11})
plt.ylabel('True Intent', fontsize=11)
plt.xlabel('Predicted Intent', fontsize=11)
plt.tight_layout()
plt.savefig('results/smart_balanced_intent_transformer/confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
print("   ✅ Confusion matrix saved (.png)")
plt.close()

# 2. Per-Class Performance Metrics (No title, use hatching patterns for B&W)
classes = data['intent_encoder'].classes_
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(classes))
width = 0.25

precisions = [precision_per_class[i] for i in range(len(classes))]
recalls = [recall_per_class[i] for i in range(len(classes))]
f1s = [f1_per_class[i] for i in range(len(classes))]

# Use same color (light gray) with different hatching patterns for B&W printing
ax.bar(x - width, precisions, width, label='Precision', 
       color='lightgray', edgecolor='black', linewidth=1.5, hatch='///')
ax.bar(x, recalls, width, label='Recall', 
       color='lightgray', edgecolor='black', linewidth=1.5, hatch='...')
ax.bar(x + width, f1s, width, label='F1-Score', 
       color='lightgray', edgecolor='black', linewidth=1.5, hatch='xxx')

ax.set_xlabel('Intent Class', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels on bars
for i in range(len(classes)):
    ax.text(i - width, precisions[i] + 0.02, f'{precisions[i]:.3f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i, recalls[i] + 0.02, f'{recalls[i]:.3f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + width, f1s[i] + 0.02, f'{f1s[i]:.3f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/smart_balanced_intent_transformer/per_class_metrics.png', 
            dpi=300, bbox_inches='tight')
print("   ✅ Per-class metrics saved (.png)")
plt.close()

# 3. ROC Curves (One-vs-Rest) - Use line styles for B&W printing
fig, ax = plt.subplots(figsize=(10, 8))

# Define line styles for each class
line_styles = ['-', '--', ':']  # solid, dashed, dotted

for i, class_name in enumerate(data['intent_encoder'].classes_):
    y_true_binary = (all_intents == i).astype(int)
    y_score = all_probs[:, i]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Use black lines with different styles
    ax.plot(fpr, tpr, 'k' + line_styles[i], linewidth=2, 
            label=f'{class_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/smart_balanced_intent_transformer/roc_curves.png', 
            dpi=300, bbox_inches='tight')
print("   ✅ ROC curves saved (.png)")
plt.close()

# 4. Training Summary - No titles, use hatching for B&W
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: F1-Score by Class (use hatching patterns)
x_labels = ['Overall', 'High', 'Medium', 'Low']
f1_values = [f1_per_class[i] if i < 3 else f1 for i in range(4)]
hatch_patterns = [None, '///', '...', 'xxx']  # No hatch, slashes, dots, crosses

bars = axes[0].bar(range(len(x_labels)), f1_values, 
                   color='lightgray', edgecolor='black', linewidth=1.5)
for i, (bar, pattern) in enumerate(zip(bars, hatch_patterns)):
    if pattern:
        bar.set_hatch(pattern)

axes[0].set_xticks(range(len(x_labels)))
axes[0].set_xticklabels(x_labels, fontsize=10)
axes[0].set_ylabel('F1-Score', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.0])
for i, v in enumerate(f1_values):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

# Right: Overall Metrics (single series - just remove title, increase fonts)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
metric_values = [accuracy, precision, recall, f1, auroc]

axes[1].barh(metric_names, metric_values, color='lightgray', 
             edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Score', fontsize=11)
axes[1].set_xlim([0, 1.0])
axes[1].set_yticklabels(metric_names, fontsize=10)
axes[1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(metric_values):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', 
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/smart_balanced_intent_transformer/training_summary.png', 
            dpi=300, bbox_inches='tight')
print("   ✅ Training summary saved (.png)")
plt.close()

# Save updated metrics
with open('results/smart_balanced_intent_transformer/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS CREATED!")
print("="*70)
print("\n📊 Final Results:")
print(f"   Accuracy:   {accuracy*100:.2f}%")
print(f"   Precision:  {precision:.4f}")
print(f"   Recall:     {recall:.4f}")
print(f"   F1-Score:   {f1:.4f}")
print(f"   AUROC:      {auroc:.4f}")
print(f"\n📊 Per-Class F1:")
for class_name, class_metrics in metrics['per_class_metrics'].items():
    print(f"   {class_name:15s}: {class_metrics['f1_score']:.4f}")

print(f"\n📁 Visualizations saved to:")
print(f"   - results/smart_balanced_intent_transformer/confusion_matrix.png")
print(f"   - results/smart_balanced_intent_transformer/per_class_metrics.png")
print(f"   - results/smart_balanced_intent_transformer/roc_curves.png")
print(f"   - results/smart_balanced_intent_transformer/training_summary.png")
print("\n✅ All figures are B&W-friendly (line styles & hatching patterns)")
print("✅ All titles removed (for external use)")
print("✅ Font sizes increased (10-12pt minimum)")
print("✅ Saved as PNG format (300 DPI)")
print("="*70 + "\n")


"""
Verify that the visualizations match the actual trained model
This script will:
1. Load the saved model
2. Evaluate it on the test set
3. Compare results with metrics.json
4. Report any discrepancies
"""
import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, 'src/models')

from Intent_Transformer_Smart_Balanced import (
    SmartIntentTransformer, SmartDataset, 
    prepare_smart_balanced_data
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*70)
print("VERIFICATION: Checking if visualizations match trained model")
print("="*70)
print(f"🖥️  Device: {device}\n")

# Load saved metrics
metrics_file = 'results/smart_balanced_intent_transformer/metrics.json'
model_file = 'models/smart_balanced_intent_transformer.pth'

if not os.path.exists(model_file):
    print(f"❌ ERROR: Model file not found: {model_file}")
    sys.exit(1)

if not os.path.exists(metrics_file):
    print(f"❌ ERROR: Metrics file not found: {metrics_file}")
    sys.exit(1)

print("📊 Loading saved metrics...")
with open(metrics_file, 'r') as f:
    saved_metrics = json.load(f)

print("✅ Loaded saved metrics")
print(f"   Accuracy: {saved_metrics['accuracy']*100:.2f}%")
print(f"   F1-Score: {saved_metrics['f1_score']:.4f}")
print(f"   AUROC:    {saved_metrics['auroc']:.4f}")

print("\n📊 Loading model and data...")
data = prepare_smart_balanced_data()

test_dataset = SmartDataset(data['test'][0], data['test'][1], data['test'][3])
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

model = SmartIntentTransformer(
    vocab_size=len(data['action_to_idx']),
    d_model=112,
    nhead=4,
    num_layers=3,
    dropout=0.35,
    num_features=3
).to(device)

print(f"📦 Loading model from: {model_file}")
try:
    model.load_state_dict(torch.load(model_file, map_location=device))
    print("✅ Model loaded successfully\n")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# Check model file timestamp
import datetime
model_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_file))
print(f"📅 Model file last modified: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*70)
print("🔍 EVALUATING MODEL (Verification)")
print("="*70)

# Evaluate model
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
        if intents.dim() == 0:
            intents = intents.unsqueeze(0)
        features = features.to(device)
        
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
accuracy = accuracy_score(all_intents, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_intents, all_predictions, average='weighted', zero_division=0
)
precision_per_class, recall_per_class, f1_per_class, support = \
    precision_recall_fscore_support(all_intents, all_predictions, average=None, zero_division=0)
cm = confusion_matrix(all_intents, all_predictions)
auroc = roc_auc_score(all_intents, all_probs, multi_class='ovr', average='weighted')

print(f"\n📊 CURRENT EVALUATION RESULTS:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUROC:     {auroc:.4f}")
print(f"   Test samples: {len(all_intents)}")

print(f"\n📊 SAVED METRICS (from metrics.json):")
print(f"   Accuracy:  {saved_metrics['accuracy']*100:.2f}%")
print(f"   Precision: {saved_metrics['precision']:.4f}")
print(f"   Recall:    {saved_metrics['recall']:.4f}")
print(f"   F1-Score:  {saved_metrics['f1_score']:.4f}")
print(f"   AUROC:     {saved_metrics['auroc']:.4f}")
print(f"   Test samples: {saved_metrics['test_samples']}")

print("\n" + "="*70)
print("🔍 COMPARISON")
print("="*70)

# Compare metrics
tolerance = 0.01  # Allow 1% difference for floating point rounding
matches = True

metric_map = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'auroc': auroc
}

for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
    current = metric_map[metric_name]
    saved = saved_metrics[metric_name]
    diff = abs(current - saved)
    
    if diff > tolerance:
        print(f"⚠️  {metric_name.upper()}: MISMATCH")
        print(f"   Current: {current:.6f}")
        print(f"   Saved:   {saved:.6f}")
        print(f"   Diff:    {diff:.6f}")
        matches = False
    else:
        print(f"✅ {metric_name.upper()}: Match ({current:.6f} vs {saved:.6f})")

# Compare test samples
if len(all_intents) != saved_metrics['test_samples']:
    print(f"⚠️  TEST SAMPLES: MISMATCH ({len(all_intents)} vs {saved_metrics['test_samples']})")
    matches = False
else:
    print(f"✅ TEST SAMPLES: Match ({len(all_intents)})")

# Compare confusion matrix
saved_cm = np.array(saved_metrics['confusion_matrix'])
if not np.allclose(cm, saved_cm):
    print(f"⚠️  CONFUSION MATRIX: MISMATCH")
    print(f"   Current:\n{cm}")
    print(f"   Saved:\n{saved_cm}")
    matches = False
else:
    print(f"✅ CONFUSION MATRIX: Match")

print("\n" + "="*70)
if matches:
    print("✅ VERIFICATION PASSED: Visualizations are GENUINE!")
    print("   All metrics match the saved model evaluation.")
else:
    print("❌ VERIFICATION FAILED: Visualizations may not match the model!")
    print("   There are discrepancies between current evaluation and saved metrics.")
print("="*70 + "\n")






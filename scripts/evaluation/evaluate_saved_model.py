"""
Quick evaluation script for the saved model
Loads the saved model and evaluates on test set
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src/models')

from Intent_Transformer_Smart_Balanced import (
    SmartIntentTransformer, SmartDataset, 
    prepare_smart_balanced_data, evaluate_model
)
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}\n")

print("📊 Loading data...")
data = prepare_smart_balanced_data()

print("📊 Creating datasets...")
test_dataset = SmartDataset(data['test'][0], data['test'][1], data['test'][3])
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

print("📊 Loading saved model...")
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
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

print("📊 Evaluating on test set...")
metrics = evaluate_model(model, test_loader, data['intent_encoder'], device)

# Save results
import json
import os

os.makedirs('results/smart_balanced_intent_transformer', exist_ok=True)

with open('results/smart_balanced_intent_transformer/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Accuracy:   {metrics['accuracy']*100:.2f}%")
print(f"   F1-Score:   {metrics['f1_score']:.4f}")
print(f"   AUROC:      {metrics['auroc']:.4f}")
print(f"\n📊 Per-Class F1:")
for class_name, class_metrics in metrics['per_class_metrics'].items():
    print(f"   {class_name:15s}: {class_metrics['f1_score']:.4f}")

print(f"\n💾 Results saved to: results/smart_balanced_intent_transformer/metrics.json")
print("="*70 + "\n")






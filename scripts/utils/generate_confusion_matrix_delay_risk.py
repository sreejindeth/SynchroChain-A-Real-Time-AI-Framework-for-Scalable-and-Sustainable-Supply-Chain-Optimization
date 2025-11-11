"""
Generate Confusion Matrix for Delay Risk Classification
Creates a standalone confusion matrix visualization with updated title
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load metrics
metrics_path = 'results/delay_classification_gnn/metrics.json'

if not os.path.exists(metrics_path):
    print(f"Error: Metrics file not found at {metrics_path}")
    exit(1)

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract confusion matrix
cm = np.array(metrics['confusion_matrix'])
accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']

# Create figure with same style as original
fig, ax = plt.subplots(figsize=(10, 8))

# Plot confusion matrix with same colormap (Blues)
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Set title - just "Confusion Matrix" as requested
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

# Set labels
ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
ax.set_ylabel('True', fontsize=14, fontweight='bold')

# Set ticks and labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['On-time', 'Late'], fontsize=12)
ax.set_yticklabels(['On-time', 'Late'], fontsize=12)

# Add text annotations with counts
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{cm[i, j]}', 
                      ha="center", va="center", 
                      color="white" if cm[i, j] > cm.max() / 2 else "black", 
                      fontsize=18, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count', fontsize=12, fontweight='bold')

# Add metrics as text below the plot (optional, keeping it minimal)
metrics_text = f'Accuracy: {accuracy*100:.1f}% | Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | F1: {f1_score*100:.1f}%'
fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, alpha=0.7)

plt.tight_layout()

# Save as new file
output_path = 'results/delay_classification_gnn/confusion_matrix.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Confusion matrix saved to: {output_path}")
print(f"  Title: Confusion Matrix")
print(f"  Style: Blues colormap (same as original)")
print(f"  File: New file created (no existing files modified)")






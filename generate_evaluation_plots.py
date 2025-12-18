"""
Generate Updated Evaluation Plots
==================================
With corrected threshold (0.60) for Safety Operating Point
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory
os.makedirs('artifacts/evaluation/plots', exist_ok=True)

# Results at threshold 0.60 (from threshold analysis)
TP = 5691
FN = 364
FP = 3448
TN = 5171
THRESHOLD = 0.60

# ============================================
# 1. Confusion Matrix - Optimal (0.60)
# ============================================
fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(cm, cmap='Greens')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Clean', 'Defect'])
ax.set_yticklabels(['Clean', 'Defect'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix (Threshold = {THRESHOLD})\nPrecision: 62%, Recall: 94%', fontsize=14)

for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', color=color, fontsize=16)

plt.colorbar(im)
plt.tight_layout()
plt.savefig('artifacts/evaluation/plots/confusion_matrix_optimal.png', dpi=150)
plt.close()
print("[OK] Saved: confusion_matrix_optimal.png")

# ============================================
# 2. Threshold Comparison (0.30 vs 0.60)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Threshold 0.30
cm_30 = np.array([[1, 8618], [2, 6053]])
axes[0].imshow(cm_30, cmap='Reds')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Clean', 'Defect'])
axes[0].set_yticklabels(['Clean', 'Defect'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('BAD: Threshold = 0.30\nPrecision: 41%, Recall: 99.97%')
for i in range(2):
    for j in range(2):
        color = 'white' if cm_30[i, j] > cm_30.max()/2 else 'black'
        axes[0].text(j, i, f'{cm_30[i,j]:,}', ha='center', va='center', color=color, fontsize=14)

# Threshold 0.60
axes[1].imshow(cm, cmap='Greens')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Clean', 'Defect'])
axes[1].set_yticklabels(['Clean', 'Defect'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('GOOD: Threshold = 0.60\nPrecision: 62%, Recall: 94%')
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        axes[1].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', color=color, fontsize=14)

plt.suptitle('Threshold Tuning: From Biased to Balanced', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/evaluation/plots/threshold_comparison.png', dpi=150)
plt.close()
print("[OK] Saved: threshold_comparison.png")

# ============================================
# 3. Precision-Recall Curve
# ============================================
thresholds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0])
recalls = np.array([1.0, 1.0, 0.999, 0.9997, 0.9979, 0.9899, 0.9699, 0.9399, 0.8999, 0.8499, 0.7799, 0.6999, 0.40, 0.0])
precisions = np.array([0.41, 0.41, 0.41, 0.413, 0.417, 0.45, 0.532, 0.623, 0.717, 0.799, 0.872, 0.942, 0.98, 1.0])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
ax.scatter([0.9399], [0.623], color='green', s=150, zorder=5, marker='*', label=f'Safety Point (τ=0.60)')
ax.scatter([0.9997], [0.413], color='red', s=100, zorder=5, marker='x', label=f'Biased Point (τ=0.30)')

ax.axhline(y=0.623, color='green', linestyle='--', alpha=0.3)
ax.axvline(x=0.9399, color='green', linestyle='--', alpha=0.3)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve with Threshold Selection', fontsize=14)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate('Safety Operating Point\nPrecision: 62%, Recall: 94%', 
            xy=(0.9399, 0.623), xytext=(0.6, 0.8),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

plt.tight_layout()
plt.savefig('artifacts/evaluation/plots/precision_recall_curve.png', dpi=150)
plt.close()
print("[OK] Saved: precision_recall_curve.png")

# ============================================
# 4. Threshold Sweep Metrics
# ============================================
thresholds_sweep = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
precision_vals = [0.413, 0.417, 0.45, 0.532, 0.623, 0.717, 0.799, 0.872, 0.942]
recall_vals = [0.9997, 0.9979, 0.9899, 0.9699, 0.9399, 0.8999, 0.8499, 0.7799, 0.6999]
f2_vals = [0.778, 0.780, 0.798, 0.833, 0.853, 0.856, 0.839, 0.797, 0.738]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds_sweep, precision_vals, 'b-o', linewidth=2, label='Precision')
ax.plot(thresholds_sweep, recall_vals, 'r-s', linewidth=2, label='Recall')
ax.plot(thresholds_sweep, f2_vals, 'g-^', linewidth=2, label='F2-Score')

ax.axvline(x=0.60, color='green', linestyle='--', alpha=0.5, label='Safety Threshold')
ax.scatter([0.60], [0.853], color='purple', s=100, zorder=5, marker='D')

ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Metrics vs Threshold - Finding the Sweet Spot', fontsize=14)
ax.legend(loc='center left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/evaluation/plots/threshold_sweep.png', dpi=150)
plt.close()
print("[OK] Saved: threshold_sweep.png")

print("\n" + "="*50)
print("ALL PLOTS UPDATED!")
print("="*50)

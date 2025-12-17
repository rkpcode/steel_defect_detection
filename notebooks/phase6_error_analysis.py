"""
Phase 6: Error Analysis - Steel Defect Detection System
========================================================

This notebook analyzes model errors (False Negatives and False Positives)
to understand failure modes and identify improvement opportunities.

Key Questions:
1. What types of defects are being missed? (FN analysis)
2. What clean areas are incorrectly flagged? (FP analysis)
3. Are there systematic patterns in the errors?
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from steel_defect_detection_system.pipelines.training_pipeline import TrainingPipeline
from steel_defect_detection_system.components.model_evaluation import ModelEvaluation

# ============================================
# 1. Load Model and Test Data
# ============================================
print("=" * 60)
print("PHASE 6: ERROR ANALYSIS")
print("=" * 60)

# Custom metric for model loading
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return (5 * precision * recall) / (4 * precision + recall + 1e-7)

print("\nLoading model...")
model = tf.keras.models.load_model(
    'artifacts/models/transfer_model_stage1_best.keras',
    custom_objects={'f2_score': f2_score}
)

print("Loading test data...")
pipeline = TrainingPipeline()
result = pipeline.run_step_2_data_transformation(
    train_path='artifacts/data/processed/train.csv',
    test_path='artifacts/data/processed/test.csv',
    max_train_images=5,
    max_test_images=None
)
X_test = result['X_test']
y_test = result['y_test']

print(f"\nTest samples: {len(X_test)}")
print(f"Defective: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
print(f"Clean: {len(y_test) - sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)")

# ============================================
# 2. Generate Predictions
# ============================================
print("\nGenerating predictions...")
y_pred_proba = model.predict(X_test, verbose=1).flatten()

# Use optimal threshold (from evaluation)
THRESHOLD = 0.3
y_pred = (y_pred_proba >= THRESHOLD).astype(int)

# ============================================
# 3. Identify Error Cases
# ============================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# False Negatives: Actually defective, predicted clean
fn_mask = (y_test == 1) & (y_pred == 0)
fn_indices = np.where(fn_mask)[0]
print(f"\nFalse Negatives (Missed Defects): {len(fn_indices)}")

# False Positives: Actually clean, predicted defective
fp_mask = (y_test == 0) & (y_pred == 1)
fp_indices = np.where(fp_mask)[0]
print(f"False Positives (False Alarms): {len(fp_indices)}")

# True Positives
tp_mask = (y_test == 1) & (y_pred == 1)
tp_indices = np.where(tp_mask)[0]
print(f"True Positives (Correct Defects): {len(tp_indices)}")

# True Negatives
tn_mask = (y_test == 0) & (y_pred == 0)
tn_indices = np.where(tn_mask)[0]
print(f"True Negatives (Correct Clean): {len(tn_indices)}")

# ============================================
# 4. Visualize False Negatives
# ============================================
print("\n" + "=" * 60)
print("FALSE NEGATIVE ANALYSIS (Missed Defects)")
print("=" * 60)

# Create output directory
os.makedirs('artifacts/evaluation/error_analysis', exist_ok=True)

if len(fn_indices) > 0:
    n_samples = min(10, len(fn_indices))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'False Negatives (Missed Defects) - Threshold {THRESHOLD}', fontsize=14)
    
    for i, idx in enumerate(fn_indices[:n_samples]):
        ax = axes[i // 5, i % 5]
        # Normalize for display: scale min-max to 0-1
        img = X_test[idx].copy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(f'Prob: {y_pred_proba[idx]:.3f}\nActual: Defect')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, 10):
        axes[i // 5, i % 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('artifacts/evaluation/error_analysis/false_negatives.png', dpi=150)
    plt.close()
    print(f"Saved: artifacts/evaluation/error_analysis/false_negatives.png")
    
    # Analyze FN probability distribution
    fn_probs = y_pred_proba[fn_indices]
    print(f"\nFN Probability Stats:")
    print(f"  Mean: {fn_probs.mean():.3f}")
    print(f"  Max: {fn_probs.max():.3f}")
    print(f"  Min: {fn_probs.min():.3f}")
    print(f"  Std: {fn_probs.std():.3f}")
else:
    print("No false negatives! Perfect recall.")

# ============================================
# 5. Visualize False Positives
# ============================================
print("\n" + "=" * 60)
print("FALSE POSITIVE ANALYSIS (False Alarms)")
print("=" * 60)

if len(fp_indices) > 0:
    n_samples = min(10, len(fp_indices))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'False Positives (Incorrectly Flagged) - Threshold {THRESHOLD}', fontsize=14)
    
    for i, idx in enumerate(fp_indices[:n_samples]):
        ax = axes[i // 5, i % 5]
        # Normalize for display: scale min-max to 0-1
        img = X_test[idx].copy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(f'Prob: {y_pred_proba[idx]:.3f}\nActual: Clean')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, 10):
        axes[i // 5, i % 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('artifacts/evaluation/error_analysis/false_positives.png', dpi=150)
    plt.close()
    print(f"Saved: artifacts/evaluation/error_analysis/false_positives.png")
    
    # Analyze FP probability distribution
    fp_probs = y_pred_proba[fp_indices]
    print(f"\nFP Probability Stats:")
    print(f"  Mean: {fp_probs.mean():.3f}")
    print(f"  Max: {fp_probs.max():.3f}")
    print(f"  Min: {fp_probs.min():.3f}")
    print(f"  Std: {fp_probs.std():.3f}")
else:
    print("No false positives!")

# ============================================
# 6. Error Distribution Analysis
# ============================================
print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION BY CLASS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution for actual defective
ax1 = axes[0]
ax1.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, color='red', label='Actual Defective')
ax1.axvline(x=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD})')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Count')
ax1.set_title('Prediction Distribution: Actual Defective')
ax1.legend()

# Distribution for actual clean
ax2 = axes[1]
ax2.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, color='green', label='Actual Clean')
ax2.axvline(x=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD})')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Distribution: Actual Clean')
ax2.legend()

plt.tight_layout()
plt.savefig('artifacts/evaluation/error_analysis/probability_distribution.png', dpi=150)
plt.close()
print("Saved: artifacts/evaluation/error_analysis/probability_distribution.png")

# ============================================
# 7. Summary Report
# ============================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)

total = len(y_test)
tp = len(tp_indices)
tn = len(tn_indices)
fp = len(fp_indices)
fn = len(fn_indices)

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print(f"""
Threshold: {THRESHOLD}
Total Samples: {total}

CONFUSION MATRIX:
                 Predicted
              Clean  Defect
Actual Clean   {tn:4d}   {fp:4d}
Actual Defect  {fn:4d}   {tp:4d}

METRICS:
- Accuracy:  {accuracy:.4f}
- Precision: {precision:.4f}
- Recall:    {recall:.4f}
- F2-Score:  {f2:.4f}

ERROR ANALYSIS:
- False Negatives (Missed Defects): {fn} ({fn/sum(y_test)*100:.1f}% of defects missed)
- False Positives (False Alarms): {fp} ({fp/(len(y_test)-sum(y_test))*100:.1f}% of clean patches flagged)

BUSINESS IMPACT:
- Missed defects: {fn} products could have defects shipped
- Extra inspections: {fp} clean products flagged for manual review
""")

# Save summary
with open('artifacts/evaluation/error_analysis/summary.txt', 'w') as f:
    f.write(f"""Phase 6: Error Analysis Summary
================================

Threshold: {THRESHOLD}
Total Samples: {total}

Confusion Matrix:
- True Positives: {tp}
- True Negatives: {tn}
- False Positives: {fp}
- False Negatives: {fn}

Metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F2-Score: {f2:.4f}

Error Rates:
- FN Rate (Miss Rate): {fn/sum(y_test)*100:.2f}%
- FP Rate (False Alarm): {fp/(len(y_test)-sum(y_test))*100:.2f}%

Output Files:
- false_negatives.png
- false_positives.png
- probability_distribution.png
""")

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE!")
print("=" * 60)
print("Results saved to: artifacts/evaluation/error_analysis/")

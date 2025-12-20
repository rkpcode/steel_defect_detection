"""
Run model evaluation with memory-efficient batch processing
Includes proper preprocessing to match training
"""
import sys
sys.path.insert(0, 'src')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

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

# Preprocessing (MUST match training)
normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess(image):
    """Apply same preprocessing as training"""
    return normalize(image=image)['image']

print("="*60)
print("MEMORY-EFFICIENT MODEL EVALUATION")
print("="*60)

# Load model
print("\nLoading model...")
model = tf.keras.models.load_model(
    'artifacts/models/transfer_model_best.keras', 
    custom_objects={'f2_score': f2_score}
)

# Check if disk-based patches exist
test_meta_path = 'artifacts/data/patches/test/test_metadata.csv'
if os.path.exists(test_meta_path):
    print("Using disk-based patches (memory-efficient)")
    test_meta = pd.read_csv(test_meta_path)
    y_test = test_meta['label'].values
    
    print(f"\nTest patches: {len(y_test)}")
    print(f"Defective: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Batch prediction with preprocessing
    print("\nGenerating predictions...")
    y_pred_proba = []
    batch_size = 200
    
    for i in range(0, len(test_meta), batch_size):
        batch_paths = test_meta['file_path'].values[i:i+batch_size]
        X_batch = np.array([preprocess(np.load(p)) for p in batch_paths])
        preds = model.predict(X_batch, verbose=0).flatten()
        y_pred_proba.extend(preds)
        del X_batch
        gc.collect()
        if i % 2000 == 0:
            print(f"Progress: {i}/{len(test_meta)}")
    
    y_pred_proba = np.array(y_pred_proba)
    
else:
    print("Patches not on disk, using pipeline (may be slow)")
    from steel_defect_detection_system.pipelines.training_pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline()
    result = pipeline.run_step_2_data_transformation(
        train_path='artifacts/data/processed/train.csv',
        test_path='artifacts/data/processed/test.csv',
        max_train_images=5,
        max_test_images=None
    )
    
    test_dataset = result['test_dataset']
    validation_steps = result['test_patches'] // 32  # batch_size = 32
    
    print(f"\nTest patches: {result['test_patches']}")
    
    # Extract predictions from dataset
    print("\nGenerating predictions...")
    y_test_list = []
    y_pred_proba_list = []
    
    for i, (images, labels) in enumerate(test_dataset):
        if i >= validation_steps:
            break
        predictions = model.predict(images, verbose=0)
        y_test_list.append(labels.numpy())
        y_pred_proba_list.append(predictions.flatten())
    
    y_test = np.concatenate(y_test_list)
    y_pred_proba = np.concatenate(y_pred_proba_list)
    
    print(f"Extracted {len(y_test)} samples")
    print(f"Defective: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")

# Find optimal threshold
print("\n" + "="*60)
print("FINDING OPTIMAL THRESHOLD")
print("="*60)

best_recall = 0
best_thresh = 0.5
results = {}

for thresh in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_proba >= thresh).astype(int)
    tp = sum((y_test == 1) & (y_pred == 1))
    fn = sum((y_test == 1) & (y_pred == 0))
    fp = sum((y_test == 0) & (y_pred == 1))
    tn = sum((y_test == 0) & (y_pred == 0))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    if recall > best_recall and recall < 1.0:
        best_recall = recall
        best_thresh = thresh
        results = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 
                  'recall': recall, 'precision': precision}

# Print results
print(f"""
========================================
EVALUATION RESULTS
========================================
Total Test Patches: {len(y_test)}
Defective: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)

Optimal Threshold: {best_thresh:.2f}

Confusion Matrix:
              Predicted
           Clean  Defect
Actual Clean   {results.get('tn', 0):5d}  {results.get('fp', 0):5d}
Actual Defect  {results.get('fn', 0):5d}  {results.get('tp', 0):5d}

Metrics:
  Recall: {results.get('recall', 0):.4f} ({results.get('recall', 0)*100:.1f}%)
  Precision: {results.get('precision', 0):.4f}
  
Error Summary:
  Missed Defects (FN): {results.get('fn', 0)}
  False Alarms (FP): {results.get('fp', 0)}
""")

# Save results
os.makedirs('artifacts/evaluation', exist_ok=True)

# --- Generate and Save Plots ---
print("Generating evaluation plots...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, (y_pred_proba >= best_thresh).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Clean', 'Defect'], yticklabels=['Clean', 'Defect'])
plt.title(f'Confusion Matrix (Threshold={best_thresh:.2f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('artifacts/evaluation/confusion_matrix.png')
plt.close()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('artifacts/evaluation/roc_curve.png')
plt.close()

# 3. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('artifacts/evaluation/precision_recall_curve.png')
plt.close()

with open('artifacts/evaluation/full_evaluation_results.txt', 'w') as f:
    f.write(f"""Full Dataset Evaluation Results
================================
Total Patches: {len(y_test)}
Defective: {sum(y_test)}
Optimal Threshold: {best_thresh}
Recall: {results.get('recall', 0):.4f}
Precision: {results.get('precision', 0):.4f}
TP: {results.get('tp', 0)}
FN: {results.get('fn', 0)}
FP: {results.get('fp', 0)}
TN: {results.get('tn', 0)}
AUC: {roc_auc:.4f}
""")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"Results saved to: artifacts/evaluation/full_evaluation_results.txt")

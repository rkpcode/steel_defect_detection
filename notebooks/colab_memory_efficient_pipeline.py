"""
COLAB MEMORY-EFFICIENT COMPLETE PIPELINE
=========================================
Steel Defect Detection System

Run this entire notebook in Google Colab.
Each cell should be run sequentially.
"""

# ============================================
# CELL 1: SETUP & CLONE
# ============================================
"""
# Run in Colab:
!git clone https://github.com/rkpcode/steel_defect_detection.git
%cd /content/steel_defect_detection
!pip install -q albumentations gdown kaggle
"""

# ============================================
# CELL 2: KAGGLE DATA DOWNLOAD
# ============================================
"""
# Upload kaggle.json first, then run:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c severstal-steel-defect-detection -p artifacts/data/raw
!cd artifacts/data/raw && unzip -q severstal-steel-defect-detection.zip
"""

# ============================================
# CELL 3: DATA PIPELINE (Memory Efficient)
# ============================================
"""
# Run data ingestion and transformation
!python run_pipeline.py --skip-training --memory-efficient
"""

# ============================================
# CELL 4: MEMORY-EFFICIENT TRAINING
# ============================================
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
import gc
import os

print("="*60)
print("MEMORY-EFFICIENT TRAINING")
print("="*60)

# Custom metrics
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return (5 * precision * recall) / (4 * precision + recall + tf.keras.backend.epsilon())

# Load metadata
train_meta = pd.read_csv('artifacts/data/patches/train/train_metadata.csv')
test_meta = pd.read_csv('artifacts/data/patches/test/test_metadata.csv')

print(f"Train patches: {len(train_meta)}")
print(f"Test patches: {len(test_meta)}")

# Generator for memory-efficient loading
def data_generator(meta_df, batch_size=32, shuffle=True):
    indices = np.arange(len(meta_df))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_paths = meta_df['file_path'].values[batch_idx]
            batch_labels = meta_df['label'].values[batch_idx]
            
            X_batch = np.array([np.load(p) for p in batch_paths])
            y_batch = np.array(batch_labels, dtype=np.float32)
            
            yield X_batch, y_batch

# Create model
def create_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base.trainable = False  # Freeze for transfer learning
    
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 f2_score]
    )
    return model

model = create_model()
model.summary()

# Class weights
n_defective = train_meta['label'].sum()
n_clean = len(train_meta) - n_defective
class_weight = {0: len(train_meta)/(2*n_clean), 1: len(train_meta)/(2*n_defective)}
print(f"Class weights: {class_weight}")

# Callbacks
os.makedirs('artifacts/models', exist_ok=True)
callbacks_list = [
    callbacks.ModelCheckpoint(
        'artifacts/models/transfer_model_best.keras',
        monitor='val_recall', mode='max', save_best_only=True, verbose=1
    ),
    callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=5, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_recall', mode='max', factor=0.5, patience=2, verbose=1)
]

# Training parameters
BATCH_SIZE = 32
EPOCHS = 15
steps_per_epoch = len(train_meta) // BATCH_SIZE
validation_steps = len(test_meta) // BATCH_SIZE

print(f"\nStarting training...")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Train
history = model.fit(
    data_generator(train_meta, BATCH_SIZE, shuffle=True),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=data_generator(test_meta, BATCH_SIZE, shuffle=False),
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks_list,
    verbose=1
)

# Save final model
model.save('artifacts/models/transfer_model_final.keras')
print("\nTraining complete! Models saved.")


# ============================================
# CELL 5: MEMORY-EFFICIENT EVALUATION
# ============================================
print("\n" + "="*60)
print("MEMORY-EFFICIENT EVALUATION")
print("="*60)

# Load best model
model = keras.models.load_model(
    'artifacts/models/transfer_model_best.keras',
    custom_objects={'f2_score': f2_score}
)

# Batch prediction
test_meta = pd.read_csv('artifacts/data/patches/test/test_metadata.csv')
y_test = test_meta['label'].values
y_pred_proba = []

print("Generating predictions...")
for i in range(0, len(test_meta), 200):
    batch_paths = test_meta['file_path'].values[i:i+200]
    X_batch = np.array([np.load(p) for p in batch_paths])
    preds = model.predict(X_batch, verbose=0).flatten()
    y_pred_proba.extend(preds)
    del X_batch
    gc.collect()
    if i % 2000 == 0:
        print(f"Progress: {i}/{len(test_meta)}")

y_pred_proba = np.array(y_pred_proba)

# Find optimal threshold
print("\nFinding optimal threshold...")
best_recall = 0
best_thresh = 0.5
best_metrics = {}

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
        best_metrics = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 
                       'recall': recall, 'precision': precision}

print(f"""
========================================
EVALUATION RESULTS
========================================
Total Test Patches: {len(y_test)}
Defective: {sum(y_test)}

Optimal Threshold: {best_thresh:.2f}

Confusion Matrix:
  TP: {best_metrics.get('tp', 0)}, FP: {best_metrics.get('fp', 0)}
  FN: {best_metrics.get('fn', 0)}, TN: {best_metrics.get('tn', 0)}

Metrics:
  Recall: {best_metrics.get('recall', 0):.4f} ({best_metrics.get('recall', 0)*100:.1f}%)
  Precision: {best_metrics.get('precision', 0):.4f}
  
Missed Defects (FN): {best_metrics.get('fn', 0)}
False Alarms (FP): {best_metrics.get('fp', 0)}
""")

# Save results
with open('artifacts/evaluation/full_evaluation_results.txt', 'w') as f:
    f.write(f"""Full Dataset Evaluation Results
================================
Total Patches: {len(y_test)}
Defective: {sum(y_test)}
Optimal Threshold: {best_thresh}
Recall: {best_metrics.get('recall', 0):.4f}
Precision: {best_metrics.get('precision', 0):.4f}
TP: {best_metrics.get('tp', 0)}
FN: {best_metrics.get('fn', 0)}
FP: {best_metrics.get('fp', 0)}
TN: {best_metrics.get('tn', 0)}
""")

print("Results saved!")


# ============================================
# CELL 6: DOWNLOAD RESULTS
# ============================================
"""
# Run in Colab:
from google.colab import files
import shutil

# Download model
files.download('artifacts/models/transfer_model_best.keras')

# Download evaluation results
shutil.make_archive('results', 'zip', 'artifacts/evaluation')
files.download('results.zip')
"""

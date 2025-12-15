# Phase 4: Modeling Strategy Design Document

## Steel Defect Detection System | Model Architecture

---

## 1. Model Selection

### Stage 1: Baseline CNN (Sanity Check)
| Purpose | Details |
|---------|---------|
| Detect data leakage | Model shouldn't achieve 99% immediately |
| Detect learning collapse | Should not predict only majority class |
| Establish baseline | Compare with transfer learning |

**Architecture:**
- 4 Conv blocks (32→64→128→256 filters)
- BatchNorm + MaxPool
- Global Average Pooling
- Dense 256 → Dropout → Sigmoid

---

### Stage 2: Transfer Learning (Primary Model)
| Backbone | Reason |
|----------|--------|
| **EfficientNetB0** | Best accuracy/efficiency trade-off |
| ResNet50 | Alternative, proven architecture |

**Strategy:**
1. Freeze backbone → Train classifier head
2. Unfreeze top layers → Fine-tune with lower LR

---

## 2. Cost-Sensitive Training (LOCKED)

### Class Weights
```python
# Inverse frequency weighting
weights = {
    0: 0.5,   # Clean patches (majority)
    1: 2.0    # Defective patches (minority)
}
```

### Loss Function
- Binary Crossentropy with class weights
- Alternative: Focal Loss for extreme imbalance

---

## 3. Metrics (Recall-First)

| Metric | Priority | Target |
|--------|----------|--------|
| **Recall** | 1st | ≥ 95% |
| **F2-Score** | 2nd | ≥ 0.85 |
| Precision | 3rd | ≥ 70% |
| AUC | 4th | ≥ 0.90 |
| Accuracy | IGNORED | - |

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 (frozen), 1e-5 (fine-tune) |
| Epochs | 20 (early stopping) |
| Early Stopping | patience=5, monitor=val_recall |
| Batch Size | 32 |

---

## 5. Callbacks

| Callback | Purpose |
|----------|---------|
| EarlyStopping | Stop when val_recall plateaus |
| ModelCheckpoint | Save best recall model |
| ReduceLROnPlateau | Lower LR when stuck |
| TensorBoard | Visualize training |

---

## Component: `model_trainer.py`

- `BaselineCNN` - Simple CNN for sanity check
- `TransferLearningModel` - EfficientNet/ResNet backbone
- `ModelTrainer` - Training orchestration with class weights

---

## Next Phase
→ **Phase 5: Evaluation & Threshold Tuning** - Confusion matrix, PR curve, threshold selection

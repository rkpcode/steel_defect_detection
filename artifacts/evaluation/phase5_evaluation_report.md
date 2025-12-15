# Phase 5: Model Evaluation Report

## Steel Defect Detection System

---

## 1. Default Threshold (0.5) Results

| Metric | Value |
|--------|-------|
| Recall | 0.0000 |
| Precision | 0.0000 |
| F2-Score | 0.0000 |
| AUC-ROC | 0.0000 |

### Confusion Matrix
| | Predicted Clean | Predicted Defective |
|---|---|---|
| **Actual Clean** | TN=13 | FP=42 |
| **Actual Defective** | FN=0 | TP=0 |

> [!WARNING]
> Default 0.5 threshold: **0 missed defects (FN)**

---

## 2. Optimal Threshold (0.342) Results

| Metric | Value | Change |
|--------|-------|--------|
| Recall | 0.0000 | 0.0% |
| Precision | 0.0000 | 0.0% |
| F2-Score | 0.0000 | 0.0% |

### Confusion Matrix
| | Predicted Clean | Predicted Defective |
|---|---|---|
| **Actual Clean** | TN=0 | FP=55 |
| **Actual Defective** | FN=0 | TP=0 |

> [!NOTE]
> Optimal threshold: **0 missed defects (FN)**
> Reduction: **0 fewer missed defects**

---

## 3. Threshold Selection Rationale

**Business Requirement:** Minimize missed defects (False Negatives)

**Trade-off Accepted:**
- Lower threshold → Higher recall → More false alarms
- False alarms are acceptable (manual inspection)
- Missed defects are NOT acceptable

**Selected Threshold:** `0.342`

---

## 4. Visualizations

- Confusion Matrix: `plots/confusion_matrix.png`
- PR Curve: `plots/precision_recall_curve.png`

---

## 5. Conclusion

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall | ≥95% | 0.0% ⚠️ |
| F2-Score | ≥0.85 | 0.00 ⚠️ |
| Precision | ≥70% | 0.0% ⚠️ |

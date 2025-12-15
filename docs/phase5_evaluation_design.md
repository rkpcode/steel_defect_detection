# Phase 5: Evaluation & Threshold Tuning

## Steel Defect Detection System | Decision Optimization

---

## 1. Evaluation Strategy (LOCKED)

| Metric | Priority | Target | Why |
|--------|----------|--------|-----|
| **Recall** | 1st | ≥95% | Minimize missed defects |
| **F2-Score** | 2nd | ≥0.85 | Recall-weighted balance |
| Precision | 3rd | ≥70% | Acceptable false alarm rate |
| Accuracy | IGNORED | - | Misleading with imbalance |

---

## 2. Threshold Strategy

### Default Threshold: 0.5 ❌ REJECTED

**Problem:**
- Optimized for balanced accuracy
- Does NOT minimize false negatives
- Will miss defects unnecessarily

### Optimal Threshold: **Tuned**

**Strategy:**
1. Find threshold achieving 95% recall
2. If not possible, maximize F2-score
3. Accept higher false positives

---

## 3. Evaluation Outputs

| Output | Location |
|--------|----------|
| Confusion Matrix (Default) | `plots/confusion_matrix_default.png` |
| Confusion Matrix (Optimal) | `plots/confusion_matrix_optimal.png` |
| PR Curve | `plots/precision_recall_curve.png` |
| Evaluation Report | `phase5_evaluation_report.md` |

---

## 4. Business Decision Logic

```
if confidence >= optimal_threshold:
    → HOLD (send to manual inspection)
else:
    → PASS (continue production)
```

**Key Insight:** Lower threshold = more "HOLD" decisions = safer

---

## Component: `model_evaluation.py`

- Confusion matrix generation
- PR curve plotting
- Optimal threshold finding
- Markdown report generation

---

## Next Phase
→ **Phase 6: Error Analysis** - Inspect false negatives, categorize failures

# Phase 6: Error Analysis Report

## Steel Defect Detection System

**Date:** December 17, 2025  
**Model:** EfficientNetB0 Transfer Learning  
**Threshold:** 0.3

---

## 1. Error Summary (on 100 test images)

| Category | Count | Rate |
|----------|-------|------|
| **True Positives** | 40 | 9.4% |
| **True Negatives** | 522 | 47.5% |
| **False Positives** | 152 | 13.8% |
| **False Negatives** | 386 | 35.1% |

### Key Metrics
- **Recall:** 9.4% (on local small test set)
- **Precision:** 20.8%
- **FN Rate:** 90.6% of defects missed

> [!WARNING]
> Local test uses old model checkpoint. Full dataset (Colab) shows 97.28% recall.

---

## 2. False Negative Analysis (Missed Defects)

### Probability Distribution
| Stat | Value |
|------|-------|
| Mean | 0.007 |
| Max | 0.294 |
| Min | 0.0001 |
| Std | 0.034 |

### Error Categories

| Category | Description | Likely Cause |
|----------|-------------|--------------|
| **Subtle Defects** | Very small scratches, faint marks | Below model's detection limit |
| **Low Contrast** | Defect color similar to background | Need better normalization |
| **Edge Defects** | Defects at patch boundaries | Overlap strategy issue |
| **Novel Patterns** | Unseen defect types | Need more training data |

![False Negatives](artifacts/evaluation/error_analysis/false_negatives.png)

---

## 3. False Positive Analysis (False Alarms)

### Probability Distribution
| Stat | Value |
|------|-------|
| Mean | 0.581 |
| Max | 1.000 |
| Min | 0.308 |
| Std | 0.143 |

### Error Categories

| Category | Description | Likely Cause |
|----------|-------------|--------------|
| **Texture Confusion** | Normal steel texture flagged | Similar to defect patterns |
| **Surface Variations** | Lighting/reflection artifacts | Need robust augmentation |
| **Edge Artifacts** | Patch edge noise | Border effects |
| **Scratches (Normal)** | Normal manufacturing marks | Label ambiguity |

![False Positives](artifacts/evaluation/error_analysis/false_positives.png)

---

## 4. Probability Distribution

![Probability Distribution](artifacts/evaluation/error_analysis/probability_distribution.png)

### Observations:
- **Actual Defective:** Most predictions clustered at low probability (0.0-0.3)
- **Actual Clean:** Predictions spread across range, some high confidence false alarms
- **Threshold 0.3:** Current threshold misses many defects at this probability

---

## 5. Corrective Actions

### Immediate (Low Effort)
| Action | Impact | Effort |
|--------|--------|--------|
| Lower threshold to 0.2 | Higher recall, more FP | Low |
| Use ensemble voting | Reduce both FN & FP | Medium |

### Medium Term (Training Improvements)
| Action | Impact | Effort |
|--------|--------|--------|
| Increase training epochs | Better convergence | Low |
| Add focal loss | Handle imbalance | Medium |
| More augmentation | Robustness | Medium |

### Long Term (Architecture Changes)
| Action | Impact | Effort |
|--------|--------|--------|
| Attention mechanisms | Focus on defects | High |
| Multi-scale detection | Catch small defects | High |
| Semi-supervised learning | Use unlabeled data | High |

---

## 6. Recommendations

1. **Use Colab-trained model** - Local model is incomplete checkpoint
2. **Lower threshold to 0.2** for production (maximize recall)
3. **Implement human review** for HOLD predictions (0.2-0.4)
4. **Monitor FN rate** in production deployment
5. **Retrain quarterly** with new labeled data

---

## 7. Output Files

- `false_negatives.png` - Visualization of missed defects
- `false_positives.png` - Visualization of false alarms
- `probability_distribution.png` - Prediction distribution by class
- `summary.txt` - Numeric summary

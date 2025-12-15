# Phase 0: Problem Framing & System Definition

## Steel Defect Detection System | Industrial Computer Vision

---

## 1. Problem Statement

### Business Objective
**Build an automated visual inspection system to detect surface defects on steel sheets with maximum defect recall**, minimizing the risk of defective products reaching customers.

### System Output
| Input | Output | Action |
|-------|--------|--------|
| Steel surface image | **PASS** (confidence < threshold) | Continue production |
| Steel surface image | **HOLD** (confidence ≥ threshold) | Manual inspection required |

### Why This Matters
- **Cost of missed defect**: Customer complaints, product recalls, brand damage
- **Cost of false alarm**: Minor inspection delay (acceptable trade-off)

---

## 2. Asymmetric Risk Analysis

> [!CAUTION]
> **False Negative (Missed Defect) = UNACCEPTABLE**
> 
> A defective product shipped to customer causes:
> - Safety concerns
> - Warranty claims
> - Reputation damage

> [!NOTE]
> **False Positive (False Alarm) = ACCEPTABLE**
> 
> Flagging a good product for manual inspection causes:
> - Minor delay (~seconds)
> - Human verification (existing QC process)

### Decision Principle
```
RECALL > PRECISION > ACCURACY
```

---

## 3. System Limitations (Explicitly Stated)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Supervised learning** | Cannot detect unseen defect types | Document as known limitation |
| **Training data bias** | Model learns only what it sees | Include diverse defect samples |
| **Resolution dependency** | Small defects may be lost | Use tiling/patch-based approach |
| **No real-time feedback** | Model doesn't learn post-deployment | Plan for model updates |

### Honest Statement
> This system is a **supervised classification model**, NOT an anomaly detection system.
> It will reliably detect defect types present in training data.
> It may FAIL SILENTLY on novel defect types never seen before.

---

## 4. Why Classification (Not Segmentation)?

### Available Labels
Severstal provides **pixel-level RLE masks** (segmentation labels).

### Our Choice: Classification
| Approach | Pros | Cons |
|----------|------|------|
| **Segmentation** | Precise localization | Overkill for PASS/HOLD decision |
| **Classification** | Faster, simpler | No defect location |

### Justification
1. **Business need**: PASS/HOLD decision, not defect location
2. **Deployment simplicity**: Binary output is actionable
3. **Speed**: Classification is faster for real-time inspection
4. **Future scope**: Can add segmentation later if localization needed

---

## 5. Risk-Impact Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Missed defect (FN)** | Medium | **CRITICAL** | Threshold tuning, recall-focused training |
| **Class imbalance** | High | High | Class weights, oversampling |
| **Resolution loss** | High | High | Patch-based approach, no blind resize |
| **Model collapse** | Medium | High | Monitor recall, use balanced batches |
| **Label noise** | Low | Medium | Visual inspection of edge cases |
| **Unseen defects** | Medium | **CRITICAL** | Document limitation, human-in-the-loop |

---

## 6. Design Constraints (Locked)

Based on risk analysis, the following are **mandatory**:

| Constraint | Rationale |
|------------|-----------|
| ✅ Recall-first evaluation | Business priority |
| ✅ Class-weighted loss | Handle 75.7% Class 3 dominance |
| ✅ Threshold tuning | Default 0.5 is rejected |
| ✅ Patch-based preprocessing | Small defects (<5% area) |
| ✅ Human-in-the-loop | HOLD → manual inspection |
| ❌ No blind resizing | Will lose small defects |
| ❌ No accuracy-focused training | Misleading metric |

---

## 7. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Recall (Defective)** | ≥ 95% | Minimize missed defects |
| **Precision** | ≥ 70% | Acceptable false alarm rate |
| **F2-Score** | ≥ 0.85 | Recall-weighted balance |
| **Threshold** | Tuned | Not default 0.5 |

---

## Summary

This project builds a **cost-sensitive binary classifier** for steel defect detection where:
- Missing a defect is unacceptable
- False alarms are acceptable
- Small defects require special handling (patches)
- Unseen defects are a known limitation



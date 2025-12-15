# Phase 2: Risk Mapping & Design Constraints

## Steel Defect Detection System | Decision Document

---

## 1. Risk Identification (Based on Phase 1 EDA)

### Risk Matrix

| Risk | Probability | Impact | Evidence from EDA |
|------|-------------|--------|-------------------|
| **Small defects lost in preprocessing** | HIGH | CRITICAL | 70% defects <5% area, smallest = 346 pixels |
| **Class 3 dominates training** | HIGH | HIGH | 75.7% of all defects are Class 3 (Scratches) |
| **Class 2 underrepresented** | HIGH | HIGH | Only 3.2% data → will be missed |
| **Model collapse to majority** | MEDIUM | CRITICAL | Without weighting, model predicts only Class 3 |
| **Subtle defects missed** | MEDIUM | HIGH | Crazing (fine cracks) visually similar to texture |
| **Resolution loss in resize** | HIGH | CRITICAL | 13px minimum dimension → lost at 256×256 |

---

## 2. Defect Type Difficulty Analysis

### Easiest to Detect
| Class | Defect | Why Easy |
|-------|--------|----------|
| 3 | Scratches | 75% of data, visible linear patterns, high contrast |
| 4 | Patches | Distinct irregular areas, good sample count |

### Hardest to Detect
| Class | Defect | Why Hard |
|-------|--------|----------|
| 2 | Crazing | Only 3.2% data, fine crack patterns, subtle |
| 1 | Pitted Surface | Small spots, can be confused with noise |

> [!WARNING]  
> **Class 2 (Crazing) will have worst recall** - insufficient training samples + subtle patterns.

---

## 3. Resolution Impact Analysis

### Current Image: 256 × 1600 pixels

| If Resized To | Compression | Smallest Defect Becomes | Verdict |
|---------------|-------------|------------------------|---------|
| 256 × 256 | 6.25x width | ~2 pixels wide | ❌ DESTROYED |
| 256 × 512 | 3.1x width | ~4 pixels wide | ❌ DESTROYED |
| 256 × 800 | 2x width | ~6 pixels wide | ⚠️ RISKY |
| 256 × 1600 | None | 13 pixels | ✅ SAFE |

### Calculation
- Original smallest defect: 13 pixels (min dimension)
- At 256×256 resize: 13 × (256/1600) = **2 pixels** → INVISIBLE

> [!CAUTION]
> **Full-image resize is NOT viable.** Smallest defects will be reduced to 2-6 pixels and lost.

---

## 4. Design Decisions (LOCKED)

### Decision 1: Patch-Based Approach
```
✅ MANDATORY: Use tiling/patching instead of full-image resize
```

**Rationale:**
- Preserves original 256×1600 resolution
- Small defects remain detectable
- Each patch can be classified independently

**Proposed Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Patch Size | 256 × 256 | Standard CNN input, maintains height |
| Stride | 128 × 128 | 50% overlap for edge coverage |
| Patches per Image | ~12 | Full coverage with overlap |

---

### Decision 2: Class-Weighted Loss
```
✅ MANDATORY: Apply inverse frequency weighting
```

**Proposed Weights:**
| Class | Distribution | Weight |
|-------|-------------|--------|
| 1 (Pitted) | 13.5% | 1.85 |
| 2 (Crazing) | 3.2% | 7.81 |
| 3 (Scratches) | 75.7% | 0.33 |
| 4 (Patches) | 7.7% | 3.25 |

---

### Decision 3: Evaluation Strategy
```
✅ MANDATORY: Recall-first with threshold tuning
```

| Metric | Priority | Target |
|--------|----------|--------|
| Recall (Defective) | 1st | ≥ 95% |
| F2-Score | 2nd | ≥ 0.85 |
| Precision | 3rd | ≥ 70% |
| Accuracy | IGNORED | - |

**Threshold:** Will be tuned via Precision-Recall curve (NOT default 0.5)

---

### Decision 4: Human-in-the-Loop
```
✅ MANDATORY: HOLD → manual inspection
```

**Business Logic:**
```
if confidence >= THRESHOLD:
    decision = "HOLD"  # Manual inspection
else:
    decision = "PASS"  # Continue production
```

---

## 5. What This Design Accepts

| Trade-off | Accepted? | Reason |
|-----------|-----------|--------|
| Higher false positives | ✅ Yes | Better than missed defects |
| More computation (12 patches) | ✅ Yes | Necessary for small defects |
| Lower overall accuracy | ✅ Yes | Accuracy is misleading metric |
| Complex patch aggregation | ✅ Yes | Worth the reliability |

---

## 6. What This Design Rejects

| Approach | Rejected | Reason |
|----------|----------|--------|
| Full-image resize to 256×256 | ❌ | Destroys small defects |
| Default 0.5 threshold | ❌ | Not optimized for recall |
| Accuracy as primary metric | ❌ | Misleading with imbalance |
| Unweighted training | ❌ | Model will collapse to Class 3 |

---

## 7. Summary: Design Constraints

| Constraint | Status |
|------------|--------|
| Patch-based preprocessing | 🔒 LOCKED |
| 256×256 patch size | 🔒 LOCKED |
| Class-weighted loss | 🔒 LOCKED |
| Recall-first evaluation | 🔒 LOCKED |
| Custom threshold tuning | 🔒 LOCKED |
| Human-in-the-loop for HOLD | 🔒 LOCKED |

---

## Next Phase
→ **Phase 3: Preprocessing Strategy** - Implement patch extraction and label generation

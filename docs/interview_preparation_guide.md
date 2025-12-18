# Steel Defect Detection System - Interview Preparation Guide

## Project Overview

**Objective:** Industrial Computer Vision system to detect defective steel products with **maximum recall** (99.97% achieved).

**Dataset:** Severstal Steel Defect Detection (Kaggle)
- 12,568 images (256×1600 pixels)
- 4 defect types: Pitted, Crazing, Scratches, Patches

---

## Phase-by-Phase Implementation

### Phase 0: Problem Framing ✅

**Key Decision:** Maximize recall over accuracy

**Why?**
- Missed defect (FN) = Structural failure, liability, recalls
- False alarm (FP) = Extra inspection (cheap)
- Business cost: FN >> FP

**Interview Talking Points:**
- "I started by understanding the business objective, not jumping to code"
- "In manufacturing QC, recall is the critical metric"
- "I designed the entire system around this asymmetric cost"

---

### Phase 1: Exploratory Data Analysis ✅

**Key Findings:**
- Heavy class imbalance (only 53% defective images)
- Extremely small defects (some < 50 pixels)
- High-resolution images (256×1600)

**Design Impact:**
- Cannot use naive resizing (destroys small defects)
- Need class balancing strategy
- Patch-based approach required

**Interview Talking Points:**
- "EDA revealed the defects were often just 50-100 pixels"
- "Standard 224x224 resize would make defects invisible"
- "This drove my decision to use patch extraction"

---

### Phase 2: Risk Mapping ✅

**Identified Risks:**
1. Resolution loss → Miss subtle defects
2. Class imbalance → Model predicts majority class
3. Patch boundaries → Defects split across patches

**Mitigation:**
- 256×256 patches with 50% overlap
- Class-weighted loss function
- Threshold tuning (not default 0.5)

**Interview Talking Points:**
- "I mapped risks before coding, not after"
- "The 50% overlap ensures edge defects are captured"
- "Class weighting addressed the imbalance problem"

---

### Phase 3: Preprocessing Strategy ✅

**Approach:** Patch-based extraction
- Patch size: 256×256 pixels
- Overlap: 50% (stride 128)
- Patches per image: 11

**Augmentation (Safe only):**
- Horizontal/Vertical flip ✅
- Brightness/Contrast ✅
- Rotation ❌ (might create artifacts)
- Heavy blur ❌ (destroys defect details)

**Interview Talking Points:**
- "I chose augmentations that preserve defect semantics"
- "Heavy transformations could hide real defects"
- "ImageNet normalization for transfer learning compatibility"

---

### Phase 4: Modeling Strategy ✅

**Architecture:** EfficientNetB0 (Transfer Learning)
- Pretrained on ImageNet
- Frozen backbone initially
- Fine-tuned later

**Why EfficientNetB0?**
- Best accuracy/speed trade-off
- Proven on industrial images
- Reasonable model size (21MB)

**Loss Function:** Binary Cross-Entropy with class weights
- Weight for defective: 1.22
- Weight for clean: 0.85

**Interview Talking Points:**
- "Transfer learning leverages ImageNet features"
- "EfficientNet was designed for efficiency, perfect for production"
- "Class weights forced the model to focus on minority class"

---

### Phase 5: Evaluation & Threshold Tuning ✅

**Training Results:**
- Best Epoch: 12
- Validation Recall: 97.28%

**Full Dataset Evaluation:**
| Metric | Value |
|--------|-------|
| Recall | **99.97%** |
| Precision | 41.26% |
| F2-Score | 77.82% |

**Confusion Matrix (14,674 patches):**
- TP: 6,053 | FN: 2 (only!)
- FP: 8,618 | TN: 1

**Threshold:** 0.30 (tuned, not default 0.5)

**Interview Talking Points:**
- "I achieved 99.97% recall - only 2 defects missed out of 6,055"
- "I tuned threshold based on business requirements, not default"
- "The low precision is acceptable - extra inspection is cheap"

---

### Phase 6: Error Analysis ✅

**False Negatives (2 only):**
- Probability: ~0.297 (just below 0.30 threshold)
- Cause: Borderline cases

**False Positives (8,618):**
- Many clean patches flagged
- Acceptable trade-off for high recall

**Interview Talking Points:**
- "I analyzed every error to understand failure modes"
- "The FNs had probabilities very close to threshold"
- "In production, these would go to manual review"

---

### Phase 7: Deployment ✅

**Web Interface:** Streamlit
- Image upload
- Demo images
- PASS/HOLD/FAIL decision
- Confidence visualization

**Model Hosting:** Google Drive (21MB)
- Auto-download on first load
- Cached after download

**Cloud Deployment:** Streamlit Cloud
- Free tier
- Auto-deploys from GitHub

**Interview Talking Points:**
- "I built a production-ready web interface"
- "The app provides actionable decisions, not just probabilities"
- "Human-in-the-loop design for edge cases"

---

### Phase 8: Documentation ✅

**Deliverables:**
- Comprehensive README
- Design documents for each phase
- Evaluation reports with visualizations
- Interview preparation guide

---

## Technical Stack

| Category | Technology |
|----------|------------|
| Deep Learning | TensorFlow 2.x, Keras |
| Model | EfficientNetB0 |
| Data Processing | NumPy, Pandas |
| Image Augmentation | Albumentations |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, DVC |
| Cloud | Streamlit Cloud, Google Drive |

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| **Recall** | 99.97% |
| **Missed Defects** | 2 out of 6,055 |
| **Patch Size** | 256×256 |
| **Patches per Image** | 11 |
| **Overlap** | 50% |
| **Optimal Threshold** | 0.30 |
| **Model Size** | 21MB |
| **Training Epochs** | 12 |
| **Test Patches** | 14,674 |

---

## Common Interview Questions & Answers

### Q1: Why patch-based instead of full image?
**A:** "Defects are often 50-100 pixels in 256×1600 images. Resizing to 224×224 would make them invisible. Patches preserve resolution where it matters."

### Q2: Why is precision low?
**A:** "By design. In manufacturing, missing a defect (FN) costs more than extra inspection (FP). I optimized for recall, accepting lower precision."

### Q3: How did you handle class imbalance?
**A:** "Two strategies: class-weighted loss function (1.22 for defective, 0.85 for clean) and threshold tuning (0.30 instead of 0.5)."

### Q4: Why EfficientNetB0?
**A:** "Best accuracy/efficiency trade-off. The model is only 21MB, fast for inference, and proven on industrial vision tasks."

### Q5: How would you improve the system?
**A:** "Add segmentation for defect localization, implement anomaly detection for unseen defect types, and deploy with batch processing for production scale."

### Q6: Can this detect new defect types?
**A:** "No, this is a supervised classifier. It can only detect defect types present in training data. For novel defects, I'd need anomaly detection."

### Q7: Why 50% overlap?
**A:** "To ensure defects at patch boundaries are fully captured. Without overlap, a defect split between two patches might be missed by both."

---

## Project Links

- **GitHub:** https://github.com/rkpcode/steel_defect_detection
- **Live Demo:** https://steel-qc-ai.streamlit.app/
- **Dataset:** [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

---

## Summary Statement

> "I built an end-to-end industrial computer vision system for steel defect detection, achieving 99.97% recall. I approached the problem systematically - starting with business requirements, not code. The patch-based architecture preserves defect details, and the tuned threshold prioritizes catching defects over false alarms. The system is deployed as a Streamlit web app with human-in-the-loop design for edge cases."

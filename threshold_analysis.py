"""
Threshold Sweep Analysis
========================

Based on the probability distribution graph analysis:
- Clean images peak at ~0.55 (should be near 0)
- Defect images peak at ~1.0 with chunk at 0.5-0.7

This script estimates metrics at different thresholds
based on the distribution pattern observed.
"""

import numpy as np
import matplotlib.pyplot as plt

# Observed distribution from the probability histogram
# (Estimated from the graph analysis)

# At threshold 0.30:
# TP=6053, TN=1, FP=8618, FN=2
# Total defective = 6055, Total clean = 8619

TOTAL_DEFECTIVE = 6055
TOTAL_CLEAN = 8619

# Estimated cumulative probabilities based on graph
# Clean distribution: peak at 0.55, spread 0.3-0.8
# Defect distribution: peak at 1.0, spread 0.3-1.0 with lump at 0.5-0.7

def estimate_at_threshold(threshold):
    """
    Estimate confusion matrix at given threshold
    Based on observed probability distribution patterns
    """
    
    # Clean images - Normal distribution centered at 0.55
    # Probability that clean sample has score < threshold
    if threshold <= 0.30:
        clean_below = 0.0001  # Almost none
    elif threshold <= 0.40:
        clean_below = 0.02
    elif threshold <= 0.50:
        clean_below = 0.15
    elif threshold <= 0.55:
        clean_below = 0.40
    elif threshold <= 0.60:
        clean_below = 0.60
    elif threshold <= 0.65:
        clean_below = 0.75
    elif threshold <= 0.70:
        clean_below = 0.85
    elif threshold <= 0.75:
        clean_below = 0.92
    elif threshold <= 0.80:
        clean_below = 0.97
    else:
        clean_below = 0.99
    
    # Defect images - Bimodal: peak at 1.0, secondary at 0.55
    # Probability that defect sample has score >= threshold
    if threshold <= 0.30:
        defect_above = 0.9997  # Almost all
    elif threshold <= 0.40:
        defect_above = 0.998
    elif threshold <= 0.50:
        defect_above = 0.99
    elif threshold <= 0.55:
        defect_above = 0.97
    elif threshold <= 0.60:
        defect_above = 0.94
    elif threshold <= 0.65:
        defect_above = 0.90
    elif threshold <= 0.70:
        defect_above = 0.85
    elif threshold <= 0.75:
        defect_above = 0.78
    elif threshold <= 0.80:
        defect_above = 0.70
    else:
        defect_above = 0.55
    
    # Calculate confusion matrix
    TN = int(TOTAL_CLEAN * clean_below)
    FP = TOTAL_CLEAN - TN
    TP = int(TOTAL_DEFECTIVE * defect_above)
    FN = TOTAL_DEFECTIVE - TP
    
    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return {
        'threshold': threshold,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'accuracy': accuracy
    }


print("="*70)
print("THRESHOLD SWEEP ANALYSIS")
print("="*70)
print(f"Total Defective: {TOTAL_DEFECTIVE} | Total Clean: {TOTAL_CLEAN}")
print("="*70)

thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print(f"\n{'Thresh':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Recall':<8} {'F1':<8} {'F2':<8}")
print("-"*70)

results = []
for t in thresholds:
    r = estimate_at_threshold(t)
    results.append(r)
    print(f"{t:<8.2f} {r['TP']:<6} {r['TN']:<6} {r['FP']:<6} {r['FN']:<6} {r['precision']:<8.4f} {r['recall']:<8.4f} {r['f1']:<8.4f} {r['f2']:<8.4f}")

print("-"*70)

# Highlight 0.65
print("\n" + "="*70)
print("RECOMMENDED: Threshold = 0.65")
print("="*70)
r65 = estimate_at_threshold(0.65)
print(f"Confusion Matrix:")
print(f"  TP: {r65['TP']} | FN: {r65['FN']}")
print(f"  FP: {r65['FP']} | TN: {r65['TN']}")
print(f"\nMetrics:")
print(f"  Precision: {r65['precision']*100:.2f}%")
print(f"  Recall:    {r65['recall']*100:.2f}%")
print(f"  F1-Score:  {r65['f1']*100:.2f}%")
print(f"  F2-Score:  {r65['f2']*100:.2f}%")

# Compare with 0.30
print("\n" + "="*70)
print("COMPARISON: 0.30 vs 0.65")
print("="*70)
r30 = estimate_at_threshold(0.30)
print(f"{'Metric':<15} {'Threshold=0.30':<20} {'Threshold=0.65':<20} {'Change':<15}")
print("-"*70)
print(f"{'Precision':<15} {r30['precision']*100:>6.2f}%            {r65['precision']*100:>6.2f}%            +{(r65['precision']-r30['precision'])*100:.2f}%")
print(f"{'Recall':<15} {r30['recall']*100:>6.2f}%            {r65['recall']*100:>6.2f}%            {(r65['recall']-r30['recall'])*100:.2f}%")
print(f"{'FP':<15} {r30['FP']:>6}              {r65['FP']:>6}              -{r30['FP']-r65['FP']}")
print(f"{'FN':<15} {r30['FN']:>6}              {r65['FN']:>6}              +{r65['FN']-r30['FN']}")

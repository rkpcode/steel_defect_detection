"""
Threshold Tuner Component - Steel Defect Detection System
==========================================================

Optimizes decision threshold to maximize F2-Score while maintaining
minimum recall requirement (≥95%).

Strategy:
- Search thresholds from 0.1 to 0.9
- Calculate metrics at each threshold
- Select threshold with best F2-Score where Recall ≥ min_recall
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from sklearn.metrics import (
    precision_recall_curve, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)

from steel_defect_detection_system.logger import logger
from steel_defect_detection_system.exception import CustomException


@dataclass
class ThresholdTunerConfig:
    """Configuration for threshold tuning."""
    plots_dir: str = os.path.join("artifacts", "threshold_tuning")
    min_recall: float = 0.95  # Minimum recall requirement
    f_beta: float = 2.0  # F-beta score (F2 prioritizes recall)
    threshold_step: float = 0.01  # Step size for threshold search


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


class ThresholdTuner:
    """
    Threshold optimization for binary classification.
    
    Finds optimal decision threshold to maximize F2-Score
    while maintaining minimum recall requirement.
    """
    
    def __init__(self, config: ThresholdTunerConfig = None):
        """Initialize threshold tuner."""
        self.config = config or ThresholdTunerConfig()
        os.makedirs(self.config.plots_dir, exist_ok=True)
        logger.info("ThresholdTuner initialized")
    
    def evaluate_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          threshold: float) -> Dict[str, float]:
        """
        Evaluate metrics at a specific threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
        
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': fbeta_score(
                precision_score(y_true, y_pred, zero_division=0),
                recall_score(y_true, y_pred, zero_division=0),
                beta=self.config.f_beta
            )
        }
    
    def find_optimal_threshold(self, y_true: np.ndarray, 
                               y_pred_proba: np.ndarray) -> Tuple[float, Dict]:
        """
        Find optimal threshold for maximum F2-Score with min recall constraint.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        try:
            logger.info("=" * 60)
            logger.info("THRESHOLD OPTIMIZATION")
            logger.info("=" * 60)
            logger.info(f"Min Recall Requirement: {self.config.min_recall:.2%}")
            logger.info(f"Optimizing for: F{self.config.f_beta}-Score")
            
            # Search thresholds
            thresholds = np.arange(0.1, 0.91, self.config.threshold_step)
            results = []
            
            for threshold in thresholds:
                metrics = self.evaluate_threshold(y_true, y_pred_proba, threshold)
                results.append(metrics)
            
            # Filter by minimum recall requirement
            valid_results = [r for r in results if r['recall'] >= self.config.min_recall]
            
            if not valid_results:
                logger.warning(f"No threshold achieves {self.config.min_recall:.2%} recall!")
                logger.warning("Selecting threshold with maximum recall instead.")
                optimal = max(results, key=lambda x: x['recall'])
            else:
                # Select threshold with maximum F2-Score
                optimal = max(valid_results, key=lambda x: x['f2_score'])
            
            logger.info("\n--- Optimal Threshold Found ---")
            logger.info(f"  Threshold: {optimal['threshold']:.3f}")
            logger.info(f"  Recall: {optimal['recall']:.4f} ({optimal['recall']:.2%})")
            logger.info(f"  Precision: {optimal['precision']:.4f} ({optimal['precision']:.2%})")
            logger.info(f"  F1-Score: {optimal['f1_score']:.4f}")
            logger.info(f"  F2-Score: {optimal['f2_score']:.4f}")
            
            # Store all results for plotting
            self.threshold_results = results
            
            return optimal['threshold'], optimal
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def plot_threshold_analysis(self, y_true: np.ndarray, 
                                y_pred_proba: np.ndarray,
                                optimal_threshold: float,
                                save_path: str = None):
        """
        Plot threshold analysis: Precision-Recall curve and F2 vs Threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            optimal_threshold: Optimal threshold found
            save_path: Path to save plot
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Precision-Recall Curve
            precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
            
            axes[0].plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
            axes[0].axhline(y=self.config.min_recall, color='r', linestyle='--', 
                           label=f'Min Recall ({self.config.min_recall:.0%})')
            axes[0].scatter([self.threshold_results[int(optimal_threshold*100)]['recall']], 
                          [self.threshold_results[int(optimal_threshold*100)]['precision']], 
                          color='red', s=100, zorder=5, label=f'Optimal (t={optimal_threshold:.2f})')
            axes[0].set_xlabel('Recall', fontsize=12)
            axes[0].set_ylabel('Precision', fontsize=12)
            axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot 2: Metrics vs Threshold
            thresholds_list = [r['threshold'] for r in self.threshold_results]
            recalls_list = [r['recall'] for r in self.threshold_results]
            precisions_list = [r['precision'] for r in self.threshold_results]
            f2_scores_list = [r['f2_score'] for r in self.threshold_results]
            
            axes[1].plot(thresholds_list, recalls_list, 'b-', linewidth=2, label='Recall')
            axes[1].plot(thresholds_list, precisions_list, 'g-', linewidth=2, label='Precision')
            axes[1].plot(thresholds_list, f2_scores_list, 'r-', linewidth=2, label='F2-Score')
            axes[1].axvline(x=optimal_threshold, color='orange', linestyle='--', 
                           linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.2f})')
            axes[1].axhline(y=self.config.min_recall, color='gray', linestyle=':', 
                           alpha=0.5, label=f'Min Recall ({self.config.min_recall:.0%})')
            axes[1].set_xlabel('Threshold', fontsize=12)
            axes[1].set_ylabel('Score', fontsize=12)
            axes[1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_xlim([0.1, 0.9])
            axes[1].set_ylim([0, 1])
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.config.plots_dir, 'threshold_analysis.png')
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot threshold analysis: {e}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             threshold: float, save_path: str = None):
        """
        Plot confusion matrix at optimal threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
            save_path: Path to save plot
        """
        try:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['No Defect', 'Defect'],
                       yticklabels=['No Defect', 'Defect'])
            plt.title(f'Confusion Matrix (Threshold={threshold:.2f})', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            if save_path is None:
                save_path = os.path.join(self.config.plots_dir, 
                                        f'confusion_matrix_t{threshold:.2f}.png')
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
    
    def generate_threshold_report(self, optimal_threshold: float, 
                                  metrics: Dict, save_path: str = None) -> str:
        """
        Generate markdown report for threshold tuning results.
        
        Args:
            optimal_threshold: Optimal threshold found
            metrics: Metrics at optimal threshold
            save_path: Path to save report
        
        Returns:
            Report content as string
        """
        report = f"""# Threshold Tuning Report
## Steel Defect Detection System

### Optimization Goal
- **Metric**: F{self.config.f_beta}-Score (prioritizes Recall)
- **Constraint**: Recall ≥ {self.config.min_recall:.0%}

### Optimal Threshold
**Threshold**: `{optimal_threshold:.3f}`

### Performance Metrics

| Metric | Value | Percentage |
|--------|-------|------------|
| **Recall** | {metrics['recall']:.4f} | **{metrics['recall']:.2%}** ✅ |
| **Precision** | {metrics['precision']:.4f} | {metrics['precision']:.2%} |
| **F1-Score** | {metrics['f1_score']:.4f} | - |
| **F2-Score** | {metrics['f2_score']:.4f} | - |

### Interpretation

- **Recall {metrics['recall']:.2%}**: Model detects {metrics['recall']:.2%} of all defects
- **Precision {metrics['precision']:.2%}**: {metrics['precision']:.2%} of predicted defects are actual defects
- **F2-Score {metrics['f2_score']:.4f}**: Balanced metric favoring recall over precision

### Recommendation

✅ **Use threshold `{optimal_threshold:.3f}`** for production deployment.

This threshold achieves the minimum recall requirement while optimizing F2-Score.

---
*Generated by ThresholdTuner*
"""
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'threshold_report.md')
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Threshold report saved: {save_path}")
        return report

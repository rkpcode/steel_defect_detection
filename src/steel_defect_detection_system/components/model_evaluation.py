"""
Model Evaluation Component - Phase 5: Evaluation & Threshold Tuning
====================================================================
Steel Defect Detection System

LOCKED DESIGN DECISIONS:
- Recall is PRIMARY metric (≥95% target)
- Default 0.5 threshold is REJECTED
- Threshold tuned for minimum False Negatives
- F2-score for recall-weighted evaluation

This component handles:
1. Confusion Matrix generation
2. Precision, Recall, F2-score calculation
3. Precision-Recall curve plotting
4. Optimal threshold selection
5. Evaluation report generation
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, average_precision_score
)
import tensorflow as tf

from steel_defect_detection_system.exception import CustomException
from steel_defect_detection_system.logger import logger


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    
    # Paths
    evaluation_dir: str = os.path.join("artifacts", "evaluation")
    plots_dir: str = os.path.join("artifacts", "evaluation", "plots")
    
    # Threshold tuning
    default_threshold: float = 0.5
    min_recall_target: float = 0.95  # Target: 95% recall
    
    # F-beta score (beta=2 for recall-focused)
    f_beta: float = 2.0


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-beta score (recall-weighted when beta > 1)"""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


class ModelEvaluation:
    """
    Model Evaluation component.
    
    Responsibilities:
    1. Generate predictions on test set
    2. Calculate all evaluation metrics
    3. Plot confusion matrix and PR curve
    4. Find optimal threshold for minimum FN
    5. Generate evaluation report
    """
    
    def __init__(self, config: ModelEvaluationConfig = None):
        self.config = config or ModelEvaluationConfig()
        
        # Create directories
        os.makedirs(self.config.evaluation_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        
        self.results = {}
        
        logger.info("ModelEvaluation initialized")
    
    def predict(self, model: tf.keras.Model, 
                X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data.
        
        Returns:
            Tuple of (y_pred_proba, y_pred_binary)
        """
        logger.info(f"Generating predictions for {len(X_test)} samples...")
        
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred_binary = (y_pred_proba >= self.config.default_threshold).astype(int)
        
        return y_pred_proba, y_pred_binary
    
    def calculate_metrics(self, y_true: np.ndarray, 
                          y_pred_binary: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': np.mean(y_true == y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'f2_score': fbeta_score(
                precision_score(y_true, y_pred_binary, zero_division=0),
                recall_score(y_true, y_pred_binary, zero_division=0),
                beta=2.0
            ),
            'auc_roc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'average_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Handle confusion matrix for edge cases (single class in y_true or y_pred)
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        logger.info(f"Metrics calculated:")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  F2-Score: {metrics['f2_score']:.4f}")
        logger.info(f"  False Negatives: {metrics['false_negatives']}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, 
                              y_pred_binary: np.ndarray,
                              threshold: float = 0.5,
                              save_path: str = None) -> None:
        """Plot and save confusion matrix"""
        
        # Use labels=[0,1] to handle single-class edge cases
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Clean (0)', 'Defective (1)'],
                    yticklabels=['Clean (0)', 'Defective (1)'])
        plt.title(f'Confusion Matrix (Threshold={threshold:.2f})', fontsize=14)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Get values from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        plt.text(2.5, 0.5, f'FP={fp}\n(False Alarms)', fontsize=10, ha='left')
        plt.text(2.5, 1.5, f'FN={fn}\n(MISSED DEFECTS!)', fontsize=10, ha='left', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray,
                                     y_pred_proba: np.ndarray,
                                     optimal_threshold: float = None,
                                     save_path: str = None) -> None:
        """Plot Precision-Recall curve with threshold markers"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: PR Curve
        ax1 = axes[0]
        ax1.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
        ax1.fill_between(recall, precision, alpha=0.2)
        
        # Mark default 0.5 threshold
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        ax1.scatter([recall[idx_05]], [precision[idx_05]], 
                    color='red', s=100, zorder=5, label=f'Default 0.5')
        
        # Mark optimal threshold
        if optimal_threshold:
            idx_opt = np.argmin(np.abs(thresholds - optimal_threshold))
            ax1.scatter([recall[idx_opt]], [precision[idx_opt]], 
                        color='green', s=100, zorder=5, 
                        label=f'Optimal {optimal_threshold:.2f}')
        
        # Mark 95% recall line
        ax1.axvline(x=0.95, color='orange', linestyle='--', label='95% Recall Target')
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curve', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Threshold vs Metrics
        ax2 = axes[1]
        ax2.plot(thresholds, precision[:-1], 'b-', label='Precision')
        ax2.plot(thresholds, recall[:-1], 'r-', label='Recall')
        
        # Calculate F2 for each threshold
        f2_scores = [fbeta_score(p, r, 2.0) for p, r in zip(precision[:-1], recall[:-1])]
        ax2.plot(thresholds, f2_scores, 'g-', label='F2-Score')
        
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default 0.5')
        if optimal_threshold:
            ax2.axvline(x=optimal_threshold, color='green', linestyle='--', 
                        label=f'Optimal {optimal_threshold:.2f}')
        
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Metrics vs Threshold', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curve saved: {save_path}")
        
        plt.close()
    
    def find_optimal_threshold(self, y_true: np.ndarray,
                                y_pred_proba: np.ndarray) -> Tuple[float, Dict]:
        """
        Find optimal threshold for minimum False Negatives.
        
        Strategy:
        1. Find threshold that achieves target recall (95%)
        2. If not possible, find threshold with max F2-score
        
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        logger.info("Finding optimal threshold...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Strategy 1: Find threshold for target recall
        target_recall = self.config.min_recall_target
        valid_indices = np.where(recall[:-1] >= target_recall)[0]
        
        if len(valid_indices) > 0:
            # Among thresholds achieving target recall, pick highest precision
            best_idx = valid_indices[np.argmax(precision[:-1][valid_indices])]
            optimal_threshold = thresholds[best_idx]
            logger.info(f"Found threshold achieving {target_recall*100}% recall: {optimal_threshold:.3f}")
        else:
            # Strategy 2: Maximize F2-score
            f2_scores = [fbeta_score(p, r, 2.0) for p, r in zip(precision[:-1], recall[:-1])]
            best_idx = np.argmax(f2_scores)
            optimal_threshold = thresholds[best_idx]
            logger.info(f"Target recall not achievable. Using max F2-score threshold: {optimal_threshold:.3f}")
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        metrics = self.calculate_metrics(y_true, y_pred_optimal, y_pred_proba)
        metrics['threshold'] = optimal_threshold
        
        return optimal_threshold, metrics
    
    def generate_evaluation_report(self, 
                                    metrics_default: Dict,
                                    metrics_optimal: Dict,
                                    save_path: str = None) -> str:
        """Generate markdown evaluation report"""
        
        report = f"""# Phase 5: Model Evaluation Report

## Steel Defect Detection System

---

## 1. Default Threshold (0.5) Results

| Metric | Value |
|--------|-------|
| Recall | {metrics_default['recall']:.4f} |
| Precision | {metrics_default['precision']:.4f} |
| F2-Score | {metrics_default['f2_score']:.4f} |
| AUC-ROC | {metrics_default['auc_roc']:.4f} |

### Confusion Matrix
| | Predicted Clean | Predicted Defective |
|---|---|---|
| **Actual Clean** | TN={metrics_default['true_negatives']} | FP={metrics_default['false_positives']} |
| **Actual Defective** | FN={metrics_default['false_negatives']} | TP={metrics_default['true_positives']} |

> [!WARNING]
> Default 0.5 threshold: **{metrics_default['false_negatives']} missed defects (FN)**

---

## 2. Optimal Threshold ({metrics_optimal['threshold']:.3f}) Results

| Metric | Value | Change |
|--------|-------|--------|
| Recall | {metrics_optimal['recall']:.4f} | {'+' if metrics_optimal['recall'] > metrics_default['recall'] else ''}{(metrics_optimal['recall'] - metrics_default['recall'])*100:.1f}% |
| Precision | {metrics_optimal['precision']:.4f} | {'+' if metrics_optimal['precision'] > metrics_default['precision'] else ''}{(metrics_optimal['precision'] - metrics_default['precision'])*100:.1f}% |
| F2-Score | {metrics_optimal['f2_score']:.4f} | {'+' if metrics_optimal['f2_score'] > metrics_default['f2_score'] else ''}{(metrics_optimal['f2_score'] - metrics_default['f2_score'])*100:.1f}% |

### Confusion Matrix
| | Predicted Clean | Predicted Defective |
|---|---|---|
| **Actual Clean** | TN={metrics_optimal['true_negatives']} | FP={metrics_optimal['false_positives']} |
| **Actual Defective** | FN={metrics_optimal['false_negatives']} | TP={metrics_optimal['true_positives']} |

> [!NOTE]
> Optimal threshold: **{metrics_optimal['false_negatives']} missed defects (FN)**
> Reduction: **{metrics_default['false_negatives'] - metrics_optimal['false_negatives']} fewer missed defects**

---

## 3. Threshold Selection Rationale

**Business Requirement:** Minimize missed defects (False Negatives)

**Trade-off Accepted:**
- Lower threshold → Higher recall → More false alarms
- False alarms are acceptable (manual inspection)
- Missed defects are NOT acceptable

**Selected Threshold:** `{metrics_optimal['threshold']:.3f}`

---

## 4. Visualizations

- Confusion Matrix: `plots/confusion_matrix.png`
- PR Curve: `plots/precision_recall_curve.png`

---

## 5. Conclusion

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall | ≥95% | {metrics_optimal['recall']*100:.1f}% {'✅' if metrics_optimal['recall'] >= 0.95 else '⚠️'} |
| F2-Score | ≥0.85 | {metrics_optimal['f2_score']:.2f} {'✅' if metrics_optimal['f2_score'] >= 0.85 else '⚠️'} |
| Precision | ≥70% | {metrics_optimal['precision']*100:.1f}% {'✅' if metrics_optimal['precision'] >= 0.70 else '⚠️'} |
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Evaluation report saved: {save_path}")
        
        return report
    
    def initiate_model_evaluation(self, model: tf.keras.Model,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray) -> Dict:
        """
        Main method to run complete model evaluation.
        
        Args:
            model: Trained Keras model
            X_test: Test images
            y_test: Test labels
        
        Returns:
            Dictionary with all evaluation results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL EVALUATION")
            logger.info("=" * 60)
            
            # Generate predictions
            y_pred_proba, y_pred_binary = self.predict(model, X_test)
            
            # Metrics at default threshold
            logger.info("\n--- Metrics at Default Threshold (0.5) ---")
            metrics_default = self.calculate_metrics(y_test, y_pred_binary, y_pred_proba)
            metrics_default['threshold'] = 0.5
            
            # Find optimal threshold
            logger.info("\n--- Finding Optimal Threshold ---")
            optimal_threshold, metrics_optimal = self.find_optimal_threshold(
                y_test, y_pred_proba
            )
            
            # Generate predictions at optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Plot confusion matrices
            self.plot_confusion_matrix(
                y_test, y_pred_binary, threshold=0.5,
                save_path=os.path.join(self.config.plots_dir, 'confusion_matrix_default.png')
            )
            self.plot_confusion_matrix(
                y_test, y_pred_optimal, threshold=optimal_threshold,
                save_path=os.path.join(self.config.plots_dir, 'confusion_matrix_optimal.png')
            )
            
            # Plot PR curve
            self.plot_precision_recall_curve(
                y_test, y_pred_proba, optimal_threshold=optimal_threshold,
                save_path=os.path.join(self.config.plots_dir, 'precision_recall_curve.png')
            )
            
            # Generate report
            report = self.generate_evaluation_report(
                metrics_default, metrics_optimal,
                save_path=os.path.join(self.config.evaluation_dir, 'phase5_evaluation_report.md')
            )
            
            logger.info("=" * 60)
            logger.info("MODEL EVALUATION COMPLETE")
            logger.info("=" * 60)
            
            return {
                'metrics_default': metrics_default,
                'metrics_optimal': metrics_optimal,
                'optimal_threshold': optimal_threshold,
                'y_pred_proba': y_pred_proba,
                'report': report
            }
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test evaluation component
    config = ModelEvaluationConfig()
    evaluator = ModelEvaluation(config)
    
    # Create dummy data for testing
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3, 0.7, 0.5, 0.85])
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    metrics = evaluator.calculate_metrics(y_true, y_pred_binary, y_pred_proba)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

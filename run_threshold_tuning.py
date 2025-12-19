"""
Run Threshold Tuning - Steel Defect Detection System
====================================================

Optimize decision threshold on trained model to maximize F2-Score
while maintaining minimum recall requirement (≥95%).

Usage:
    python run_threshold_tuning.py
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.steel_defect_detection_system.components.threshold_tuner import ThresholdTuner, ThresholdTunerConfig
from src.steel_defect_detection_system.components.data_transformation import DataTransformation, DataTransformationConfig
from src.steel_defect_detection_system.logger import logger


def extract_labels_and_predictions(model, test_dataset, validation_steps):
    """
    Extract true labels and predicted probabilities from test dataset.
    
    Args:
        model: Trained Keras model
        test_dataset: Test tf.data.Dataset
        validation_steps: Number of validation steps
    
    Returns:
        Tuple of (y_true, y_pred_proba)
    """
    logger.info("Extracting predictions from test dataset...")
    
    y_true_list = []
    y_pred_proba_list = []
    
    # Iterate through test dataset
    for i, (images, labels) in enumerate(test_dataset):
        if i >= validation_steps:
            break
        
        # Get predictions
        predictions = model.predict(images, verbose=0)
        
        # Store results
        y_true_list.append(labels.numpy())
        y_pred_proba_list.append(predictions.flatten())
    
    # Concatenate all batches
    y_true = np.concatenate(y_true_list)
    y_pred_proba = np.concatenate(y_pred_proba_list)
    
    logger.info(f"Extracted {len(y_true)} samples")
    logger.info(f"Positive samples: {np.sum(y_true)} ({np.mean(y_true):.2%})")
    
    return y_true, y_pred_proba


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("THRESHOLD TUNING PIPELINE")
    logger.info("Steel Defect Detection System")
    logger.info("=" * 60)
    
    # Define custom F2 score metric (same as in model_trainer.py)
    def f2_score(y_true, y_pred):
        """Custom F2 score metric for model loading."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
        fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        
        f2 = 5 * precision * recall / (4 * precision + recall + tf.keras.backend.epsilon())
        return f2
    
    # Load trained model with custom objects
    model_path = "artifacts/models/transfer_model_best.keras"
    logger.info(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={'f2_score': f2_score})
    logger.info("✅ Model loaded successfully")
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    transformer = DataTransformation(DataTransformationConfig())
    
    # Load test data
    test_csv = "artifacts/data/processed/test.csv"
    result = transformer.initiate_data_transformation(
        train_path="artifacts/data/processed/train.csv",
        test_path=test_csv
    )
    
    test_dataset = result['test_dataset']
    validation_steps = result['test_patches'] // transformer.config.batch_size
    
    logger.info(f"✅ Test dataset loaded: {result['test_patches']} patches")
    
    # Extract predictions
    y_true, y_pred_proba = extract_labels_and_predictions(
        model, test_dataset, validation_steps
    )
    
    # Run threshold tuning
    tuner = ThresholdTuner(ThresholdTunerConfig(min_recall=0.95, f_beta=2.0))
    
    optimal_threshold, metrics = tuner.find_optimal_threshold(y_true, y_pred_proba)
    
    # Generate visualizations
    logger.info("\nGenerating threshold analysis plots...")
    tuner.plot_threshold_analysis(y_true, y_pred_proba, optimal_threshold)
    tuner.plot_confusion_matrix(y_true, y_pred_proba, optimal_threshold)
    
    # Generate report
    logger.info("\nGenerating threshold tuning report...")
    report = tuner.generate_threshold_report(optimal_threshold, metrics)
    
    # Save optimal threshold
    threshold_file = "artifacts/threshold_tuning/optimal_threshold.txt"
    with open(threshold_file, 'w') as f:
        f.write(f"{optimal_threshold:.4f}\n")
        f.write(f"# Optimal threshold for F2-Score optimization\n")
        f.write(f"# Recall: {metrics['recall']:.4f}\n")
        f.write(f"# Precision: {metrics['precision']:.4f}\n")
        f.write(f"# F2-Score: {metrics['f2_score']:.4f}\n")
    
    logger.info(f"✅ Optimal threshold saved: {threshold_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n📊 Results:")
    logger.info(f"   Optimal Threshold: {optimal_threshold:.3f}")
    logger.info(f"   Recall: {metrics['recall']:.4f} ({metrics['recall']:.2%})")
    logger.info(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})")
    logger.info(f"   F2-Score: {metrics['f2_score']:.4f}")
    logger.info(f"\n📁 Artifacts:")
    logger.info(f"   - Threshold analysis: artifacts/threshold_tuning/threshold_analysis.png")
    logger.info(f"   - Confusion matrix: artifacts/threshold_tuning/confusion_matrix_t{optimal_threshold:.2f}.png")
    logger.info(f"   - Report: artifacts/threshold_tuning/threshold_report.md")
    logger.info(f"   - Optimal threshold: {threshold_file}")

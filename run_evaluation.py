"""Run model evaluation and generate plots"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import tensorflow as tf
from steel_defect_detection_system.pipelines.training_pipeline import TrainingPipeline
from steel_defect_detection_system.components.model_evaluation import ModelEvaluation

# Custom metric for model loading
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return (5 * precision * recall) / (4 * precision + recall + 1e-7)

print("Loading model...")
model = tf.keras.models.load_model(
    'artifacts/models/transfer_model_stage1_best.keras', 
    custom_objects={'f2_score': f2_score}
)

print("Loading test data...")
pipeline = TrainingPipeline()
result = pipeline.run_step_2_data_transformation(
    train_path='artifacts/data/processed/train.csv',
    test_path='artifacts/data/processed/test.csv',
    max_train_images=5,
    max_test_images=100
)
X_test = result['X_test']
y_test = result['y_test']

print(f"Test samples: {len(X_test)}")
print(f"Defective: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")

print("\nRunning evaluation...")
evaluator = ModelEvaluation()
eval_result = evaluator.initiate_model_evaluation(model, X_test, y_test)

print("\n" + "="*50)
print("EVALUATION COMPLETE!")
print("="*50)
print(f"Optimal threshold: {eval_result['optimal_threshold']:.3f}")
print(f"Results saved to: artifacts/evaluation/")

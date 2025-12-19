"""
Model Trainer Component - Phase 4: Modeling Strategy
=====================================================
Steel Defect Detection System

LOCKED DESIGN DECISIONS:
- Class-weighted loss (mandatory for imbalance)
- Recall-focused training
- Baseline CNN first, then transfer learning
- F2-score as secondary metric

Models:
1. Baseline CNN - Simple model for sanity check
2. Transfer Learning - ResNet50/EfficientNet with fine-tuning
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0

from src.steel_defect_detection_system.exception import CustomException
from src.steel_defect_detection_system.logger import logger
from src.steel_defect_detection_system.utils.mlflow_utils import get_tracker, log_training_metrics

@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    
    # Paths
    model_dir: str = os.path.join("artifacts", "models")
    logs_dir: str = os.path.join("artifacts", "training_logs")
    
    # Model configuration
    input_shape: Tuple[int, int, int] = (256, 256, 3)
    num_classes: int = 1  # Binary classification (sigmoid output)
    
    # Training configuration
    learning_rate: float = 1e-4
    epochs: int = 20
    early_stopping_patience: int = 5
    
    # Transfer learning
    backbone: str = "efficientnet"  # "resnet50" or "efficientnet"
    freeze_backbone: bool = True
    fine_tune_at: int = 100  # Unfreeze layers after this index


class MLflowCallback(keras.callbacks.Callback):
    """Custom callback to log metrics to MLflow after each epoch."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics to MLflow at the end of each epoch."""
        if logs:
            log_training_metrics(logs, epoch + 1)


class BaselineCNN:
    """
    Simple CNN for sanity check.
    
    Purpose:
    - Detect data leakage
    - Detect learning collapse (predicting only majority class)
    - Establish baseline performance
    """
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> keras.Model:
        """Build simple baseline CNN"""
        
        model = models.Sequential([
            # Input normalization (if not done in preprocessing)
            layers.Input(shape=input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary output
        ])
        
        logger.info("Baseline CNN built")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model


class TransferLearningModel:
    """
    Transfer learning model with pre-trained backbone.
    
    Options:
    - ResNet50: Good for general features
    - EfficientNetB0: Better efficiency for similar accuracy
    """
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int] = (256, 256, 3),
              backbone: str = "efficientnet",
              freeze_backbone: bool = True) -> keras.Model:
        """Build transfer learning model"""
        
        # Select backbone
        if backbone == "resnet50":
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:  # efficientnet
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        # Freeze backbone if specified
        base_model.trainable = not freeze_backbone
        
        # Build model
        inputs = keras.Input(shape=input_shape)
        
        # Data augmentation can be added here if needed
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        logger.info(f"Transfer learning model built ({backbone})")
        logger.info(f"Backbone trainable: {not freeze_backbone}")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    @staticmethod
    def unfreeze_for_fine_tuning(model: keras.Model, fine_tune_at: int = 100):
        """Unfreeze backbone layers for fine-tuning"""
        
        # Find the base model (first layer after input)
        base_model = model.layers[1]
        base_model.trainable = True
        
        # Freeze layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        logger.info(f"Model unfrozen for fine-tuning at layer {fine_tune_at}")
        
        return model


class ModelTrainer:
    """
    Main Model Trainer component.
    
    Responsibilities:
    1. Build baseline or transfer learning model
    2. Compile with class-weighted loss
    3. Train with callbacks (early stopping, checkpoints)
    4. Monitor recall and F2-score
    5. Save best model
    """
    
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        
        # Create directories
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        self.model = None
        self.history = None
        
        logger.info("ModelTrainer initialized")
    
    def build_model(self, model_type: str = "transfer") -> keras.Model:
        """
        Build model based on type.
        
        Args:
            model_type: "baseline" or "transfer"
        """
        if model_type == "baseline":
            self.model = BaselineCNN.build(self.config.input_shape)
        else:
            self.model = TransferLearningModel.build(
                input_shape=self.config.input_shape,
                backbone=self.config.backbone,
                freeze_backbone=self.config.freeze_backbone
            )
        
        return self.model
    
    def compile_model(self, class_weights: Dict[int, float] = None):
        """
        Compile model with class-weighted loss.
        
        Uses Binary Crossentropy with class weights for imbalance handling.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Custom F2 score metric (recall-weighted)
        def f2_score(y_true, y_pred):
            # Cast to float32 to avoid type mismatch
            y_true = tf.cast(y_true, tf.float32)
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
            
            tp = tf.reduce_sum(y_true * y_pred_binary)
            fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
            fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
            
            precision = tp / (tp + fp + tf.keras.backend.epsilon())
            recall = tp / (tp + fn + tf.keras.backend.epsilon())
            
            # F2 = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
            f2 = 5 * precision * recall / (4 * precision + recall + tf.keras.backend.epsilon())
            return f2
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                f2_score
            ]
        )
        
        logger.info("Model compiled with recall-focused metrics")
        if class_weights:
            logger.info(f"Class weights will be used: {class_weights}")
    
    def get_callbacks(self, model_name: str = "model") -> list:
        """Get training callbacks"""
        
        callbacks_list = [
            # Early stopping based on validation recall
            callbacks.EarlyStopping(
                monitor='val_recall',
                patience=self.config.early_stopping_patience,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.model_dir, f"{model_name}_best.keras"),
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config.logs_dir, model_name),
                histogram_freq=1
            ),
            
            # MLflow logging
            MLflowCallback()
        ]
        
        return callbacks_list
    
    def train(self, train_dataset: tf.data.Dataset, 
              val_dataset: tf.data.Dataset,
              class_weights: Dict[int, float] = None,
              model_name: str = "model",
              steps_per_epoch: int = None,
              validation_steps: int = None) -> dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training TF dataset
            val_dataset: Validation TF dataset
            class_weights: Class weights for imbalance
            model_name: Name for saving
            steps_per_epoch: Number of steps per epoch (required if dataset uses repeat())
            validation_steps: Number of validation steps (required if dataset uses repeat())
        
        Returns:
            Training history
        """
        try:
            logger.info("=" * 60)
            logger.info(f"TRAINING: {model_name}")
            logger.info("=" * 60)
            
            if self.model is None:
                raise ValueError("Model not built. Call build_model first.")
            
            # Get callbacks
            callbacks_list = self.get_callbacks(model_name)
            
            # Log training info
            if steps_per_epoch:
                logger.info(f"Steps per epoch: {steps_per_epoch}")
            if validation_steps:
                logger.info(f"Validation steps: {validation_steps}")
            
            # Train
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.epochs,
                class_weight=class_weights,
                callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )
            
            # Log final metrics
            logger.info("\nTraining Complete!")
            logger.info(f"Final train recall: {self.history.history['recall'][-1]:.4f}")
            logger.info(f"Final val recall: {self.history.history['val_recall'][-1]:.4f}")
            
            # Save final model
            final_path = os.path.join(self.config.model_dir, f"{model_name}_final.keras")
            self.model.save(final_path)
            logger.info(f"Model saved to: {final_path}")
            
            return self.history.history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_with_fine_tuning(self, train_dataset: tf.data.Dataset,
                               val_dataset: tf.data.Dataset,
                               class_weights: Dict[int, float] = None,
                               model_name: str = "transfer") -> dict:
        """
        Two-stage training: frozen backbone then fine-tuning.
        
        Stage 1: Train with frozen backbone
        Stage 2: Unfreeze and fine-tune with lower learning rate
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 1: TRANSFER LEARNING (FROZEN BACKBONE)")
            logger.info("=" * 60)
            
            # Stage 1: Frozen backbone
            history1 = self.train(
                train_dataset, val_dataset,
                class_weights=class_weights,
                model_name=f"{model_name}_stage1"
            )
            
            logger.info("=" * 60)
            logger.info("STAGE 2: FINE-TUNING (UNFROZEN)")
            logger.info("=" * 60)
            
            # Unfreeze for fine-tuning
            self.model = TransferLearningModel.unfreeze_for_fine_tuning(
                self.model, 
                fine_tune_at=self.config.fine_tune_at
            )
            
            # Recompile with lower learning rate
            self.config.learning_rate = self.config.learning_rate / 10
            self.compile_model(class_weights)
            
            # Stage 2: Fine-tune
            history2 = self.train(
                train_dataset, val_dataset,
                class_weights=class_weights,
                model_name=f"{model_name}_stage2"
            )
            
            # Combine histories
            combined_history = {
                k: history1.get(k, []) + history2.get(k, [])
                for k in set(list(history1.keys()) + list(history2.keys()))
            }
            
            return combined_history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(self, train_dataset: tf.data.Dataset,
                                 val_dataset: tf.data.Dataset,
                                 class_weights: Dict[int, float] = None,
                                 model_type: str = "transfer",
                                 fine_tune: bool = True,
                                 steps_per_epoch: int = None,
                                 validation_steps: int = None) -> Dict:
        """
        Main method to run model training.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            class_weights: For imbalance handling
            model_type: "baseline" or "transfer"
            fine_tune: Whether to do fine-tuning (transfer only)
            steps_per_epoch: Number of training steps per epoch
            validation_steps: Number of validation steps per epoch
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("=" * 60)
            
            # Build model
            self.build_model(model_type)
            
            # Compile
            self.compile_model(class_weights)
            
            # Train
            if model_type == "transfer" and fine_tune:
                history = self.train_with_fine_tuning(
                    train_dataset, val_dataset,
                    class_weights=class_weights,
                    model_name="transfer_model"
                )
            else:
                history = self.train(
                    train_dataset, val_dataset,
                    class_weights=class_weights,
                    model_name=f"{model_type}_model",
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps
                )
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING COMPLETE")
            logger.info("=" * 60)
            
            return {
                'model': self.model,
                'history': history,
                'config': self.config
            }
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test model building
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config)
    
    # Build baseline
    print("\n--- Baseline CNN ---")
    baseline = trainer.build_model("baseline")
    baseline.summary()
    
    # Build transfer learning
    print("\n--- Transfer Learning (EfficientNet) ---")
    trainer2 = ModelTrainer(config)
    transfer = trainer2.build_model("transfer")
    transfer.summary()

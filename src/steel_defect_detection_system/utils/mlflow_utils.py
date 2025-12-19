"""
MLflow Utilities for Steel Defect Detection System
===================================================

Handles MLflow experiment tracking and DagsHub integration.
"""

import os
import mlflow
import dagshub
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from steel_defect_detection_system.logger import logger


# Load environment variables
load_dotenv()


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.
    
    Features:
    - DagsHub integration
    - Automatic experiment setup
    - Metric and parameter logging
    - Model artifact logging
    """
    
    def __init__(self, experiment_name: str = "Steel-Defect-Detection"):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.run_id = None
        self._setup_dagshub()
        self._setup_experiment()
    
    def _setup_dagshub(self):
        """Setup DagsHub connection if credentials are available."""
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        dagshub_user = os.getenv("DAGSHUB_USER_NAME")
        dagshub_repo = os.getenv("DAGSHUB_REPO_NAME", "steel_defect_detection")
        
        if dagshub_token and dagshub_user:
            try:
                # Initialize DagsHub
                dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
                logger.info(f"✅ DagsHub connected: {dagshub_user}/{dagshub_repo}")
            except Exception as e:
                logger.warning(f"⚠️ DagsHub connection failed: {e}")
                logger.info("Continuing with local MLflow tracking...")
        else:
            logger.info("ℹ️ DagsHub credentials not found. Using local MLflow tracking.")
            logger.info("Set DAGSHUB_TOKEN and DAGSHUB_USER_NAME in .env for DagsHub integration.")
    
    def _setup_experiment(self):
        """Setup MLflow experiment."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"📊 MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
    
    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Optional tags for the run
        """
        try:
            mlflow.start_run(run_name=run_name, tags=tags or {})
            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"🚀 Started MLflow run: {run_name} (ID: {self.run_id})")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        try:
            mlflow.log_params(params)
            logger.info(f"📝 Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number (e.g., epoch)
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric to MLflow.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact (file) to MLflow.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"📦 Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log a Keras model to MLflow.
        
        Args:
            model: Keras model
            artifact_path: Path within artifact store
        """
        try:
            mlflow.keras.log_model(model, artifact_path)
            logger.info(f"🤖 Logged model to: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info(f"✅ Ended MLflow run: {self.run_id}")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag {key}: {e}")


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker(experiment_name: str = "Steel-Defect-Detection") -> MLflowTracker:
    """
    Get or create the global MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        MLflowTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(experiment_name)
    return _tracker


def log_training_metrics(history: Dict, epoch: int):
    """
    Helper to log training metrics from Keras history.
    
    Args:
        history: Keras history dictionary
        epoch: Current epoch number
    """
    tracker = get_tracker()
    
    # Training metrics
    if 'loss' in history:
        tracker.log_metric('train_loss', history['loss'], step=epoch)
    if 'accuracy' in history:
        tracker.log_metric('train_accuracy', history['accuracy'], step=epoch)
    if 'recall' in history:
        tracker.log_metric('train_recall', history['recall'], step=epoch)
    if 'precision' in history:
        tracker.log_metric('train_precision', history['precision'], step=epoch)
    if 'auc' in history:
        tracker.log_metric('train_auc', history['auc'], step=epoch)
    if 'f2_score' in history:
        tracker.log_metric('train_f2_score', history['f2_score'], step=epoch)
    
    # Validation metrics
    if 'val_loss' in history:
        tracker.log_metric('val_loss', history['val_loss'], step=epoch)
    if 'val_accuracy' in history:
        tracker.log_metric('val_accuracy', history['val_accuracy'], step=epoch)
    if 'val_recall' in history:
        tracker.log_metric('val_recall', history['val_recall'], step=epoch)
    if 'val_precision' in history:
        tracker.log_metric('val_precision', history['val_precision'], step=epoch)
    if 'val_auc' in history:
        tracker.log_metric('val_auc', history['val_auc'], step=epoch)
    if 'val_f2_score' in history:
        tracker.log_metric('val_f2_score', history['val_f2_score'], step=epoch)

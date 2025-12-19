"""
Utility Functions for Steel Defect Detection System
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from steel_defect_detection_system.logger import logger

# MLflow imports (optional - will gracefully fail if not installed)
try:
    import mlflow
    import dagshub
    from dotenv import load_dotenv
    MLFLOW_AVAILABLE = True
    load_dotenv()
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not available. Install with: pip install mlflow dagshub python-dotenv")


# ============================================================================
# RLE Encoding/Decoding Functions
# ============================================================================

def rle_to_mask(rle_string: str, height: int = 256, width: int = 1600) -> np.ndarray:
    """
    Convert Run-Length Encoding (RLE) string to binary mask.
    
    Severstal RLE format: pairs of (start_position, run_length)
    Position is 1-indexed, column-major order
    
    Args:
        rle_string: RLE encoded string (space-separated)
        height: Image height (default 256)
        width: Image width (default 1600)
    
    Returns:
        Binary mask of shape (height, width)
    """
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    # Parse RLE string
    rle_numbers = [int(x) for x in rle_string.split()]
    starts = rle_numbers[0::2]
    lengths = rle_numbers[1::2]
    
    # Create flat mask (column-major order for Severstal)
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    
    for start, length in zip(starts, lengths):
        # RLE is 1-indexed
        start_idx = start - 1
        mask_flat[start_idx:start_idx + length] = 1
    
    # Reshape to (height, width) - column major order
    mask = mask_flat.reshape((height, width), order='F')
    
    return mask


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Convert binary mask to RLE string.
    
    Args:
        mask: Binary mask of shape (height, width)
    
    Returns:
        RLE encoded string
    """
    # Flatten in column-major order
    flat = mask.flatten(order='F')
    
    # Find runs
    flat = np.concatenate([[0], flat, [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


# ============================================================================
# Class Weight Calculation
# ============================================================================

def calculate_class_weights(labels: np.ndarray, num_classes: int = 5) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset.
    
    Uses inverse frequency weighting.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes (0=No defect, 1-4=Defect types)
    
    Returns:
        Dictionary of class_id -> weight
    """
    class_counts = np.bincount(labels.astype(int), minlength=num_classes)
    total_samples = len(labels)
    
    # Inverse frequency weighting: w = N / (n_classes * count)
    weights = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            weights[i] = 1.0
    
    logger.info(f"Class weights calculated: {weights}")
    return weights


# ============================================================================
# Defect Analysis Functions
# ============================================================================

def calculate_defect_coverage(mask: np.ndarray) -> float:
    """
    Calculate what percentage of image is covered by defect.
    
    Args:
        mask: Binary mask
    
    Returns:
        Percentage of pixels that are defective
    """
    total_pixels = mask.size
    defect_pixels = np.sum(mask)
    return (defect_pixels / total_pixels) * 100


def get_defect_class_name(class_id: int) -> str:
    """
    Get human-readable defect class name.
    
    Args:
        class_id: Defect class (1-4)
    
    Returns:
        Class name string
    """
    class_names = {
        0: "No Defect",
        1: "Pitted Surface",
        2: "Crazing",
        3: "Scratches",
        4: "Patches"
    }
    return class_names.get(class_id, "Unknown")


# ============================================================================
# MLflow Experiment Tracking (Optional)
# ============================================================================

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
        """Initialize MLflow tracker."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Tracking disabled.")
            return
            
        self.experiment_name = experiment_name
        self.run_id = None
        self._setup_dagshub()
        self._setup_experiment()
    
    def _setup_dagshub(self):
        """Setup DagsHub connection if credentials are available."""
        if not MLFLOW_AVAILABLE:
            return
            
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        dagshub_user = os.getenv("DAGSHUB_USER_NAME")
        dagshub_repo = os.getenv("DAGSHUB_REPO_NAME", "steel_defect_detection")
        
        if dagshub_token and dagshub_user:
            try:
                dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
                logger.info(f"✅ DagsHub connected: {dagshub_user}/{dagshub_repo}")
            except Exception as e:
                logger.warning(f"⚠️ DagsHub connection failed: {e}")
                logger.info("Continuing with local MLflow tracking...")
        else:
            logger.info("ℹ️ DagsHub credentials not found. Using local MLflow tracking.")
    
    def _setup_experiment(self):
        """Setup MLflow experiment."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"📊 MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
    
    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Start a new MLflow run."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.start_run(run_name=run_name, tags=tags or {})
            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"🚀 Started MLflow run: {run_name}")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.log_params(params)
            logger.info(f"📝 Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.end_run()
            logger.info(f"✅ Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker(experiment_name: str = "Steel-Defect-Detection") -> MLflowTracker:
    """Get or create the global MLflow tracker."""
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
    if not MLFLOW_AVAILABLE:
        return
        
    tracker = get_tracker()
    
    # Training metrics
    for metric in ['loss', 'accuracy', 'recall', 'precision', 'auc', 'f2_score']:
        if metric in history:
            tracker.log_metric(f'train_{metric}', history[metric], step=epoch)
    
    # Validation metrics
    for metric in ['loss', 'accuracy', 'recall', 'precision', 'auc', 'f2_score']:
        val_metric = f'val_{metric}'
        if val_metric in history:
            tracker.log_metric(val_metric, history[val_metric], step=epoch)

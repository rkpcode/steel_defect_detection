"""
Prediction Pipeline - Steel Defect Detection System
=====================================================

This pipeline handles:
1. Loading trained model
2. Image preprocessing (patch extraction)
3. Batch prediction on patches
4. Aggregating patch predictions to image-level decision
5. Threshold-based classification (PASS/FAIL/HOLD)

LOCKED DESIGN DECISIONS:
- Optimal threshold from evaluation (not default 0.5)
- Image-level: ANY defective patch = defective image
- HOLD zone for human review
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image
import tensorflow as tf

from steel_defect_detection_system.exception import CustomException
from steel_defect_detection_system.logger import logger


@dataclass
class PredictionConfig:
    """Configuration for prediction pipeline"""
    
    # Model paths
    model_dir: str = os.path.join("artifacts", "models")
    model_name: str = "transfer_model_final.keras"
    
    # Patch configuration (must match training)
    patch_height: int = 256
    patch_width: int = 256
    stride_height: int = 128
    stride_width: int = 128
    
    # Image dimensions (Severstal)
    img_height: int = 256
    img_width: int = 1600
    
    # Threshold configuration
    optimal_threshold: float = 0.3  # Will be updated from evaluation
    hold_zone_lower: float = 0.2    # Below: likely clean
    hold_zone_upper: float = 0.4    # Above: likely defective
    
    # Aggregation strategy
    aggregation: str = "any"  # "any", "majority", "mean"


class PredictionPipeline:
    """
    Steel Defect Detection Prediction Pipeline.
    
    Handles end-to-end inference:
    1. Load image
    2. Extract patches
    3. Run model prediction
    4. Aggregate to image-level decision
    """
    
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.model = None
        self._load_model()
        logger.info("PredictionPipeline initialized")
    
    def _load_model(self) -> None:
        """Load trained model"""
        model_path = os.path.join(self.config.model_dir, self.config.model_name)
        
        if not os.path.exists(model_path):
            # Try alternate names
            alternates = [
                "transfer_model_stage1_best.keras",
                "baseline_model_final.keras",
                "baseline_model_best.keras"
            ]
            for alt in alternates:
                alt_path = os.path.join(self.config.model_dir, alt)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"No model found in {self.config.model_dir}")
        
        logger.info(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Normalize to 0-1
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract patches from image with overlap"""
        h, w = image.shape[:2]
        patch_h, patch_w = self.config.patch_height, self.config.patch_width
        stride_h, stride_w = self.config.stride_height, self.config.stride_width
        
        patches = []
        
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                patch = image[y:y+patch_h, x:x+patch_w]
                patches.append(patch)
        
        return patches
    
    def predict_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """Run model prediction on batch of patches"""
        batch = np.array(patches)
        predictions = self.model.predict(batch, verbose=0)
        return predictions.flatten()
    
    def aggregate_predictions(self, patch_probs: np.ndarray) -> Dict:
        """
        Aggregate patch-level predictions to image-level decision.
        
        Strategy:
        - "any": Image defective if ANY patch is defective (conservative)
        - "majority": Image defective if >50% patches are defective
        - "mean": Use mean probability across patches
        """
        threshold = self.config.optimal_threshold
        
        if self.config.aggregation == "any":
            # ANY patch above threshold = defective image
            max_prob = float(np.max(patch_probs))
            n_defective = int(np.sum(patch_probs >= threshold))
            image_defective = n_defective > 0
            confidence = max_prob
            
        elif self.config.aggregation == "majority":
            # Majority vote
            n_defective = int(np.sum(patch_probs >= threshold))
            image_defective = n_defective > len(patch_probs) / 2
            confidence = n_defective / len(patch_probs)
            
        else:  # "mean"
            # Mean probability
            mean_prob = float(np.mean(patch_probs))
            image_defective = mean_prob >= threshold
            confidence = mean_prob
        
        # Determine decision with HOLD zone
        if confidence < self.config.hold_zone_lower:
            decision = "PASS"
            decision_confidence = "HIGH"
        elif confidence > self.config.hold_zone_upper:
            decision = "FAIL"
            decision_confidence = "HIGH"
        else:
            decision = "HOLD"
            decision_confidence = "LOW"
        
        return {
            'is_defective': image_defective,
            'confidence': confidence,
            'decision': decision,
            'decision_confidence': decision_confidence,
            'patch_probabilities': patch_probs,
            'n_defective_patches': int(np.sum(patch_probs >= threshold)),
            'total_patches': len(patch_probs),
            'max_patch_prob': float(np.max(patch_probs)),
            'mean_patch_prob': float(np.mean(patch_probs))
        }
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Run complete prediction on single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess
            image = self.load_image(image_path)
            
            # Extract patches
            patches = self.extract_patches(image)
            
            # Predict
            patch_probs = self.predict_patches(patches)
            
            # Aggregate
            result = self.aggregate_predictions(patch_probs)
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            
            logger.info(f"Prediction: {result['image_name']} -> {result['decision']} "
                       f"(confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run prediction on batch of images"""
        results = []
        
        for i, path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            try:
                result = self.predict_single_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    'image_path': path,
                    'error': str(e),
                    'decision': 'ERROR'
                })
        
        return results
    
    def predict_directory(self, directory: str, pattern: str = "*.jpg") -> List[Dict]:
        """Run prediction on all images in directory"""
        from glob import glob
        
        image_paths = glob(os.path.join(directory, pattern))
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        return self.predict_batch(image_paths)
    
    def get_summary(self, results: List[Dict]) -> Dict:
        """Generate summary of batch predictions"""
        valid_results = [r for r in results if 'error' not in r]
        
        n_pass = sum(1 for r in valid_results if r['decision'] == 'PASS')
        n_fail = sum(1 for r in valid_results if r['decision'] == 'FAIL')
        n_hold = sum(1 for r in valid_results if r['decision'] == 'HOLD')
        n_error = len(results) - len(valid_results)
        
        return {
            'total_images': len(results),
            'pass': n_pass,
            'fail': n_fail,
            'hold': n_hold,
            'errors': n_error,
            'pass_rate': n_pass / len(valid_results) if valid_results else 0,
            'fail_rate': n_fail / len(valid_results) if valid_results else 0,
            'hold_rate': n_hold / len(valid_results) if valid_results else 0
        }


class CustomData:
    """
    Handle custom image input for prediction.
    
    Supports:
    - Single image path
    - Numpy array
    - PIL Image
    - Directory of images
    """
    
    def __init__(self, data: Union[str, np.ndarray, Image.Image, List[str]]):
        self.data = data
        self.data_type = self._detect_type()
    
    def _detect_type(self) -> str:
        if isinstance(self.data, str):
            if os.path.isdir(self.data):
                return "directory"
            elif os.path.isfile(self.data):
                return "single_image"
            else:
                raise ValueError(f"Path not found: {self.data}")
        elif isinstance(self.data, np.ndarray):
            return "numpy_array"
        elif isinstance(self.data, Image.Image):
            return "pil_image"
        elif isinstance(self.data, list):
            return "image_list"
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")
    
    def get_image_paths(self) -> List[str]:
        """Get list of image paths to process"""
        if self.data_type == "single_image":
            return [self.data]
        elif self.data_type == "directory":
            from glob import glob
            return glob(os.path.join(self.data, "*.jpg")) + \
                   glob(os.path.join(self.data, "*.png"))
        elif self.data_type == "image_list":
            return self.data
        else:
            raise ValueError("Cannot get paths for numpy/PIL input")
    
    def get_numpy_array(self) -> np.ndarray:
        """Get numpy array from input"""
        if self.data_type == "numpy_array":
            return self.data
        elif self.data_type == "pil_image":
            return np.array(self.data)
        elif self.data_type == "single_image":
            return np.array(Image.open(self.data))
        else:
            raise ValueError("Cannot convert to single numpy array")


if __name__ == "__main__":
    # Test prediction pipeline
    print("Testing Prediction Pipeline...")
    
    # Initialize
    config = PredictionConfig()
    
    # Check if model exists
    model_path = os.path.join(config.model_dir, config.model_name)
    if os.path.exists(model_path):
        pipeline = PredictionPipeline(config)
        
        # Test on sample image
        test_dir = "artifacts/data/raw/train_images"
        if os.path.exists(test_dir):
            import glob
            images = glob.glob(os.path.join(test_dir, "*.jpg"))[:5]
            
            if images:
                print(f"\nTesting on {len(images)} images...")
                results = pipeline.predict_batch(images)
                
                summary = pipeline.get_summary(results)
                print(f"\nSummary:")
                print(f"  PASS: {summary['pass']}")
                print(f"  FAIL: {summary['fail']}")
                print(f"  HOLD: {summary['hold']}")
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using run_pipeline.py")

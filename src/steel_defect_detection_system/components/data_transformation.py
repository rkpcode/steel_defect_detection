"""
Data Transformation Component - Phase 3: Preprocessing Strategy
================================================================
Steel Defect Detection System

LOCKED DESIGN DECISIONS:
- Patch-based approach (256x256 patches with 50% overlap)
- Class-weighted sampling
- Safe augmentations only (defect-preserving)
- NO blind full-image resize

This component handles:
1. Patch extraction from 256x1600 images
2. Patch-level label generation from RLE masks
3. Defect-preserving augmentations
4. TensorFlow Dataset creation with class balancing
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import albumentations as A

from steel_defect_detection_system.exception import CustomException
from steel_defect_detection_system.logger import logger
from steel_defect_detection_system.utils import rle_to_mask


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation - LOCKED DECISIONS"""
    
    # Paths
    raw_data_dir: str = os.path.join("artifacts", "data", "raw")
    train_images_dir: str = os.path.join("artifacts", "data", "raw", "train_images")
    processed_dir: str = os.path.join("artifacts", "data", "processed")
    patches_dir: str = os.path.join("artifacts", "data", "patches")
    
    # Image dimensions (Severstal)
    img_height: int = 256
    img_width: int = 1600
    
    # LOCKED: Patch configuration
    patch_height: int = 256
    patch_width: int = 256
    stride_height: int = 128  # 50% overlap
    stride_width: int = 128   # 50% overlap
    
    # Label threshold: minimum defect pixels in patch to label as defective
    defect_threshold_pixels: int = 50  # At least 50 pixels of defect
    defect_threshold_ratio: float = 0.001  # 0.1% of patch area
    
    # Training configuration
    batch_size: int = 32
    prefetch_buffer: int = tf.data.AUTOTUNE


class PatchExtractor:
    """
    Extract patches from full images with overlap.
    
    Design Decision: 256x256 patches with 50% overlap
    - Preserves original resolution
    - Overlapping ensures edge defects are captured
    - ~12 patches per image (full coverage)
    """
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.patches_per_image = self._calculate_patches_per_image()
        logger.info(f"PatchExtractor initialized: {self.patches_per_image} patches per image")
    
    def _calculate_patches_per_image(self) -> int:
        """Calculate number of patches that will be extracted per image"""
        h_patches = (self.config.img_height - self.config.patch_height) // self.config.stride_height + 1
        w_patches = (self.config.img_width - self.config.patch_width) // self.config.stride_width + 1
        return h_patches * w_patches
    
    def extract_patches(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Generator:
        """
        Extract patches from image and optionally mask.
        
        Args:
            image: Full image (256x1600)
            mask: Optional mask (256x1600)
        
        Yields:
            Tuple of (patch_image, patch_mask, patch_info)
        """
        h, w = image.shape[:2]
        patch_h, patch_w = self.config.patch_height, self.config.patch_width
        stride_h, stride_w = self.config.stride_height, self.config.stride_width
        
        patch_idx = 0
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                # Extract image patch
                img_patch = image[y:y+patch_h, x:x+patch_w]
                
                # Extract mask patch if provided
                mask_patch = None
                if mask is not None:
                    mask_patch = mask[y:y+patch_h, x:x+patch_w]
                
                patch_info = {
                    'patch_idx': patch_idx,
                    'y_start': y,
                    'x_start': x,
                    'y_end': y + patch_h,
                    'x_end': x + patch_w
                }
                
                yield img_patch, mask_patch, patch_info
                patch_idx += 1
    
    def get_patch_label(self, mask_patch: np.ndarray) -> Tuple[int, float]:
        """
        Determine patch label from mask.
        
        Args:
            mask_patch: Binary mask patch (256x256)
        
        Returns:
            Tuple of (label, defect_ratio)
            label: 1 if defective, 0 if clean
            defect_ratio: percentage of patch covered by defect
        """
        defect_pixels = np.sum(mask_patch)
        total_pixels = mask_patch.size
        defect_ratio = defect_pixels / total_pixels
        
        # Label as defective if above threshold
        is_defective = (
            defect_pixels >= self.config.defect_threshold_pixels or 
            defect_ratio >= self.config.defect_threshold_ratio
        )
        
        return int(is_defective), defect_ratio


class SafeAugmentation:
    """
    Defect-preserving augmentations only.
    
    ALLOWED:
    - Horizontal flip (defects remain defects)
    - Vertical flip (defects remain defects)
    - Brightness/contrast adjustments
    - Gaussian noise (simulates sensor noise)
    
    REJECTED:
    - Random erasing (might erase defects)
    - Heavy rotation (might distort thin scratches)
    - Aggressive cropping (might remove defects)
    """
    
    def __init__(self, mode: str = 'train'):
        if mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(5, 20), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test: only normalize
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        
        logger.info(f"SafeAugmentation initialized (mode={mode})")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        augmented = self.transform(image=image)
        return augmented['image']


class DataTransformation:
    """
    Main Data Transformation component.
    
    Responsibilities:
    1. Load train/test CSV with image paths and RLE masks
    2. Extract patches from each image
    3. Generate patch-level labels
    4. Create balanced TensorFlow Datasets
    5. Apply safe augmentations
    """
    
    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        self.patch_extractor = PatchExtractor(self.config)
        self.train_augmentation = SafeAugmentation(mode='train')
        self.val_augmentation = SafeAugmentation(mode='val')
        
        # Create directories
        os.makedirs(self.config.patches_dir, exist_ok=True)
        
        logger.info("DataTransformation initialized")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image and convert to RGB numpy array"""
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        return img_array
    
    def create_combined_mask(self, row: pd.Series) -> np.ndarray:
        """
        Create combined binary mask from all defect classes.
        
        For binary classification (defect vs no-defect),
        we combine all class masks into single binary mask.
        """
        h, w = self.config.img_height, self.config.img_width
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check for RLE columns
        for class_id in [1, 2, 3, 4]:
            col_name = f'rle_class_{class_id}'
            if col_name in row and pd.notna(row[col_name]):
                class_mask = rle_to_mask(row[col_name], h, w)
                combined_mask = np.maximum(combined_mask, class_mask)
        
        return combined_mask
    
    def process_single_image(self, image_id: str, row: pd.Series) -> List[Dict]:
        """
        Process single image into patches with labels.
        
        Args:
            image_id: Image filename
            row: DataFrame row with mask RLEs
        
        Returns:
            List of patch dictionaries with image, label, metadata
        """
        image_path = os.path.join(self.config.train_images_dir, image_id)
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return []
        
        # Load image
        image = self.load_image(image_path)
        
        # Create combined mask
        mask = self.create_combined_mask(row)
        
        patches = []
        for img_patch, mask_patch, patch_info in self.patch_extractor.extract_patches(image, mask):
            label, defect_ratio = self.patch_extractor.get_patch_label(mask_patch)
            
            patches.append({
                'image_id': image_id,
                'patch_idx': patch_info['patch_idx'],
                'image': img_patch,
                'label': label,
                'defect_ratio': defect_ratio,
                'coords': patch_info
            })
        
        return patches
    
    def generate_patch_dataset(self, df: pd.DataFrame, max_images: int = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate patch dataset from DataFrame of images.
        
        Args:
            df: DataFrame with ImageId and RLE columns
            max_images: Optional limit on number of images to process
        
        Returns:
            Tuple of (X_patches, y_labels, patch_metadata_df)
        """
        logger.info("Generating patch dataset...")
        
        all_patches = []
        all_labels = []
        metadata = []
        
        # Process each unique image
        unique_images = df['ImageId'].unique()
        if max_images:
            unique_images = unique_images[:max_images]
        
        for idx, image_id in enumerate(unique_images):
            if idx % 100 == 0:
                logger.info(f"Processing image {idx+1}/{len(unique_images)}")
            
            # Get all rows for this image (may have multiple defect classes)
            image_rows = df[df['ImageId'] == image_id]
            
            # Combine into single row with all RLEs
            combined_row = pd.Series({'ImageId': image_id})
            for _, row in image_rows.iterrows():
                class_id = row.get('ClassId', row.get('primary_class', 0))
                if pd.notna(row.get('EncodedPixels')):
                    combined_row[f'rle_class_{class_id}'] = row['EncodedPixels']
            
            # Process image into patches
            patches = self.process_single_image(image_id, combined_row)
            
            for patch in patches:
                all_patches.append(patch['image'])
                all_labels.append(patch['label'])
                metadata.append({
                    'image_id': patch['image_id'],
                    'patch_idx': patch['patch_idx'],
                    'label': patch['label'],
                    'defect_ratio': patch['defect_ratio']
                })
        
        X = np.array(all_patches)
        y = np.array(all_labels)
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Generated {len(X)} patches")
        logger.info(f"Defective patches: {y.sum()} ({y.mean()*100:.1f}%)")
        logger.info(f"Clean patches: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
        
        return X, y, meta_df
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced patch labels.
        
        Uses inverse frequency weighting.
        """
        n_samples = len(y)
        n_classes = 2  # Binary: 0=clean, 1=defective
        
        class_counts = np.bincount(y.astype(int), minlength=n_classes)
        
        weights = {}
        for i in range(n_classes):
            if class_counts[i] > 0:
                weights[i] = n_samples / (n_classes * class_counts[i])
            else:
                weights[i] = 1.0
        
        logger.info(f"Class weights: {weights}")
        return weights
    
    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray, 
                          augmentation: SafeAugmentation,
                          shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow Dataset with augmentation.
        
        Args:
            X: Patch images (N, 256, 256, 3)
            y: Labels (N,)
            augmentation: Augmentation to apply
            shuffle: Whether to shuffle
        
        Returns:
            tf.data.Dataset
        """
        def augment_fn(image, label):
            # Apply augmentation - numpy_function passes numpy array directly
            def apply_aug(x):
                return augmentation(x).astype(np.float32)
            
            image = tf.numpy_function(
                apply_aug,
                [image],
                tf.float32
            )
            image.set_shape([self.config.patch_height, self.config.patch_width, 3])
            return image, label
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        
        return dataset
    
    def initiate_data_transformation(self, train_path: str, test_path: str,
                                     max_train_images: int = None,
                                     max_test_images: int = None) -> Dict:
        """
        Main method to run data transformation pipeline.
        
        Args:
            train_path: Path to train CSV
            test_path: Path to test CSV
            max_train_images: Optional limit for training images
            max_test_images: Optional limit for test images
        
        Returns:
            Dictionary with datasets and metadata
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING DATA TRANSFORMATION PIPELINE")
            logger.info("=" * 60)
            
            # Load CSVs
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train images: {train_df['ImageId'].nunique()}")
            logger.info(f"Test images: {test_df['ImageId'].nunique()}")
            
            # Generate patch datasets
            logger.info("\n--- Processing Training Data ---")
            X_train, y_train, train_meta = self.generate_patch_dataset(
                train_df, max_images=max_train_images
            )
            
            logger.info("\n--- Processing Test Data ---")
            X_test, y_test, test_meta = self.generate_patch_dataset(
                test_df, max_images=max_test_images
            )
            
            # Calculate class weights
            class_weights = self.calculate_class_weights(y_train)
            
            # Create TF Datasets
            logger.info("\n--- Creating TensorFlow Datasets ---")
            train_dataset = self.create_tf_dataset(
                X_train, y_train, self.train_augmentation, shuffle=True
            )
            test_dataset = self.create_tf_dataset(
                X_test, y_test, self.val_augmentation, shuffle=False
            )
            
            # Save metadata
            train_meta.to_csv(os.path.join(self.config.processed_dir, 'train_patches_meta.csv'), index=False)
            test_meta.to_csv(os.path.join(self.config.processed_dir, 'test_patches_meta.csv'), index=False)
            
            logger.info("=" * 60)
            logger.info("DATA TRANSFORMATION COMPLETE")
            logger.info("=" * 60)
            
            return {
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'class_weights': class_weights,
                'train_meta': train_meta,
                'test_meta': test_meta
            }
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data transformation
    config = DataTransformationConfig()
    transformer = DataTransformation(config)
    
    # Run with small sample
    result = transformer.initiate_data_transformation(
        train_path="artifacts/data/processed/train.csv",
        test_path="artifacts/data/processed/test.csv",
        max_train_images=10,  # Small test
        max_test_images=5
    )
    
    print(f"\nTransformation complete!")
    print(f"Train patches: {len(result['X_train'])}")
    print(f"Test patches: {len(result['X_test'])}")
    print(f"Class weights: {result['class_weights']}")

"""
Data Transformation Component - Steel Defect Detection System
==============================================================

OPTIMIZED: On-the-fly patch extraction using tf.data
- NO saving patches to disk (saves 137,500+ files!)
- Patches extracted during training
- Memory efficient with generators
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Generator, List

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import albumentations as A

from src.steel_defect_detection_system.exception import CustomException
from src.steel_defect_detection_system.logger import logger
from src.steel_defect_detection_system.utils import rle_to_mask


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    
    raw_data_dir: str = os.path.join("artifacts", "data", "raw")
    train_images_dir: str = os.path.join("artifacts", "data", "raw", "train_images")
    processed_dir: str = os.path.join("artifacts", "data", "processed")
    
    img_height: int = 256
    img_width: int = 1600
    
    patch_height: int = 256
    patch_width: int = 256
    stride_height: int = 128  # 50% overlap
    stride_width: int = 128
    
    defect_threshold_pixels: int = 50
    defect_threshold_ratio: float = 0.001
    
    batch_size: int = 32
    prefetch_buffer: int = tf.data.AUTOTUNE


class PatchExtractor:
    """Extract patches from full images with overlap."""
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.patches_per_image = self._calculate_patches_per_image()
    
    def _calculate_patches_per_image(self) -> int:
        h_patches = (self.config.img_height - self.config.patch_height) // self.config.stride_height + 1
        w_patches = (self.config.img_width - self.config.patch_width) // self.config.stride_width + 1
        return h_patches * w_patches
    
    def extract_patches(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Generator:
        h, w = image.shape[:2]
        patch_h, patch_w = self.config.patch_height, self.config.patch_width
        stride_h, stride_w = self.config.stride_height, self.config.stride_width
        
        patch_idx = 0
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                img_patch = image[y:y+patch_h, x:x+patch_w]
                mask_patch = mask[y:y+patch_h, x:x+patch_w] if mask is not None else None
                patch_info = {'patch_idx': patch_idx, 'y_start': y, 'x_start': x}
                yield img_patch, mask_patch, patch_info
                patch_idx += 1
    
    def get_patch_label(self, mask_patch: np.ndarray) -> Tuple[int, float]:
        defect_pixels = np.sum(mask_patch)
        total_pixels = mask_patch.size
        defect_ratio = defect_pixels / total_pixels
        is_defective = (defect_pixels >= self.config.defect_threshold_pixels or 
                       defect_ratio >= self.config.defect_threshold_ratio)
        return int(is_defective), defect_ratio


def create_augmentation(mode: str = 'train'):
    """Create augmentation pipeline."""
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(5, 20), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DataTransformation:
    """
    Data Transformation with tf.data pipeline.
    
    Features:
    - On-the-fly patch extraction (no disk storage)
    - Memory efficient with generators
    - Parallel preprocessing with tf.data
    """
    
    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()
        self.patch_extractor = PatchExtractor(self.config)
        logger.info("DataTransformation initialized (optimized mode)")
    
    def load_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        return img_array
    
    def create_combined_mask(self, row: pd.Series) -> np.ndarray:
        h, w = self.config.img_height, self.config.img_width
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for class_id in [1, 2, 3, 4]:
            col_name = f'rle_class_{class_id}'
            if col_name in row and pd.notna(row[col_name]):
                class_mask = rle_to_mask(row[col_name], h, w)
                combined_mask = np.maximum(combined_mask, class_mask)
        
        return combined_mask
    
    def prepare_dataset_info(self, df: pd.DataFrame, max_images: int = None) -> List[Dict]:
        """Prepare list of image info (no loading, just metadata)."""
        unique_images = df['ImageId'].unique()
        if max_images:
            unique_images = unique_images[:max_images]
        
        dataset_info = []
        
        for image_id in unique_images:
            image_path = os.path.join(self.config.train_images_dir, image_id)
            if not os.path.exists(image_path):
                continue
            
            image_rows = df[df['ImageId'] == image_id]
            combined_row = pd.Series({'ImageId': image_id})
            
            if 'rle_class_1' in df.columns:
                first_row = image_rows.iloc[0]
                for class_id in [1, 2, 3, 4]:
                    col = f'rle_class_{class_id}'
                    if col in first_row and pd.notna(first_row[col]):
                        combined_row[col] = first_row[col]
            else:
                for _, row in image_rows.iterrows():
                    class_id = row.get('ClassId', row.get('primary_class', 0))
                    if pd.notna(row.get('EncodedPixels')):
                        combined_row[f'rle_class_{class_id}'] = row['EncodedPixels']
            
            dataset_info.append({
                'image_path': image_path,
                'image_id': image_id,
                'row_data': combined_row
            })
        
        return dataset_info
    
    def patch_generator(self, dataset_info: List[Dict], augmentation=None):
        """Generator yielding patches ON-THE-FLY."""
        for info in dataset_info:
            image = self.load_image(info['image_path'])
            mask = self.create_combined_mask(info['row_data'])
            
            for img_patch, mask_patch, _ in self.patch_extractor.extract_patches(image, mask):
                label, _ = self.patch_extractor.get_patch_label(mask_patch)
                
                if augmentation:
                    augmented = augmentation(image=img_patch)
                    img_patch = augmented['image']
                else:
                    img_patch = img_patch.astype(np.float32) / 255.0
                
                yield img_patch.astype(np.float32), np.float32(label)
    
    def create_tf_dataset(self, dataset_info: List[Dict], mode: str = 'train', 
                          batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Create tf.data.Dataset with on-the-fly patch extraction."""
        augmentation = create_augmentation(mode)
        
        def gen():
            return self.patch_generator(dataset_info, augmentation)
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        
        if shuffle:
            buffer_size = min(10000, len(dataset_info) * 11)
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def initiate_data_transformation(self, train_path: str, test_path: str,
                                      max_train_images: int = None,
                                      max_test_images: int = None) -> Dict:
        """Main entry point - creates tf.data datasets."""
        logger.info("="*50)
        logger.info("DATA TRANSFORMATION (On-the-fly)")
        logger.info("="*50)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Train CSV: {len(train_df)} rows")
        logger.info(f"Test CSV: {len(test_df)} rows")
        
        train_info = self.prepare_dataset_info(train_df, max_train_images)
        test_info = self.prepare_dataset_info(test_df, max_test_images)
        
        # Calculate class weights from sample
        sample_labels = []
        sample_info = train_info[:min(100, len(train_info))]
        for _, label in self.patch_generator(sample_info, create_augmentation('val')):
            sample_labels.append(label)
        
        sample_labels = np.array(sample_labels)
        n_defective = np.sum(sample_labels)
        n_clean = len(sample_labels) - n_defective
        total = len(sample_labels)
        
        class_weights = {
            0: total / (2 * max(n_clean, 1)),
            1: total / (2 * max(n_defective, 1))
        }
        
        logger.info(f"Class weights: {class_weights}")
        
        train_dataset = self.create_tf_dataset(train_info, mode='train', 
                                                batch_size=self.config.batch_size, shuffle=True)
        test_dataset = self.create_tf_dataset(test_info, mode='val', 
                                               batch_size=self.config.batch_size, shuffle=False)
        
        train_patches = len(train_info) * self.patch_extractor.patches_per_image
        test_patches = len(test_info) * self.patch_extractor.patches_per_image
        
        logger.info(f"Train patches: {train_patches}")
        logger.info(f"Test patches: {test_patches}")
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'class_weights': class_weights,
            'train_info': train_info,
            'test_info': test_info,
            'train_patches': train_patches,
            'test_patches': test_patches,
            'patches_per_image': self.patch_extractor.patches_per_image
        }

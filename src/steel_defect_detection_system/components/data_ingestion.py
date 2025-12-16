"""
Data Ingestion Component for Steel Defect Detection System

Handles:
- Kaggle dataset download (Severstal Steel Defect Detection)
- Data parsing and structure analysis
- Stratified train/validation split
- Class distribution analysis
"""
import os
import sys
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.steel_defect_detection_system.exception import CustomException
from src.steel_defect_detection_system.logger import logger


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    # Paths
    raw_data_dir: str = os.path.join("artifacts", "data", "raw")
    train_data_path: str = os.path.join("artifacts", "data", "processed", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "processed", "test.csv")
    
    # Kaggle dataset
    kaggle_dataset: str = "severstal-steel-defect-detection"
    
    # Split configuration
    test_size: float = 0.2
    random_state: int = 42
    
    # Image info (Severstal specific)
    img_height: int = 256
    img_width: int = 1600


class DataIngestion:
    """
    Data Ingestion component for Severstal Steel Defect Dataset
    
    Key responsibilities:
    1. Download dataset from Kaggle
    2. Parse train.csv with RLE masks
    3. Create image-level labels for classification
    4. Stratified split maintaining class balance
    5. Generate class distribution report
    """
    
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
        
        # Create directories
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
        
        logger.info("Data Ingestion initialized")
    
    def download_from_kaggle(self) -> str:
        """
        Download Severstal dataset from Kaggle
        
        Requires: kaggle.json credentials in ~/.kaggle/
        
        Returns:
            Path to extracted data directory
        """
        try:
            logger.info(f"Downloading dataset: {self.config.kaggle_dataset}")
            
            # Import kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Download competition files
            api.competition_download_files(
                competition=self.config.kaggle_dataset,
                path=self.config.raw_data_dir,
                quiet=False
            )
            
            # Extract zip file
            zip_path = os.path.join(self.config.raw_data_dir, f"{self.config.kaggle_dataset}.zip")
            if os.path.exists(zip_path):
                logger.info(f"Extracting {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.config.raw_data_dir)
                os.remove(zip_path)  # Clean up zip
            
            logger.info(f"Dataset downloaded and extracted to {self.config.raw_data_dir}")
            return self.config.raw_data_dir
            
        except Exception as e:
            logger.error(f"Kaggle download failed: {str(e)}")
            raise CustomException(e, sys)
    
    def load_and_parse_annotations(self) -> pd.DataFrame:
        """
        Load train.csv and parse into image-level DataFrame
        
        Original format: ImageId_ClassId, EncodedPixels
        Our format: ImageId, has_defect, defect_types, defect_masks
        
        Returns:
            DataFrame with parsed annotations
        """
        try:
            train_csv_path = os.path.join(self.config.raw_data_dir, "train.csv")
            
            if not os.path.exists(train_csv_path):
                raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
            
            logger.info(f"Loading annotations from {train_csv_path}")
            df = pd.read_csv(train_csv_path)
            
            # CSV already has ImageId and ClassId as separate columns
            # No need to parse from ImageId_ClassId
            
            # Check if image has defect (EncodedPixels is not empty)
            df['has_defect'] = df['EncodedPixels'].notna()
            
            logger.info(f"Loaded {len(df)} annotation rows")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_image_level_labels(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert row-level annotations to image-level labels
        
        For classification task, we need:
        - Binary: has_any_defect (0 or 1)
        - Multi-label: defect_class_1, defect_class_2, defect_class_3, defect_class_4
        - Primary class: most prominent defect for stratification
        
        Returns:
            DataFrame with one row per image
        """
        try:
            logger.info("Creating image-level labels...")
            
            # Get unique images
            unique_images = annotations_df['ImageId'].unique()
            
            image_labels = []
            
            for img_id in unique_images:
                img_rows = annotations_df[annotations_df['ImageId'] == img_id]
                
                # Binary label
                has_any_defect = img_rows['has_defect'].any()
                
                # Multi-label (which classes have defects)
                defect_classes = []
                for class_id in [1, 2, 3, 4]:
                    class_row = img_rows[img_rows['ClassId'] == class_id]
                    if not class_row.empty and class_row['has_defect'].values[0]:
                        defect_classes.append(class_id)
                
                # Primary class (for stratification) - use smallest class_id if multiple
                primary_class = min(defect_classes) if defect_classes else 0
                
                # Store RLE for each class (for segmentation later)
                rle_dict = {}
                for class_id in [1, 2, 3, 4]:
                    class_row = img_rows[img_rows['ClassId'] == class_id]
                    if not class_row.empty:
                        rle_dict[f'rle_class_{class_id}'] = class_row['EncodedPixels'].values[0]
                    else:
                        rle_dict[f'rle_class_{class_id}'] = np.nan
                
                image_labels.append({
                    'ImageId': img_id,
                    'has_defect': int(has_any_defect),
                    'primary_class': primary_class,
                    'defect_classes': ','.join(map(str, defect_classes)) if defect_classes else '',
                    'num_defects': len(defect_classes),
                    **rle_dict
                })
            
            labels_df = pd.DataFrame(image_labels)
            
            logger.info(f"Created labels for {len(labels_df)} images")
            logger.info(f"Defective images: {labels_df['has_defect'].sum()} ({labels_df['has_defect'].mean()*100:.1f}%)")
            
            return labels_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_class_distribution(self, labels_df: pd.DataFrame) -> Dict:
        """
        Analyze and log class distribution
        
        Returns:
            Dictionary with distribution statistics
        """
        try:
            logger.info("=" * 50)
            logger.info("CLASS DISTRIBUTION ANALYSIS")
            logger.info("=" * 50)
            
            # Binary distribution
            has_defect_counts = labels_df['has_defect'].value_counts()
            logger.info(f"\nBinary Distribution:")
            logger.info(f"  No Defect: {has_defect_counts.get(0, 0)}")
            logger.info(f"  Has Defect: {has_defect_counts.get(1, 0)}")
            
            # Per-class distribution
            class_counts = labels_df['primary_class'].value_counts().sort_index()
            logger.info(f"\nPrimary Class Distribution:")
            
            class_names = {0: 'No Defect', 1: 'Pitted', 2: 'Crazing', 3: 'Scratches', 4: 'Patches'}
            
            for class_id, count in class_counts.items():
                pct = count / len(labels_df) * 100
                logger.info(f"  Class {class_id} ({class_names.get(class_id, 'Unknown')}): {count} ({pct:.1f}%)")
            
            # Images with multiple defects
            multi_defect = (labels_df['num_defects'] > 1).sum()
            logger.info(f"\nImages with multiple defect types: {multi_defect}")
            
            logger.info("=" * 50)
            
            return {
                'total_images': len(labels_df),
                'defective_images': int(has_defect_counts.get(1, 0)),
                'class_distribution': class_counts.to_dict(),
                'multi_defect_images': int(multi_defect)
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def stratified_split(self, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split
        
        Stratification is based on primary_class to maintain
        class balance in both splits.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            logger.info(f"Performing stratified split (test_size={self.config.test_size})")
            
            train_df, test_df = train_test_split(
                labels_df,
                test_size=self.config.test_size,
                stratify=labels_df['primary_class'],
                random_state=self.config.random_state
            )
            
            logger.info(f"Train set: {len(train_df)} images")
            logger.info(f"Test set: {len(test_df)} images")
            
            # Verify stratification
            train_dist = train_df['primary_class'].value_counts(normalize=True)
            test_dist = test_df['primary_class'].value_counts(normalize=True)
            
            logger.info("Stratification verification:")
            logger.info(f"Train class distribution: {train_dist.to_dict()}")
            logger.info(f"Test class distribution: {test_dist.to_dict()}")
            
            return train_df, test_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> Tuple[str, str, Dict]:
        """
        Main method to run complete data ingestion pipeline
        
        Steps:
        1. Download from Kaggle (if not exists)
        2. Load and parse annotations
        3. Create image-level labels
        4. Analyze class distribution
        5. Stratified split
        6. Save processed data
        
        Returns:
            Tuple of (train_path, test_path, distribution_stats)
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING DATA INGESTION PIPELINE")
            logger.info("=" * 60)
            
            # Check if data already exists
            train_csv = os.path.join(self.config.raw_data_dir, "train.csv")
            if not os.path.exists(train_csv):
                logger.info("Raw data not found, downloading from Kaggle...")
                self.download_from_kaggle()
            else:
                logger.info(f"Raw data found at {self.config.raw_data_dir}")
            
            # Load annotations
            annotations_df = self.load_and_parse_annotations()
            
            # Create image-level labels
            labels_df = self.create_image_level_labels(annotations_df)
            
            # Analyze distribution
            distribution_stats = self.analyze_class_distribution(labels_df)
            
            # Stratified split
            train_df, test_df = self.stratified_split(labels_df)
            
            # Save processed data
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Train data saved to: {self.config.train_data_path}")
            logger.info(f"Test data saved to: {self.config.test_data_path}")
            
            logger.info("=" * 60)
            logger.info("DATA INGESTION COMPLETE")
            logger.info("=" * 60)
            
            return self.config.train_data_path, self.config.test_data_path, distribution_stats
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data ingestion
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    
    train_path, test_path, stats = ingestion.initiate_data_ingestion()
    print(f"\nData ingestion complete!")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"Stats: {stats}")

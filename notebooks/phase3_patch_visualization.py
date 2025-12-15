"""
Phase 3: Patch Visualization Script
====================================
Visualize patch extraction and labeling to verify preprocessing works correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from steel_defect_detection_system.components.data_transformation import (
    DataTransformationConfig, PatchExtractor, DataTransformation
)
from steel_defect_detection_system.utils import rle_to_mask

# Configuration
RAW_DATA_DIR = project_root / "artifacts" / "data" / "raw"
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
OUTPUT_DIR = project_root / "docs" / "eda_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_patch_extraction(num_images=2):
    """Visualize how images are broken into patches"""
    print("Visualizing patch extraction...")
    
    config = DataTransformationConfig()
    patch_extractor = PatchExtractor(config)
    
    # Load train.csv
    train_csv = RAW_DATA_DIR / "train.csv"
    df = pd.read_csv(train_csv)
    
    # Get sample images with defects
    defect_images = df[df['EncodedPixels'].notna()]['ImageId'].unique()[:num_images]
    
    fig, axes = plt.subplots(num_images, 2, figsize=(20, 5*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx, image_id in enumerate(defect_images):
        # Load image
        img_path = TRAIN_IMAGES_DIR / image_id
        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        
        # Get mask
        img_rows = df[df['ImageId'] == image_id]
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for _, row in img_rows.iterrows():
            if pd.notna(row['EncodedPixels']):
                class_mask = rle_to_mask(row['EncodedPixels'], 256, 1600)
                mask = np.maximum(mask, class_mask)
        
        # Original image with mask
        ax1 = axes[img_idx, 0]
        overlay = img.copy().astype(float) / 255
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask * 0.5, 0, 1)
        ax1.imshow(overlay)
        ax1.set_title(f'Original: {image_id} (256×1600)', fontsize=12)
        ax1.axis('off')
        
        # Show patch boundaries
        for patch_img, patch_mask, patch_info in patch_extractor.extract_patches(img, mask):
            x, y = patch_info['x_start'], patch_info['y_start']
            # Draw rectangle
            ax1.add_patch(plt.Rectangle(
                (x, y), config.patch_width, config.patch_height,
                fill=False, edgecolor='yellow', linewidth=1
            ))
        
        # Show sample patches
        ax2 = axes[img_idx, 1]
        patches = list(patch_extractor.extract_patches(img, mask))
        
        # Create grid of patches
        n_patches = len(patches)
        cols = min(6, n_patches)
        rows = (n_patches + cols - 1) // cols
        
        patch_grid = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
        
        for i, (patch_img, patch_mask, info) in enumerate(patches):
            r, c = i // cols, i % cols
            
            # Add mask overlay
            patch_overlay = patch_img.copy().astype(float) / 255
            patch_overlay[:, :, 0] = np.clip(patch_overlay[:, :, 0] + patch_mask * 0.5, 0, 1)
            
            patch_grid[r*256:(r+1)*256, c*256:(c+1)*256] = (patch_overlay * 255).astype(np.uint8)
        
        ax2.imshow(patch_grid)
        ax2.set_title(f'{n_patches} Patches (256×256 each)', fontsize=12)
        ax2.axis('off')
    
    plt.suptitle('Patch Extraction Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'patch_extraction_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR / 'patch_extraction_demo.png'}")


def visualize_patch_labels(num_images=3):
    """Visualize patch labels (defective vs clean)"""
    print("\nVisualizing patch labels...")
    
    config = DataTransformationConfig()
    transformer = DataTransformation(config)
    
    # Load sample data
    train_csv = RAW_DATA_DIR / "train.csv"
    df = pd.read_csv(train_csv)
    
    # Get images with defects
    defect_images = df[df['EncodedPixels'].notna()]['ImageId'].unique()[:num_images]
    
    fig, axes = plt.subplots(num_images, 1, figsize=(20, 4*num_images))
    if num_images == 1:
        axes = [axes]
    
    for img_idx, image_id in enumerate(defect_images):
        # Get rows for this image
        img_rows = df[df['ImageId'] == image_id]
        
        # Combine into single row
        combined_row = pd.Series({'ImageId': image_id})
        for _, row in img_rows.iterrows():
            class_id = row['ClassId']
            if pd.notna(row['EncodedPixels']):
                combined_row[f'rle_class_{class_id}'] = row['EncodedPixels']
        
        # Process image
        patches = transformer.process_single_image(image_id, combined_row)
        
        if not patches:
            continue
        
        # Create visualization
        n_patches = len(patches)
        patch_viz = np.zeros((256, n_patches * 256, 3), dtype=np.uint8)
        
        for i, patch in enumerate(patches):
            img = patch['image']
            label = patch['label']
            ratio = patch['defect_ratio']
            
            # Add border based on label
            if label == 1:
                # Red border for defective
                img_with_border = np.copy(img)
                img_with_border[:5, :] = [255, 0, 0]
                img_with_border[-5:, :] = [255, 0, 0]
                img_with_border[:, :5] = [255, 0, 0]
                img_with_border[:, -5:] = [255, 0, 0]
            else:
                # Green border for clean
                img_with_border = np.copy(img)
                img_with_border[:5, :] = [0, 255, 0]
                img_with_border[-5:, :] = [0, 255, 0]
                img_with_border[:, :5] = [0, 255, 0]
                img_with_border[:, -5:] = [0, 255, 0]
            
            patch_viz[:, i*256:(i+1)*256] = img_with_border
        
        axes[img_idx].imshow(patch_viz)
        
        # Count labels
        defective = sum(1 for p in patches if p['label'] == 1)
        clean = len(patches) - defective
        
        axes[img_idx].set_title(
            f'{image_id}: {defective} defective (red) | {clean} clean (green)',
            fontsize=11
        )
        axes[img_idx].axis('off')
    
    plt.suptitle('Patch Labels: Red=Defective, Green=Clean', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'patch_labels_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR / 'patch_labels_demo.png'}")


def main():
    print("="*60)
    print("PHASE 3: PATCH VISUALIZATION")
    print("="*60)
    
    visualize_patch_extraction(num_images=2)
    visualize_patch_labels(num_images=3)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

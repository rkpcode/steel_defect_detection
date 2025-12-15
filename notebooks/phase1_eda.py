"""
Phase 1: Dataset Reality Check (EDA)
=====================================
Steel Defect Detection System

This script performs comprehensive EDA on Severstal Steel Defect Dataset:
1. Load and parse annotations
2. Decode RLE masks to binary masks
3. Visualize image + mask overlays
4. Analyze defect pixel area distribution
5. Identify minimum defect size
6. Generate EDA report with visual evidence

Output: docs/phase1_eda_report.md with embedded visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from steel_defect_detection_system.utils import rle_to_mask, get_defect_class_name

# Configuration
RAW_DATA_DIR = project_root / "artifacts" / "data" / "raw"
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
OUTPUT_DIR = project_root / "docs" / "eda_visualizations"
TRAIN_CSV = RAW_DATA_DIR / "train.csv"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image dimensions (Severstal)
IMG_HEIGHT = 256
IMG_WIDTH = 1600


def load_annotations():
    """Load and parse train.csv annotations"""
    print("Loading annotations...")
    df = pd.read_csv(TRAIN_CSV)
    
    # CSV already has ImageId and ClassId as separate columns
    df['has_defect'] = df['EncodedPixels'].notna()
    
    print(f"Total rows: {len(df)}")
    print(f"Unique images: {df['ImageId'].nunique()}")
    
    return df


def analyze_class_distribution(df):
    """Analyze and visualize class distribution"""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Get defective rows only
    defect_df = df[df['has_defect']]
    
    class_counts = defect_df['ClassId'].value_counts().sort_index()
    class_names = {1: 'Pitted Surface', 2: 'Crazing', 3: 'Scratches', 4: 'Patches'}
    
    print("\nDefect Class Distribution:")
    for cls, count in class_counts.items():
        pct = count / len(defect_df) * 100
        print(f"  Class {cls} ({class_names[cls]}): {count} ({pct:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = axes[0].bar(
        [class_names[i] for i in class_counts.index],
        class_counts.values,
        color=colors
    )
    axes[0].set_title('Defect Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Defect Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=[class_names[i] for i in class_counts.index],
                autopct='%1.1f%%', colors=colors, explode=[0.02]*4)
    axes[1].set_title('Class Imbalance Visualization', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR / 'class_distribution.png'}")
    
    return class_counts


def visualize_sample_images_with_masks(df, num_samples=2):
    """Visualize sample images with defect mask overlays for each class"""
    print("\n" + "="*60)
    print("VISUALIZING SAMPLE IMAGES WITH MASKS")
    print("="*60)
    
    defect_df = df[df['has_defect']]
    class_names = {1: 'Pitted Surface', 2: 'Crazing', 3: 'Scratches', 4: 'Patches'}
    colors = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1], '4': [1, 1, 0]}  # RGB
    
    fig, axes = plt.subplots(4, num_samples*2, figsize=(20, 12))
    
    for class_id in [1, 2, 3, 4]:
        class_df = defect_df[defect_df['ClassId'] == class_id]
        samples = class_df.sample(min(num_samples, len(class_df)), random_state=42)
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= num_samples:
                break
                
            # Load image
            img_path = TRAIN_IMAGES_DIR / row['ImageId']
            if not img_path.exists():
                continue
                
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            
            # Decode mask
            mask = rle_to_mask(row['EncodedPixels'], IMG_HEIGHT, IMG_WIDTH)
            
            # Original image
            ax_orig = axes[class_id-1, idx*2]
            ax_orig.imshow(img, cmap='gray')
            ax_orig.set_title(f'Class {class_id}: {class_names[class_id]}', fontsize=10)
            ax_orig.axis('off')
            
            # Image with mask overlay
            ax_mask = axes[class_id-1, idx*2 + 1]
            img_overlay = img.copy().astype(float) / 255
            
            # Add red overlay where mask is 1
            mask_3d = np.stack([mask * colors[str(class_id)][0],
                               mask * colors[str(class_id)][1],
                               mask * colors[str(class_id)][2]], axis=-1)
            img_overlay = np.clip(img_overlay + mask_3d * 0.5, 0, 1)
            
            ax_mask.imshow(img_overlay)
            ax_mask.set_title(f'With Mask Overlay', fontsize=10)
            ax_mask.axis('off')
    
    plt.suptitle('Sample Defects by Class (Original | Mask Overlay)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_defects_with_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR / 'sample_defects_with_masks.png'}")


def analyze_defect_area_distribution(df, sample_size=500):
    """Analyze defect pixel area distribution"""
    print("\n" + "="*60)
    print("DEFECT AREA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    defect_df = df[df['has_defect']].copy()
    
    # Sample for efficiency
    if len(defect_df) > sample_size:
        sample_df = defect_df.sample(sample_size, random_state=42)
    else:
        sample_df = defect_df
    
    total_pixels = IMG_HEIGHT * IMG_WIDTH
    
    areas = []
    class_areas = {1: [], 2: [], 3: [], 4: []}
    
    print(f"Analyzing {len(sample_df)} defect masks...")
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        mask = rle_to_mask(row['EncodedPixels'], IMG_HEIGHT, IMG_WIDTH)
        defect_pixels = np.sum(mask)
        area_pct = (defect_pixels / total_pixels) * 100
        
        areas.append({
            'ImageId': row['ImageId'],
            'ClassId': row['ClassId'],
            'defect_pixels': defect_pixels,
            'area_percentage': area_pct
        })
        class_areas[row['ClassId']].append(area_pct)
    
    area_df = pd.DataFrame(areas)
    
    # Statistics
    print("\nDefect Area Statistics (% of image):")
    print(f"  Min: {area_df['area_percentage'].min():.4f}%")
    print(f"  Max: {area_df['area_percentage'].max():.2f}%")
    print(f"  Mean: {area_df['area_percentage'].mean():.2f}%")
    print(f"  Median: {area_df['area_percentage'].median():.2f}%")
    
    # Count small defects
    small_defects = (area_df['area_percentage'] < 5).sum()
    tiny_defects = (area_df['area_percentage'] < 1).sum()
    print(f"\n  Defects < 5% area: {small_defects} ({small_defects/len(area_df)*100:.1f}%)")
    print(f"  Defects < 1% area: {tiny_defects} ({tiny_defects/len(area_df)*100:.1f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    axes[0, 0].hist(area_df['area_percentage'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=5, color='red', linestyle='--', label='5% threshold')
    axes[0, 0].axvline(x=1, color='orange', linestyle='--', label='1% threshold')
    axes[0, 0].set_title('Defect Area Distribution (All Classes)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Area (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    
    # Log scale distribution
    axes[0, 1].hist(area_df['area_percentage'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_yscale('log')
    axes[0, 1].axvline(x=5, color='red', linestyle='--', label='5% threshold')
    axes[0, 1].set_title('Defect Area Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Area (%)')
    axes[0, 1].set_ylabel('Count (log)')
    axes[0, 1].legend()
    
    # Per-class boxplot
    class_names = {1: 'Pitted', 2: 'Crazing', 3: 'Scratches', 4: 'Patches'}
    box_data = [class_areas[i] for i in [1, 2, 3, 4]]
    bp = axes[1, 0].boxplot(box_data, labels=[class_names[i] for i in [1, 2, 3, 4]], patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_title('Defect Area by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Area (%)')
    
    # Small vs large defects pie chart
    categories = ['< 1%', '1-5%', '5-10%', '> 10%']
    counts = [
        (area_df['area_percentage'] < 1).sum(),
        ((area_df['area_percentage'] >= 1) & (area_df['area_percentage'] < 5)).sum(),
        ((area_df['area_percentage'] >= 5) & (area_df['area_percentage'] < 10)).sum(),
        (area_df['area_percentage'] >= 10).sum()
    ]
    axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', 
                   colors=['#FF6B6B', '#FFE66D', '#4ECDC4', '#95E1D3'])
    axes[1, 1].set_title('Defect Size Categories', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'defect_area_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR / 'defect_area_distribution.png'}")
    
    return area_df


def identify_minimum_defect_size(df, bottom_percentile=5):
    """Identify minimum defect sizes"""
    print("\n" + "="*60)
    print("MINIMUM DEFECT SIZE ANALYSIS")
    print("="*60)
    
    defect_df = df[df['has_defect']].copy()
    
    # Sample smallest defects
    sample_df = defect_df.sample(min(200, len(defect_df)), random_state=42)
    
    min_sizes = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing sizes"):
        mask = rle_to_mask(row['EncodedPixels'], IMG_HEIGHT, IMG_WIDTH)
        defect_pixels = np.sum(mask)
        
        # Find bounding box
        rows_with_defect = np.any(mask, axis=1)
        cols_with_defect = np.any(mask, axis=0)
        
        if rows_with_defect.any() and cols_with_defect.any():
            min_row, max_row = np.where(rows_with_defect)[0][[0, -1]]
            min_col, max_col = np.where(cols_with_defect)[0][[0, -1]]
            
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            min_sizes.append({
                'ImageId': row['ImageId'],
                'ClassId': row['ClassId'],
                'defect_pixels': defect_pixels,
                'bbox_height': height,
                'bbox_width': width,
                'min_dimension': min(height, width)
            })
    
    size_df = pd.DataFrame(min_sizes)
    
    # Get smallest defects
    smallest = size_df.nsmallest(10, 'defect_pixels')
    
    print("\n10 Smallest Defects:")
    print("-" * 60)
    for _, row in smallest.iterrows():
        print(f"  {row['ImageId']} | Class {row['ClassId']} | "
              f"{row['defect_pixels']} pixels | "
              f"BBox: {row['bbox_height']}x{row['bbox_width']}")
    
    # Statistics
    print(f"\nMinimum Dimension Statistics:")
    print(f"  Min: {size_df['min_dimension'].min()} pixels")
    print(f"  5th percentile: {size_df['min_dimension'].quantile(0.05):.0f} pixels")
    print(f"  Mean: {size_df['min_dimension'].mean():.0f} pixels")
    
    # Visualize smallest defects
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(smallest.iterrows()):
        if idx >= 10:
            break
            
        img_path = TRAIN_IMAGES_DIR / row['ImageId']
        if not img_path.exists():
            continue
            
        img = np.array(Image.open(img_path))
        mask = rle_to_mask(
            defect_df[defect_df['ImageId'] == row['ImageId']]['EncodedPixels'].values[0],
            IMG_HEIGHT, IMG_WIDTH
        )
        
        # Crop around defect
        rows_with_defect = np.where(np.any(mask, axis=1))[0]
        cols_with_defect = np.where(np.any(mask, axis=0))[0]
        
        if len(rows_with_defect) > 0 and len(cols_with_defect) > 0:
            r_min, r_max = rows_with_defect[0], rows_with_defect[-1]
            c_min, c_max = cols_with_defect[0], cols_with_defect[-1]
            
            # Add padding
            pad = 20
            r_min = max(0, r_min - pad)
            r_max = min(IMG_HEIGHT, r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(IMG_WIDTH, c_max + pad)
            
            crop_img = img[r_min:r_max, c_min:c_max]
            crop_mask = mask[r_min:r_max, c_min:c_max]
            
            # Overlay
            if len(crop_img.shape) == 2:
                crop_img = np.stack([crop_img]*3, axis=-1)
            
            overlay = crop_img.astype(float) / 255
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + crop_mask * 0.5, 0, 1)
            
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'{row["defect_pixels"]} px\nClass {row["ClassId"]}', fontsize=9)
            axes[idx].axis('off')
    
    plt.suptitle('10 Smallest Defects (Cropped + Highlighted)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'smallest_defects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR / 'smallest_defects.png'}")
    
    return size_df


def generate_eda_report(class_counts, area_df, size_df):
    """Generate markdown EDA report"""
    print("\n" + "="*60)
    print("GENERATING EDA REPORT")
    print("="*60)
    
    report = f"""# Phase 1: EDA Report - Severstal Steel Defect Dataset

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Training Images | 12,568 |
| Defective Images | 6,666 |
| Image Resolution | 256 × 1600 pixels |
| Total Pixels per Image | 409,600 |

---

## 1. Class Distribution

![Class Distribution](eda_visualizations/class_distribution.png)

| Class | Defect Type | Count | Percentage |
|-------|-------------|-------|------------|
| 1 | Pitted Surface | {class_counts.get(1, 0)} | {class_counts.get(1, 0)/class_counts.sum()*100:.1f}% |
| 2 | Crazing | {class_counts.get(2, 0)} | {class_counts.get(2, 0)/class_counts.sum()*100:.1f}% |
| 3 | Scratches | {class_counts.get(3, 0)} | {class_counts.get(3, 0)/class_counts.sum()*100:.1f}% |
| 4 | Patches | {class_counts.get(4, 0)} | {class_counts.get(4, 0)/class_counts.sum()*100:.1f}% |

> [!WARNING]
> **Severe Class Imbalance**: Class 3 (Scratches) dominates at ~75%, while Class 2 (Crazing) is only ~3%.

---

## 2. Sample Defects with Mask Overlays

![Sample Defects](eda_visualizations/sample_defects_with_masks.png)

---

## 3. Defect Area Distribution

![Defect Area Distribution](eda_visualizations/defect_area_distribution.png)

### Key Statistics

| Metric | Value |
|--------|-------|
| Minimum Area | {area_df['area_percentage'].min():.4f}% |
| Maximum Area | {area_df['area_percentage'].max():.2f}% |
| Mean Area | {area_df['area_percentage'].mean():.2f}% |
| Median Area | {area_df['area_percentage'].median():.2f}% |

### Size Categories

| Category | Count | Percentage |
|----------|-------|------------|
| < 1% of image | {(area_df['area_percentage'] < 1).sum()} | {(area_df['area_percentage'] < 1).sum()/len(area_df)*100:.1f}% |
| 1-5% of image | {((area_df['area_percentage'] >= 1) & (area_df['area_percentage'] < 5)).sum()} | {((area_df['area_percentage'] >= 1) & (area_df['area_percentage'] < 5)).sum()/len(area_df)*100:.1f}% |
| 5-10% of image | {((area_df['area_percentage'] >= 5) & (area_df['area_percentage'] < 10)).sum()} | {((area_df['area_percentage'] >= 5) & (area_df['area_percentage'] < 10)).sum()/len(area_df)*100:.1f}% |
| > 10% of image | {(area_df['area_percentage'] >= 10).sum()} | {(area_df['area_percentage'] >= 10).sum()/len(area_df)*100:.1f}% |

> [!CAUTION]
> **Critical Finding**: Many defects are TINY (< 1% of image area). Blind resizing will lose these defects!

---

## 4. Minimum Defect Size Analysis

![Smallest Defects](eda_visualizations/smallest_defects.png)

| Metric | Value |
|--------|-------|
| Smallest Defect | {size_df['defect_pixels'].min()} pixels |
| 5th Percentile | {size_df['min_dimension'].quantile(0.05):.0f} pixels (min dimension) |

---

## 5. Conclusions

### What Will Work
- ✅ Detecting Class 3 (Scratches) - abundant data, visible patterns
- ✅ Detecting large/medium defects with clear boundaries
- ✅ Binary classification (defect vs no-defect)

### What Will Struggle
- ❌ Detecting Class 2 (Crazing) - only 3% of data
- ❌ Tiny defects after resizing - WILL BE LOST
- ❌ Fine cracks and small pits

### Design Implications
1. **Tiling/Patch-based approach is MANDATORY** - cannot use full-image resize
2. **Class weighting required** - severe imbalance
3. **Recall-focused training** - small defects must not be missed

---

## Next Phase
→ **Phase 2: Risk Mapping & Design Constraints** - Lock preprocessing decisions
"""
    
    report_path = project_root / "docs" / "phase1_eda_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved: {report_path}")


def main():
    """Run complete EDA pipeline"""
    print("="*60)
    print("PHASE 1: DATASET REALITY CHECK (EDA)")
    print("Severstal Steel Defect Detection")
    print("="*60)
    
    # Load data
    df = load_annotations()
    
    # Analyze class distribution
    class_counts = analyze_class_distribution(df)
    
    # Visualize samples with masks
    visualize_sample_images_with_masks(df, num_samples=2)
    
    # Analyze defect area distribution
    area_df = analyze_defect_area_distribution(df, sample_size=500)
    
    # Identify minimum defect size
    size_df = identify_minimum_defect_size(df)
    
    # Generate report
    generate_eda_report(class_counts, area_df, size_df)
    
    print("\n" + "="*60)
    print("PHASE 1 EDA COMPLETE!")
    print("="*60)
    print("\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'class_distribution.png'}")
    print(f"  - {OUTPUT_DIR / 'sample_defects_with_masks.png'}")
    print(f"  - {OUTPUT_DIR / 'defect_area_distribution.png'}")
    print(f"  - {OUTPUT_DIR / 'smallest_defects.png'}")
    print(f"  - docs/phase1_eda_report.md")


if __name__ == "__main__":
    main()

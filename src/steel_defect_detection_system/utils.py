"""
Utility Functions for Steel Defect Detection System
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from steel_defect_detection_system.logger import logger


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

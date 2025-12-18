"""
OPTIMIZED COLAB PIPELINE - Steel Defect Detection
==================================================

Optimizations:
1. On-the-fly patch extraction (NO disk storage)
2. tf.data pipeline for memory efficiency
3. Zip deleted immediately after extraction
4. Only best model saved (save_best_only=True)
"""

# ============================================
# CELL 1: SETUP
# ============================================
"""
# Run in Colab:
!git clone https://github.com/rkpcode/steel_defect_detection.git
%cd /content/steel_defect_detection
!pip install -q albumentations gdown kaggle
"""

# ============================================
# CELL 2: KAGGLE DATA DOWNLOAD
# ============================================
"""
# Upload kaggle.json, then run:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c severstal-steel-defect-detection -p artifacts/data/raw
!unzip -q artifacts/data/raw/severstal-steel-defect-detection.zip -d artifacts/data/raw
!rm artifacts/data/raw/severstal-steel-defect-detection.zip  # DELETE ZIP IMMEDIATELY
"""

# ============================================
# CELL 3: DATA PREPARATION
# ============================================
import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
import albumentations as A
import gc
from PIL import Image

print("="*60)
print("OPTIMIZED PIPELINE - ON-THE-FLY PATCH EXTRACTION")
print("="*60)

# Configuration
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap
IMG_HEIGHT = 256
IMG_WIDTH = 1600
BATCH_SIZE = 32
EPOCHS = 15

# RLE to mask function
def rle_to_mask(rle_string, height, width):
    """Convert RLE to binary mask."""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((width, height)).T  # Note: Kaggle format is column-first

# Load and parse train.csv
print("\nLoading train.csv...")
train_csv_path = 'artifacts/data/raw/train.csv'
train_images_dir = 'artifacts/data/raw/train_images'

df = pd.read_csv(train_csv_path)
print(f"Total rows: {len(df)}")

# Create image-level DataFrame with RLE data
print("Preparing image data...")
images_data = []
for image_id in df['ImageId'].unique():
    image_rows = df[df['ImageId'] == image_id]
    
    has_defect = False
    rle_data = {}
    for _, row in image_rows.iterrows():
        if pd.notna(row.get('EncodedPixels')):
            has_defect = True
            class_id = row['ClassId']
            rle_data[f'rle_class_{class_id}'] = row['EncodedPixels']
    
    images_data.append({
        'ImageId': image_id,
        'has_defect': has_defect,
        **rle_data
    })

images_df = pd.DataFrame(images_data)
print(f"Unique images: {len(images_df)}")
print(f"Defective images: {images_df['has_defect'].sum()} ({images_df['has_defect'].mean()*100:.1f}%)")

# Train/Test split (80/20)
from sklearn.model_selection import train_test_split
train_images, test_images = train_test_split(
    images_df, test_size=0.2, stratify=images_df['has_defect'], random_state=42
)
print(f"\nTrain images: {len(train_images)}")
print(f"Test images: {len(test_images)}")


# ============================================
# CELL 4: ON-THE-FLY DATA PIPELINE
# ============================================
print("\n" + "="*60)
print("CREATING ON-THE-FLY DATA PIPELINE")
print("="*60)

# Augmentation
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_aug = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    """Load image as numpy array."""
    img = Image.open(image_path)
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    return img_array

def create_combined_mask(row):
    """Create combined mask from all classes."""
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for class_id in [1, 2, 3, 4]:
        col = f'rle_class_{class_id}'
        if col in row and pd.notna(row[col]):
            class_mask = rle_to_mask(row[col], IMG_HEIGHT, IMG_WIDTH)
            mask = np.maximum(mask, class_mask)
    return mask

def extract_patches_from_image(image, mask):
    """Extract patches on-the-fly."""
    patches = []
    labels = []
    
    for y in range(0, IMG_HEIGHT - PATCH_SIZE + 1, STRIDE):
        for x in range(0, IMG_WIDTH - PATCH_SIZE + 1, STRIDE):
            img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Label: 1 if >=50 defect pixels
            label = 1 if np.sum(mask_patch) >= 50 else 0
            
            patches.append(img_patch)
            labels.append(label)
    
    return patches, labels

def data_generator(image_df, augmentation, shuffle=True):
    """Generator for on-the-fly patch extraction."""
    indices = image_df.index.tolist()
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            row = image_df.loc[idx]
            image_path = os.path.join(train_images_dir, row['ImageId'])
            
            if not os.path.exists(image_path):
                continue
            
            # Load image and mask
            image = load_image(image_path)
            mask = create_combined_mask(row)
            
            # Extract patches on-the-fly
            patches, labels = extract_patches_from_image(image, mask)
            
            # Yield each patch
            for patch, label in zip(patches, labels):
                aug_patch = augmentation(image=patch)['image']
                yield aug_patch.astype(np.float32), np.float32(label)

# Calculate patches per epoch
patches_per_image = ((IMG_HEIGHT - PATCH_SIZE) // STRIDE + 1) * ((IMG_WIDTH - PATCH_SIZE) // STRIDE + 1)
train_patches_per_epoch = len(train_images) * patches_per_image
test_patches = len(test_images) * patches_per_image

print(f"Patches per image: {patches_per_image}")
print(f"Train patches (estimated): {train_patches_per_epoch}")
print(f"Test patches (estimated): {test_patches}")

# Create tf.data datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_aug, shuffle=True),
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_images, val_aug, shuffle=False),
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Datasets created!")


# ============================================
# CELL 5: MODEL TRAINING
# ============================================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Custom metric
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return (5 * precision * recall) / (4 * precision + recall + tf.keras.backend.epsilon())

# Create model
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             f2_score]
)

model.summary()

# Class weights
class_weight = {0: 0.85, 1: 1.22}  # Approximate

# Callbacks - ONLY save best model!
os.makedirs('artifacts/models', exist_ok=True)
callbacks_list = [
    callbacks.ModelCheckpoint(
        'artifacts/models/best_model.keras',
        monitor='val_recall', mode='max',
        save_best_only=True,  # ONLY BEST!
        verbose=1
    ),
    callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=5, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_recall', mode='max', factor=0.5, patience=2, verbose=1)
]

# Calculate steps
steps_per_epoch = train_patches_per_epoch // BATCH_SIZE
validation_steps = test_patches // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Train
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_dataset,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks_list,
    verbose=1
)

print("\nTraining complete!")
print(f"Best model saved to: artifacts/models/best_model.keras")


# ============================================
# CELL 6: DOWNLOAD RESULTS
# ============================================
"""
# Download model
from google.colab import files
files.download('artifacts/models/best_model.keras')
"""

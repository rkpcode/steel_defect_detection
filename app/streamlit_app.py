"""
Steel Defect Detection System - Streamlit Web App
===================================================

A modern web interface for detecting defects in steel images.
Upload an image and get instant PASS/FAIL/HOLD predictions.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .result-pass {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .result-fail {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    .result-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Upload area */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1);
        border: 2px dashed #667eea;
        border-radius: 15px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Custom Metrics (must match training)
# ============================================
def f2_score(y_true, y_pred):
    """F2-Score: Emphasizes recall over precision"""
    import tensorflow as tf
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    beta = 2.0
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + tf.keras.backend.epsilon())
    return f2


# ============================================
# Model Loading (Cached) with Google Drive Support
# ============================================

# Google Drive Model URL - Replace with your shared link
# Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# Extract FILE_ID and use: https://drive.google.com/uc?id=FILE_ID
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1BgYItrZL2TYAbJAQiKDV_LlNMIoKxBfI"  # Set this to your Google Drive direct download URL

def download_model_from_gdrive(url, output_path):
    """Download model from Google Drive"""
    try:
        import gdown
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False

@st.cache_resource
def load_model():
    """Load the trained model (cached) - supports local and Google Drive"""
    model_paths = [
        "artifacts/models/transfer_model_best.keras",  # New transfer learning model (99.55% recall)
        "artifacts/models/transfer_model_final.keras",
        "artifacts/models/baseline_model_final.keras",
        "artifacts/models/baseline_model_best.keras"
    ]
    
    custom_objects = {'f2_score': f2_score}
    
    # Try local paths first
    for path in model_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            return model, path
    
    # If no local model, try Google Drive download
    if GDRIVE_MODEL_URL:
        download_path = "artifacts/models/transfer_model_best.keras"
        st.info("Downloading model from Google Drive... (first time only)")
        if download_model_from_gdrive(GDRIVE_MODEL_URL, download_path):
            model = tf.keras.models.load_model(download_path, custom_objects=custom_objects)
            return model, download_path
    
    return None, None


# ============================================
# Prediction Functions
# ============================================
def extract_patches(image: np.ndarray, patch_size: int = 256, stride: int = 128):
    """Extract patches from image"""
    h, w = image.shape[:2]
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    
    return patches, coords


def predict_image(model, image: np.ndarray, threshold: float = 0.37):
    """Run prediction on image"""
    # Normalize
    img_normalized = image.astype(np.float32) / 255.0
    
    # Extract patches
    patches, coords = extract_patches(img_normalized)
    
    if not patches:
        return None
    
    # Predict
    batch = np.array(patches)
    predictions = model.predict(batch, verbose=0).flatten()
    
    # Aggregate
    max_prob = float(np.max(predictions))
    mean_prob = float(np.mean(predictions))
    n_defective = int(np.sum(predictions >= threshold))
    
    # Decision based on threshold
    if max_prob < threshold:
        decision = "PASS"
        decision_class = "result-pass"
    else:
        decision = "FAIL"
        decision_class = "result-fail"
    
    return {
        'decision': decision,
        'decision_class': decision_class,
        'confidence': max_prob,
        'mean_prob': mean_prob,
        'n_defective': n_defective,
        'total_patches': len(patches),
        'patch_probs': predictions,
        'coords': coords
    }


# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Steel Defect Detection System</h1>
        <p>AI-powered quality inspection for steel manufacturing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.20,
            max_value=0.70,
            value=0.37,
            step=0.01,
            help="Optimal Threshold: 0.37 (99.55% Recall, 47.36% Precision, F2=0.816). Lower = more sensitive"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Decision Logic")
        st.markdown(f"""
        | Confidence | Decision |
        |------------|----------|
        | < {threshold:.2f} | ✅ **PASS** |
        | ≥ {threshold:.2f} | ❌ **FAIL** |
        
        *Optimal: 0.37 (99.55% Recall, F2=0.816)*
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Info")
        model, model_path = load_model()
        if model:
            st.success(f"Model loaded!")
            st.caption(f"Path: {model_path}")
        else:
            st.error("No model found!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a steel image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a steel surface image for defect detection"
        )
        
        # Demo images
        st.markdown("---")
        st.markdown("### 🖼️ Or Try Demo Images")
        
        demo_dir = "artifacts/data/raw/train_images"
        if os.path.exists(demo_dir):
            import glob
            demo_images = glob.glob(os.path.join(demo_dir, "*.jpg"))[:6]
            
            if demo_images:
                demo_cols = st.columns(3)
                for i, demo_path in enumerate(demo_images[:3]):
                    with demo_cols[i]:
                        if st.button(f"Demo {i+1}", key=f"demo_{i}"):
                            st.session_state['demo_image'] = demo_path
    
    with col2:
        st.markdown("### 🔍 Prediction Result")
        
        # Get image to process
        image_to_process = None
        
        if uploaded_file is not None:
            # Read file bytes first
            bytes_data = uploaded_file.read()
            from io import BytesIO
            image = Image.open(BytesIO(bytes_data))
            image_to_process = np.array(image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        elif 'demo_image' in st.session_state:
            image = Image.open(st.session_state['demo_image'])
            image_to_process = np.array(image)
            st.image(image, caption="Demo Image", use_container_width=True)
        
        # Run prediction
        if image_to_process is not None and model is not None:
            with st.spinner("Analyzing image..."):
                # Convert grayscale to RGB
                if len(image_to_process.shape) == 2:
                    image_to_process = np.stack([image_to_process] * 3, axis=-1)
                
                result = predict_image(model, image_to_process, threshold)
            
            if result:
                # Decision card
                decision_emoji = {"PASS": "✅", "FAIL": "❌", "HOLD": "⚠️"}
                st.markdown(f"""
                <div class="result-card {result['decision_class']}">
                    <h1>{decision_emoji[result['decision']]} {result['decision']}</h1>
                    <p style="font-size: 1.5rem;">Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Max Probability", f"{result['confidence']:.1%}")
                with metric_cols[1]:
                    st.metric("Mean Probability", f"{result['mean_prob']:.1%}")
                with metric_cols[2]:
                    st.metric("Defective Patches", f"{result['n_defective']}/{result['total_patches']}")
                
                # Patch analysis
                with st.expander("📊 Detailed Patch Analysis"):
                    st.markdown("**Patch-level Probabilities:**")
                    
                    # Create heatmap-style visualization
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.bar(range(len(result['patch_probs'])), result['patch_probs'], 
                           color=['red' if p > threshold else 'green' for p in result['patch_probs']])
                    ax.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
                    ax.set_xlabel('Patch Index')
                    ax.set_ylabel('Defect Probability')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
        
        elif model is None:
            st.warning("⚠️ No model loaded. Please train a model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); padding: 1rem;">
        <p>Steel Defect Detection System | Built with ❤️ using Deep Learning</p>
        <p>EfficientNetB0 Transfer Learning | 99.55% Recall | Threshold: 0.37</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

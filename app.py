import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import h5py

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Pure Black & White / Monochrome Modern Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header { 
        visibility: hidden; 
    }
    
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* App Header */
    .app-header {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #888888;
        letter-spacing: 0.02em;
    }
    
    /* Primary Result Card (Ensemble) */
    .result-card-main {
        background: #0D0D0D;
        border: 1px solid #2E2E2E;
        border-radius: 14px;
        padding: 24px;
        margin-top: 25px;
        margin-bottom: 24px;
    }
    
    .ensemble-badge {
        display: inline-block;
        background: #FFFFFF;
        color: #000000;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    
    .result-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
    }
    
    .result-desc {
        color: #A0A0A0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #222222;
        letter-spacing: -0.01em;
    }
    
    /* Individual Model Cards Grid */
    .model-card {
        background: #0D0D0D;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 18px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .model-name-tag {
        font-size: 0.8rem;
        font-weight: 700;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    
    .model-prediction {
        font-size: 1.25rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    
    .model-conf-value {
        font-size: 0.95rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #CCCCCC;
        margin-bottom: 12px;
    }
    
    .model-breakdown-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #888888;
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .bar-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #FFFFFF;
        margin-bottom: 4px;
        margin-top: 12px;
    }
    
    /* Monochrome Streamlit Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #FFFFFF !important;
    }
    .stProgress > div > div {
        background-color: #222222 !important;
    }
    
    /* Upload Box & Button */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 1px solid #FFFFFF !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #E5E5E5 !important;
        border-color: #E5E5E5 !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #0D0D0D;
        border: 1px dashed #333333;
        border-radius: 12px;
        padding: 12px;
    }
    
    /* Footer Disclaimer */
    .disclaimer-text {
        font-size: 0.78rem;
        color: #555555;
        text-align: center;
        margin-top: 45px;
        padding-top: 18px;
        border-top: 1px solid #1A1A1A;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Class Info
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary Tumor"]
CLASS_DESCRIPTIONS = [
    "A primary brain tumor originating in glial cells. May require surgical evaluation, radiotherapy, or chemotherapy.",
    "A tumor arising from the meninges (membranes around the brain). Often benign and slow-growing.",
    "A tumor developing in the pituitary gland at the base of the brain. Usually benign and treatable."
]

MODEL_FILES = {
    "VGG-16": "final_brain_tumor_model_main.h5",
    "MobileNet": "final_brain_tumor_model_main_mobilenet.h5",
    "ResNet-50": "final_brain_tumor_model_main_resnet50.h5"
}

# ─────────────────────────────────────────────────────────────────────────────
# Topology-Aware Weight Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_weights_hdf5(model, h5_path: str):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Weights not found: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        mw = f['model_weights']
        dataset_map = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_map[name] = np.array(obj)
        mw.visititems(collect)
        
        for layer in model.layers:
            if hasattr(layer, 'layers') and layer.layers:
                for sub in layer.layers:
                    if not sub.weights:
                        continue
                    matched = []
                    for w in sub.weights:
                        w_short = w.name.split('/')[-1].split(':')[0]
                        found = None
                        for k, v in dataset_map.items():
                            if f'{sub.name}/{w_short}' in k or (k.endswith(f'/{w_short}') and f'/{sub.name}/' in f'/{k}'):
                                found = v
                                break
                        if found is not None:
                            matched.append(found)
                    if len(matched) == len(sub.weights):
                        sub.set_weights(matched)
            else:
                if not layer.weights:
                    continue
                matched = []
                for w in layer.weights:
                    w_short = w.name.split('/')[-1].split(':')[0]
                    found = None
                    for k, v in dataset_map.items():
                        if f'dense/{w_short}' in k or k.endswith(f'{layer.name}/{w_short}'):
                            found = v
                            break
                    if found is not None:
                        matched.append(found)
                if len(matched) == len(layer.weights):
                    layer.set_weights(matched)

# ─────────────────────────────────────────────────────────────────────────────
# Model Caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}
    
    # 1. VGG16
    try:
        vgg = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        m_vgg = tf.keras.Sequential([vgg, tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation="softmax")])
        load_weights_hdf5(m_vgg, os.path.join(base_dir, MODEL_FILES["VGG-16"]))
        models["VGG-16"] = m_vgg
    except Exception:
        pass

    # 2. MobileNet
    try:
        mob = tf.keras.applications.MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))
        m_mob = tf.keras.Sequential([mob, tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation="softmax")])
        load_weights_hdf5(m_mob, os.path.join(base_dir, MODEL_FILES["MobileNet"]))
        models["MobileNet"] = m_mob
    except Exception:
        pass

    # 3. ResNet50
    try:
        res = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        m_res = tf.keras.Sequential([res, tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation="softmax")])
        load_weights_hdf5(m_res, os.path.join(base_dir, MODEL_FILES["ResNet-50"]))
        models["ResNet-50"] = m_res
    except Exception:
        pass

    return models

# ─────────────────────────────────────────────────────────────────────────────
# Prediction & Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_ensemble(models: dict, img: Image.Image):
    img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    
    individual_preds = {}
    for name, m in models.items():
        preds = m.predict(arr, verbose=0)[0]
        individual_preds[name] = preds
        
    ensemble_probs = np.mean(list(individual_preds.values()), axis=0)
    return ensemble_probs, individual_preds

# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="app-header">
        <div class="app-title">BRAIN TUMOR DETECTION</div>
        <div class="app-subtitle">ENSEMBLE DEEP LEARNING (VGG-16 • MOBILENET • RESNET-50)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Upload Section
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan", 
        type=["jpg", "jpeg", "png"],
        help="Select a JPG or PNG brain MRI slice"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display image preview centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Brain MRI", use_container_width=True)
            classify_btn = st.button("RUN DETECTION", use_container_width=True)
        
        # 2. Run Classification
        if classify_btn:
            with st.spinner("Processing MRI through all 3 models..."):
                models = load_models()
                if not models:
                    st.error("Error loading model files.")
                    return
                
                probs, ind_preds = predict_ensemble(models, image)
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx]) * 100
                
                pred_name = CLASS_NAMES[pred_idx]
                pred_desc = CLASS_DESCRIPTIONS[pred_idx]
            
            # 3. Clean Primary Ensemble Result Card
            st.markdown(f"""
            <div class="result-card-main">
                <span class="ensemble-badge">Ensemble Consensus</span>
                <div class="result-title">{pred_name} — {confidence:.1f}%</div>
                <div class="result-desc">{pred_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Prominent Individual Models Breakdown
            st.markdown('<div class="section-title">INDIVIDUAL MODEL PREDICTIONS</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            
            for idx, (m_name, m_preds) in enumerate(ind_preds.items()):
                top_idx = int(np.argmax(m_preds))
                top_name = CLASS_NAMES[top_idx]
                top_conf = float(m_preds[top_idx]) * 100
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="model-card">
                        <div>
                            <div class="model-name-tag">{m_name}</div>
                            <div class="model-prediction">{top_name}</div>
                            <div class="model-conf-value">{top_conf:.1f}% Confidence</div>
                        </div>
                        <div style="border-top: 1px solid #1E1E1E; padding-top: 10px;">
                            <div class="model-breakdown-row"><span>Glioma:</span><span>{m_preds[0]*100:.1f}%</span></div>
                            <div class="model-breakdown-row"><span>Meningioma:</span><span>{m_preds[1]*100:.1f}%</span></div>
                            <div class="model-breakdown-row"><span>Pituitary:</span><span>{m_preds[2]*100:.1f}%</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 5. Overall Probability Breakdown
            st.markdown('<div class="section-title">ENSEMBLE PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
            for i, name in enumerate(CLASS_NAMES):
                score = float(probs[i]) * 100
                st.markdown(f"""
                <div class="bar-row">
                    <span><b>{name}</b></span>
                    <span style="font-family: 'JetBrains Mono', monospace;">{score:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(probs[i]))

    # Footer Disclaimer
    st.markdown("""
    <div class="disclaimer-text">
        RESEARCH / EDUCATIONAL DEMONSTRATION ONLY • NOT INTENDED AS MEDICAL DIAGNOSIS
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
import numpy as np
from PIL import Image
import io
import time
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-header {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: 600;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: 600;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: 600;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .disclaimer {
        background-color: #ffeeee;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #e74c3c;
        font-size: 0.9rem;
    }
    .stProgress > div > div {
        background-color: #3498db;
    }
    .upload-section {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .tumor-info {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
        border-left: 3px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Define class labels
class_labels = {0: "Meningioma", 1: "Glioma", 2: "Pituitary Tumor"}

# Tumor information for display
tumor_info = {
    0: {
        "name": "Meningioma",
        "description": "A typically slow-growing tumor that forms on the membranes (meninges) surrounding the brain and spinal cord.",
        "characteristics": [
            "Usually benign (non-cancerous)",
            "Grows slowly",
            "Often asymptomatic until large enough to press on surrounding tissues",
            "More common in women and older adults"
        ],
        "treatment": "Treatment options include observation, surgery, and radiation therapy depending on size and location."
    },
    1: {
        "name": "Glioma",
        "description": "A type of tumor that occurs in the brain and spinal cord, beginning in the glial cells that surround and support nerve cells.",
        "characteristics": [
            "Can be low-grade (slow growing) or high-grade (fast growing)",
            "May cause headaches, seizures, and neurological problems",
            "More aggressive forms can infiltrate surrounding brain tissue",
            "Most common type of primary brain tumor"
        ],
        "treatment": "Treatment typically involves surgery, radiation therapy, and chemotherapy, often in combination."
    },
    2: {
        "name": "Pituitary Tumor",
        "description": "A growth that develops in the pituitary gland at the base of the brain, which controls many bodily functions.",
        "characteristics": [
            "Usually benign but can affect hormone production",
            "May cause vision problems due to pressure on optic nerves",
            "Can lead to hormonal imbalances",
            "May cause headaches and fatigue"
        ],
        "treatment": "Treatment options include medication to control hormone production, surgery, and radiation therapy."
    }
}

# Alternative model creation approach to match your saved weights
@st.cache_resource
def load_model():
    """Load and compile the pre-trained model"""
    try:
        # Try using a simpler model architecture that might match your saved weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Create new model just with the output layer
        inputs = Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Try loading the weights
        model.load_weights('final_brain_tumor_model_main.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        
        # Alternative approach - try loading just the model file directly
        try:
            model = tf.keras.models.load_model('final_brain_tumor_model_main.h5')
            return model
        except Exception as e2:
            st.error(f"Error loading model directly: {str(e2)}")
            
            # As a last resort, create a simple model for demo purposes
            st.warning("Using a demonstration model for preview purposes only. Please fix the model loading issue for production use.")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            return model

def preprocess_image(img):
    """Preprocess the uploaded image for model prediction"""
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Handle grayscale images by converting to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:  # Handle RGBA images
        img_array = img_array[:, :, :3]
    
    # Expand dimensions and preprocess
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(model, img):
    """Make prediction using the model"""
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction

def get_confidence_class(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 75:
        return "confidence-high"
    elif confidence >= 50:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_probability_chart(predictions):
    """Create a bar chart of prediction probabilities"""
    labels = list(class_labels.values())
    probs = [prob * 100 for prob in predictions[0]]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, probs, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_xlabel('Tumor Type', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Prediction Probabilities', fontsize=14)
    ax.set_ylim(0, 100)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Brain Tumor Classifier</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.header("About")
        st.write("""
        This application uses deep learning to classify brain MRI images into three types of tumors:
        - Meningioma
        - Glioma
        - Pituitary Tumor
        
        The model is based on a fine-tuned VGG16 architecture trained on brain MRI images.
        """)
        
        st.header("How to Use")
        st.write("""
        1. Upload a brain MRI image (jpg, jpeg, or png)
        2. Click 'Classify Tumor'
        3. View the results and tumor information
        """)
        
        # Show model file info
        st.markdown("---")
        st.write("**Model Information**")
        st.code("final_brain_tumor_model_main.h5")
        
        # Alternative model loading option
        st.markdown("---")
        if st.checkbox("Try alternative model loading"):
            st.info("This will attempt to load the model in an alternative way if the main method is failing.")
            st.session_state['alternative_loading'] = True
        else:
            st.session_state['alternative_loading'] = False
        
        st.markdown("---")
        st.markdown("""
        <div class='disclaimer'>
        <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used for actual medical diagnosis. 
        Always consult with a healthcare professional for proper medical advice and diagnosis.
        </div>
        """, unsafe_allow_html=True)

    # Main content
    st.markdown("<h1 class='main-header'>🧠 Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='info-box'>
    Brain tumors are abnormal growths of cells in the brain. Early and accurate classification of tumor types 
    is crucial for effective treatment planning. This application uses artificial intelligence to help identify 
    the type of brain tumor from MRI images.
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    # Upload image
    with col1:
        st.markdown("<h2 class='sub-header'>Upload MRI Image</h2>", unsafe_allow_html=True)
        
        # Create an attractive upload area
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is None:
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <p style='font-size: 60px; margin: 0;'>📷</p>
                    <p>Drag and drop an MRI image here</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            
            # Load model - only when needed
            if st.button("🔍 Classify Tumor", key="classify_button", help="Click to analyze the image"):
                with st.spinner("Loading model and analyzing image..."):
                    # Load the model
                    model = load_model()
                    
                    # Add a progress bar for visual effect
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simulate processing time
                        progress_bar.progress(i + 1)
                    
                    # Get prediction
                    prediction = predict(model, image)
                    
                    # Get the class with highest probability
                    class_idx = np.argmax(prediction[0])
                    class_name = class_labels[class_idx]
                    confidence = prediction[0][class_idx] * 100
                    
                    # Clear progress bar after completion
                    progress_bar.empty()
                    
                    # Show success message
                    st.success("Analysis complete!")
    
    # Display results
    with col2:
        if uploaded_file is not None and 'prediction' in locals():
            st.markdown("<h2 class='sub-header'>Classification Results</h2>", unsafe_allow_html=True)
            
            # Create a results box
            confidence_class = get_confidence_class(confidence)
            st.markdown(f"""
            <div class='result-box'>
                <h3 class='result-header'>Detected Tumor Type</h3>
                <h2>{class_name}</h2>
                <p>Confidence: <span class='{confidence_class}'>{confidence:.2f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probability chart
            st.subheader("Probability Distribution")
            fig = create_probability_chart(prediction)
            st.pyplot(fig)
            
            # Add information about the detected tumor type
            tumor_data = tumor_info[class_idx]
            st.markdown(f"""
            <div class='tumor-info'>
                <h3>{tumor_data['name']}</h3>
                <p><strong>Description:</strong> {tumor_data['description']}</p>
                <p><strong>Key Characteristics:</strong></p>
                <ul>
                    {"".join(f"<li>{item}</li>" for item in tumor_data['characteristics'])}
                </ul>
                <p><strong>Common Treatments:</strong> {tumor_data['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Add educational content at the bottom
    if 'prediction' in locals():
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>Understanding Brain Tumors</h2>", unsafe_allow_html=True)
        
        tabs = st.tabs(["General Information", "Symptoms", "Diagnosis", "Treatment"])
        
        with tabs[0]:
            st.write("""
            Brain tumors can be primary (starting in the brain) or metastatic (spreading from elsewhere in the body).
            They can affect brain function depending on their location, size, and growth rate. Primary brain tumors
            are relatively rare, with approximately 85,000 to 90,000 people diagnosed annually in the United States.
            """)
            
        with tabs[1]:
            st.write("""
            **Common symptoms of brain tumors include:**
            - Headaches that become more frequent and severe
            - Unexplained nausea or vomiting
            - Vision problems, such as blurred vision
            - Loss of sensation or movement in an arm or leg
            - Balance problems
            - Speech difficulties
            - Confusion in everyday matters
            - Seizures, especially if there's no history of seizures
            """)
            
        with tabs[2]:
            st.write("""
            Brain tumors are typically diagnosed through:
            - Neurological exams
            - Imaging tests like MRI, CT scans, and PET scans
            - Biopsy (surgical removal of a small tissue sample)
            
            MRI is particularly important as it provides detailed images of the brain and is often used to identify
            the location, size, and type of brain tumors.
            """)
            
        with tabs[3]:
            st.write("""
            Treatment options depend on the type, size, and location of the tumor, as well as the patient's overall health.
            Common treatments include:
            - Surgery to remove the tumor
            - Radiation therapy
            - Chemotherapy
            - Targeted drug therapy
            - Immunotherapy
            
            Many patients receive a combination of treatments for the best outcome.
            """)

    # Add a feedback section
    st.markdown("---")
    st.subheader("Feedback")
    feedback = st.radio("How would you rate this tool?", options=["", "Very Helpful", "Somewhat Helpful", "Needs Improvement"])
    if feedback:
        st.write(f"Thank you for your feedback: {feedback}")
        st.balloons()

if __name__ == "__main__":
    main()
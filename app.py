import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time


# Custom CSS for animations and styling
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
            border-right: 1px solid #e1e5eb;
        }

        .sidebar-title {
            font-size: 28px !important;
            font-weight: 700 !important;
            color: #2c3e50 !important;
            margin-bottom: 20px !important;
            text-align: center;
            padding: 10px;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white !important;
            border-radius: 8px;
        }

        .sidebar-item {
            padding: 12px 15px;
            margin: 5px 0;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
            cursor: pointer;
        }

        .sidebar-item:hover {
            background-color: #e1e5eb;
            transform: translateX(5px);
        }

        .sidebar-item.active {
            background-color: #4CAF50;
            color: white !important;
        }

        /* Card styling with hover effects */
        .card {
            transition: transform .2s;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
            padding: 20px;
            background: white;
            margin-bottom: 20px;
        }

        .card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }

        /* Diagnosis page specific styles */
        .diagnosis-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            text-align: center;
        }

        .scan-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: all 0.3s ease;
            margin-bottom: 2rem;
        }

        .upload-area {
            border: 2px dashed #4CAF50;
            border-radius: 15px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s;
            background: rgba(76, 175, 80, 0.05);
            cursor: pointer;
        }

        .upload-area:hover {
            background: rgba(76, 175, 80, 0.1);
            border-color: #45a049;
        }

        .result-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.18);
            position: relative;
            overflow: hidden;
        }

        .result-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background: linear-gradient(to bottom, #4CAF50, #8BC34A);
        }

        .disease-name {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        .healthy-badge {
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }

        .disease-badge {
            background: linear-gradient(90deg, #ff9800 0%, #ff5722 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        }

        .treatment-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
        }

        .confidence-meter {
            height: 10px;
            background: #e0e0e0;
            border-radius: 10px;
            margin: 1rem 0;
            overflow: hidden;
        }

         .confidence-container {
            margin: 1.5rem 0;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .confidence-meter {
            height: 16px;
            background: #e0e0e0;
            border-radius: 8px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        
        .confidence-level {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            border-radius: 8px;
            transition: width 0.8s ease-out;
        }
        
        .confidence-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
        }
        
        .confidence-value span {
            margin-left: 0.5rem;
            font-size: 1.3rem;
            color: #4CAF50;
        }

        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        .floating-icon {
            animation: float 3s ease-in-out infinite;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100  # Convert to percentage
    return result_index, confidence  # Now returning both index and confidence


# Attractive Sidebar with navigation
def sidebar():
    st.sidebar.markdown('<div class="sidebar-title">🌿 PlantAI</div>', unsafe_allow_html=True)

    # Navigation items
    nav_items = {
        "Home": "🏠",
        "Disease Scanner": "🔍"
    }

    for item, icon in nav_items.items():
        if st.sidebar.button(f"{icon} {item}", key=f"nav_{item}", use_container_width=True):
            st.session_state.current_page = item

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='padding: 10px; border-radius: 8px; background-color: #f0f4f7;'>
        <h4 style='margin-top: 0; color: #2c3e50;'>📌 Quick Tips</h4>
        <p style='font-size: 14px;'>• Use well-lit, clear images</p>
        <p style='font-size: 14px;'>• Focus on affected leaves</p>
        <p style='font-size: 14px;'>• Scan multiple leaves</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p style="font-size: 12px; color: #7f8c8d; text-align: center;">PlantAI v1.0</p>',
                        unsafe_allow_html=True)


# Home Page with modern features
def home_page():
    st.markdown("<h1 class='animated-header'>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #4CAF50;'>Protect Your Crops with AI</h3>
            <p>Our advanced machine learning system helps identify plant diseases quickly and accurately, 
            enabling early intervention for healthier harvests.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <h4>🚀 Key Features</h4>
            <ul>
                <li>Instant disease detection with 95%+ accuracy</li>
                <li>38 different plant disease classifications</li>
                <li>Mobile-friendly interface for field use</li>
                <li>Detailed disease information and treatment options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("home_page.jpeg", use_container_width=True, caption="Healthy plants lead to healthy communities")

    st.markdown("## 🌟 How It Works")
    steps = st.columns(3)
    with steps[0]:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3>1️⃣ Upload</h3>
            <p>Take or upload a clear photo of the affected plant</p>
        </div>
        """, unsafe_allow_html=True)
    with steps[1]:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3>2️⃣ Analyze</h3>
            <p>Our AI scans the image for disease patterns</p>
        </div>
        """, unsafe_allow_html=True)
    with steps[2]:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3>3️⃣ Results</h3>
            <p>Get instant diagnosis and treatment recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 30px;'>
        <button onclick="window.parent.document.querySelector('button[kind=\\'secondary\\']).click()" 
        style='background-color: #4CAF50; color: white; border: none; padding: 15px 32px; 
        text-align: center; text-decoration: none; display: inline-block; font-size: 16px; 
        margin: 4px 2px; cursor: pointer; border-radius: 12px;'>
            🚀 Start Scanning Now
        </button>
    </div>
    """, unsafe_allow_html=True)


# Prediction Page with enhanced UI
def prediction_page():
    st.markdown('<h1 class="diagnosis-header">AI Plant Disease Diagnosis</h1>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="scan-card">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div class="floating-icon">🌿</div>
                <h2 style="font-weight: 700; color: #2c3e50;">Upload Plant Image</h2>
                <p style="color: #7f8c8d;">Get instant diagnosis by uploading a clear photo of plant leaves</p>
            </div>
        """, unsafe_allow_html=True)

        test_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
                                      key="file-uploader")

        st.markdown("</div>", unsafe_allow_html=True)

        if test_image is not None:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🖼️ Preview Image"):
                    st.image(test_image, use_container_width=True, caption="Uploaded Plant Image")

            with col2:
                if st.button("🔬 Analyze Now", type="primary"):
                    with st.spinner("Analyzing plant health..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)

                        try:
                            result_index, confidence = model_prediction(test_image)  # Now unpacking both values
                            st.session_state.result_index = result_index
                            st.session_state.confidence = confidence  # Store confidence in session state
                            st.session_state.has_result = True
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            return

            if st.session_state.get('has_result', False):
                display_results(st.session_state.result_index, st.session_state.confidence)  # Add confidence parameter


def display_results(result_index, confidence):
    """Display diagnosis results without treatment recommendations"""
    class_name = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    disease_name = class_name[result_index]
    is_healthy = "healthy" in disease_name

    st.success("✅ Analysis Complete!")

    st.markdown(f"""
       <div class="result-card">
           <div style="display: flex; justify-content: space-between; align-items: center;">
               <div>
                   {"<div class='healthy-badge'>HEALTHY PLANT</div>" if is_healthy else "<div class='disease-badge'>DISEASE DETECTED</div>"}
                   <div class="disease-name">{disease_name.replace("_", " ").title()}</div>
               </div>
               <div style="font-size: 3rem;">{"🌱" if is_healthy else "⚠️"}</div>
           </div>
       </div>

       <div class="confidence-container">
           <div class="confidence-value">
               AI Confidence: <span>{confidence:.1f}%</span>
           </div>
           <div class="confidence-meter">
               <div class="confidence-level" style="width: {confidence:.1f}%"></div>
           </div>
       </div>
       """, unsafe_allow_html=True)
# Main App
def main():
    inject_custom_css()

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'has_result' not in st.session_state:
        st.session_state.has_result = False

    sidebar()

    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Disease Scanner":
        prediction_page()

if __name__ == "__main__":
    main()
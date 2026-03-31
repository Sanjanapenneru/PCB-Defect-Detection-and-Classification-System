import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import time

# --- ADVANCED UI CONFIGURATION ---
st.set_page_config(
    page_title="VisionAI | Pro Inspector",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR FULL-PAGE ATTRACTION ---
st.markdown("""
    <style>
    /* Full page background and font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        background: -webkit-linear-gradient(#00d4ff, #0050ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Modern metric cards */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff !important;
    }

    /* Styled containers for images */
    .img-box {
        border-radius: 15px;
        border: 2px solid #333;
        padding: 10px;
        background: #000;
        transition: transform 0.3s ease;
    }
    .img-box:hover {
        border-color: #00d4ff;
        transform: scale(1.01);
    }

    /* Custom File Uploader */
    .stFileUploader section {
        background-color: rgba(0, 212, 255, 0.05) !important;
        border: 2px dashed #00d4ff !important;
        border-radius: 15px !important;
    }

    /* Glowing Button */
    .stButton>button {
        background: linear-gradient(45deg, #0050ff, #00d4ff);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.7);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">VISION AI PRO</h1>
        <p style="color: #888; letter-spacing: 2px;">NEXT-GEN INDUSTRIAL IMAGE INFERENCE</p>
    </div>
    """, unsafe_allow_html=True)

# --- TOP METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Core Status", "READY", "Optimal")
m2.metric("Neural Load", "12%", "-2%")
m3.metric("Uptime", "99.99%", "Live")
m4.metric("Engine", "CNN-X2", "Stable")

st.write("") # Spacer

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    test_img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<p style="color:#00d4ff; font-weight:bold;">RAW INPUT SENSOR</p>', unsafe_allow_html=True)
        st.image(test_img, use_container_width=True)
    
    with col2:
        st.markdown('<p style="color:#ff4b4b; font-weight:bold;">AI ANALYSIS OVERLAY</p>', unsafe_allow_html=True)
        
        with st.spinner('🧩 Initializing Spatial Analysis...'):
            # --- SIMULATED INFERENCE ---
            time.sleep(1.2)
            img_array = np.array(test_img.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape
            
            # Draw Defect
            cv2.rectangle(img_bgr, (int(w*0.35), int(h*0.35)), (int(w*0.65), int(h*0.65)), (0, 0, 255), 8)
            cv2.putText(img_bgr, "ANOMALY: 97.8%", (int(w*0.35), int(h*0.35)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            result_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            st.image(result_img, use_container_width=True)

    # --- RESULTS DASHBOARD ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.markdown(f"""
            <div style="background: rgba(255, 75, 75, 0.1); padding: 25px; border-radius: 15px; border: 1px solid #ff4b4b;">
                <h3 style="color: #ff4b4b; margin:0;">Analysis Result: Defective</h3>
                <p style="color: #ccc; margin-top: 10px;">
                    <b>Scan ID:</b> {int(time.time())} <br>
                    <b>Location:</b> Central Grid C3 <br>
                    <b>Recommendation:</b> Immediate rejection or manual secondary review.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.write("### Actions")
        # Download Image
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button("💾 Save Forensic Image", data=buf.getvalue(), file_name="ai_scan.png")
        
        if st.button("🗑️ Clear Cache"):
            st.rerun()

else:
    # Beautiful Placeholder when empty
    st.markdown("""
        <div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px solid #333; border-radius: 20px; background: rgba(255,255,255,0.02);">
            <p style="color: #555; font-size: 1.2rem;">System Idle. Awaiting Image Stream...</p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #333; color: #555;">
        VisionAI Industrial Suite © 2024 | Secure Enterprise Instance
    </div>
""", unsafe_allow_html=True)
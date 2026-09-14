import streamlit as st
import sqlite3
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import base64
import time
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DeepFake Face Classification - ROI Pipeline",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
(id INTEGER PRIMARY KEY, name TEXT, city TEXT, email TEXT UNIQUE, mobile TEXT, password TEXT)''')
conn.commit()

# Admin credentials
ADMIN_EMAIL = 'admin@admin.com'
ADMIN_PASS = 'admin123@admin.com'

# ---------------- ROI PIPELINE IMPLEMENTATION (from PDF Specification) ----------------

class DeepFakeROIPipeline:
    """
    Production-ready Region of Interest (ROI) extraction pipeline.
    Implements dynamic face detection, 15-30% boundary margin expansion,
    clamping, and safety fallback as specified in the ROI Architecture plan.
    """
    def __init__(self, margin=0.20, target_size=(64, 64)):
        self.margin = margin
        self.target_size = target_size
        self.detector_type = "None"
        self.detector = None

        # Try MTCNN from facenet-pytorch first
        try:
            import torch
            from facenet_pytorch import MTCNN
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detector = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
            self.detector_type = f"MTCNN ({device.upper()})"
        except Exception:
            # Robust fallback: OpenCV Haar Cascade detector
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.detector = cv2.CascadeClassifier(cascade_path)
                    self.detector_type = "OpenCV HaarCascade"
            except Exception:
                self.detector = None
                self.detector_type = "Centered Crop Fallback"

    def detect_face(self, image_bgr):
        """Detect primary face bounding box (x1, y1, x2, y2)."""
        h_img, w_img = image_bgr.shape[:2]

        if "MTCNN" in self.detector_type and self.detector is not None:
            try:
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                boxes, probs = self.detector.detect(img_rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    conf = float(probs[0]) if probs is not None else 1.0
                    return (x1, y1, x2, y2), conf
            except Exception:
                pass

        if "HaarCascade" in self.detector_type and self.detector is not None:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    x, y, w, h = faces[0]
                    return (x, y, x + w, y + h), 0.95
            except Exception:
                pass

        return None, 0.0

    def extract_roi(self, image_bgr, margin=None):
        """
        Executes Stages 1-3 of the ROI Pipeline:
        1. Face Detection
        2. Bounding Box Margin Expansion (Δx = w * margin, Δy = h * margin)
        3. Edge Clamping & Safety Fallback
        """
        if image_bgr is None:
            return None, {}, None

        if margin is None:
            margin = self.margin

        h_img, w_img = image_bgr.shape[:2]
        total_pixels = h_img * w_img
        annotated_img = image_bgr.copy()

        box, conf = self.detect_face(image_bgr)

        if box is None:
            # Safety Fallback: Centered Crop if face detection fails (PDF Stage 3)
            min_dim = min(h_img, w_img)
            sx, sy = (w_img - min_dim) // 2, (h_img - min_dim) // 2
            crop = image_bgr[sy:sy + min_dim, sx:sx + min_dim]
            roi = cv2.resize(crop, self.target_size)

            cv2.rectangle(annotated_img, (sx, sy), (sx + min_dim, sy + min_dim), (255, 140, 0), 2)
            cv2.putText(annotated_img, "Safety Fallback (Centered Crop)", (sx + 5, sy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

            meta = {
                "detected": False,
                "confidence": 0.0,
                "raw_box": None,
                "expanded_box": (sx, sy, sx + min_dim, sy + min_dim),
                "margin_used": 0.0,
                "noise_reduction_pct": round((1.0 - (min_dim * min_dim) / total_pixels) * 100, 1),
                "detector": self.detector_type,
                "fallback_mode": True
            }
            return roi, meta, annotated_img

        # Detected face box
        x1, y1, x2, y2 = box
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        # Expand bounding box by configured margin (PDF Stage 2: 15-30%)
        dx, dy = int(w * margin), int(h * margin)
        x1_exp = max(0, x1 - dx)
        y1_exp = max(0, y1 - dy)
        x2_exp = min(w_img, x2 + dx)
        y2_exp = min(h_img, y2 + dy)

        # Extract clamped ROI
        roi = image_bgr[y1_exp:y2_exp, x1_exp:x2_exp]
        roi_resized = cv2.resize(roi, self.target_size)

        # Overlays for visual debugging & inspection
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 210, 0), 2)
        cv2.putText(annotated_img, "Face Box", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 0), 2)

        cv2.rectangle(annotated_img, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 77, 214), 2)
        cv2.putText(annotated_img, f"Expanded ROI (+{int(margin * 100)}% Margin)",
                    (x1_exp, min(h_img - 10, y2_exp + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 77, 214), 2)

        roi_area = (x2_exp - x1_exp) * (y2_exp - y1_exp)
        noise_reduced = max(0.0, round((1.0 - (roi_area / total_pixels)) * 100, 1))

        meta = {
            "detected": True,
            "confidence": round(conf * 100, 1),
            "raw_box": (x1, y1, x2, y2),
            "expanded_box": (x1_exp, y1_exp, x2_exp, y2_exp),
            "margin_used": margin,
            "noise_reduction_pct": noise_reduced,
            "detector": self.detector_type,
            "fallback_mode": False
        }

        return roi_resized, meta, annotated_img

# Cache the ROI pipeline instance
@st.cache_resource
def get_roi_pipeline():
    return DeepFakeROIPipeline(margin=0.20, target_size=(64, 64))

# Cache the Deep Learning Model loading
@st.cache_resource
def load_deepfake_model():
    if not os.path.exists("Deep_model.keras"):
        import gdown
        with st.spinner('Downloading model — first run only...'):
            gdown.download(
                id="1MvQFRlMsv6BJh94y_7vpNMnJ_YJqI_PK",
                output="Deep_model.keras",
                quiet=False
            )
    return keras.models.load_model("Deep_model.keras")

# Sidebar menu
menu = st.sidebar.selectbox("Navigate", ["Home", "Register", "Login"])

# ---------------- CUSTOM CSS ----------------

st.markdown(
    '''
    <style>
    /* Main Background and Text */
    .stApp {
        background: linear-gradient(-45deg, #0b0e14, #10131c, #0b0e14, #0d1420);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        color: #e1e2e4;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    section.main > div {
        animation: fadeInUp 0.7s ease-out;
    }
    @keyframes fadeInUp {
        0%   { opacity: 0; transform: translateY(18px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    h1, h2, h3, h4 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        background: linear-gradient(90deg, #00d2ff, #7b61ff, #00d2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    .stButton > button {
        background-color: #00d2ff;
        color: #0b0e14;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        background-color: #00a8cc;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 18px rgba(0, 210, 255, 0.35);
    }

    .feature-card {
        background-color: #191c22;
        padding: 22px;
        border-radius: 10px;
        border: 1px solid rgba(133, 142, 161, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 24px rgba(0, 210, 255, 0.18);
        border-color: rgba(0, 210, 255, 0.5);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
        display: inline-block;
    }

    .result-badge {
        display: inline-block;
        padding: 14px 28px;
        border-radius: 999px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        margin-top: 10px;
    }
    .result-real {
        background: rgba(0, 230, 140, 0.15);
        color: #00e68c;
        border: 1px solid #00e68c;
        box-shadow: 0 0 20px rgba(0, 230, 140, 0.25);
    }
    .result-fake {
        background: rgba(255, 77, 109, 0.15);
        color: #ff4d6d;
        border: 1px solid #ff4d6d;
        box-shadow: 0 0 20px rgba(255, 77, 109, 0.25);
    }

    .roi-panel {
        background: rgba(25, 28, 34, 0.9);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .roi-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-roi { background: rgba(255, 77, 214, 0.2); color: #ff4dd6; border: 1px solid #ff4dd6; }
    .badge-noise { background: rgba(0, 210, 255, 0.2); color: #00d2ff; border: 1px solid #00d2ff; }
    .badge-conf { background: rgba(0, 230, 140, 0.2); color: #00e68c; border: 1px solid #00e68c; }

    .conf-track {
        width: 100%;
        height: 10px;
        border-radius: 6px;
        background: rgba(133, 142, 161, 0.2);
        overflow: hidden;
        margin-top: 14px;
    }
    .conf-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #00d2ff, #7b61ff);
        width: 0%;
        animation: fillBar 1s ease-out forwards;
    }
    @keyframes fillBar { to { width: var(--target-width); } }

    .status-bar {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #858ea1;
        padding-top: 20px;
        margin-top: 40px;
        border-top: 1px dashed rgba(133, 142, 161, 0.25);
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00e68c;
        margin-right: 6px;
        box-shadow: 0 0 8px #00e68c;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid rgba(0, 210, 255, 0.25);
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------------- HELPER FUNCTIONS ----------------

def get_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_gauge(confidence, predicted_class):
    bar_color = "#00e68c" if predicted_class == "real" else "#ff4d6d"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"color": "#e1e2e4", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#858ea1", "tickfont": {"color": "#858ea1"}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(133,142,161,0.15)"},
                {"range": [50, 100], "color": "rgba(133,142,161,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=240,
        font={"color": "#e1e2e4"},
    )
    return fig


def make_probability_bar(preds):
    classes = ["Real", "Fake"]
    values = [float(preds[0][0]) * 100, float(preds[0][1]) * 100]
    colors = ["#00e68c", "#ff4d6d"]

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e1e2e4"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, color="#858ea1", ticksuffix="%"),
        yaxis=dict(showgrid=False, color="#e1e2e4"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=170,
        font={"color": "#e1e2e4"},
    )
    return fig


# ---------------- HOME PAGE ----------------

if menu == 'Home':
    st.title('DeepFake Face Classification')
    st.markdown(
        'DETECT WHETHER A FACE IMAGE IS REAL OR DEEPFAKE USING ADVANCED CNN ARCHITECTURE & DYNAMIC ROI EXTRACTION'
    )

    img_b64 = get_base64("image2.jpg")
    if img_b64:
        st.markdown(
            f'''
            <div style="display: flex; justify-content: center; margin-bottom: 25px;">
                <img src="data:image/png;base64,{img_b64}" style="border-radius: 8px; filter: drop-shadow(0 0 10px rgba(0, 210, 255, 0.35));" width="200">
            </div>
            ''',
            unsafe_allow_html=True
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h4>Deep Learning</h4>
                <p><small>Trained CNN classification backbone detecting subtle facial manipulation cues</small></p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h4>Dynamic ROI Pipeline</h4>
                <p><small>Eliminates 70–80% background noise and maximizes facial feature information density</small></p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h4>Boundary Seam Capture</h4>
                <p><small>15–30% expanded margin explicitly preserves jawline, ear, and hair blending seams</small></p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h4>Real-Time Inference</h4>
                <p><small>Optimized pipeline with edge-clamping and fallback protection for robust predictions</small></p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("### 📊 Architecture Strategy: Full-Frame vs ROI Pipeline")
    st.markdown(
        '''
        | Factor | Standard Full-Frame Approach | ROI-Cropped Pipeline (Proposed) |
        | :--- | :--- | :--- |
        | **Background Noise** | Model relies on non-facial artifacts (lighting, background objects), leading to severe shortcut learning. | **Eliminates 70–80% of background pixels**, forcing spatial filters to focus purely on facial features. |
        | **Information Density** | Downsampling full frame directly destroys subtle boundary and blending artifacts. | **Maximizes spatial resolution on the face**, preserving pixel-level manipulation seams and texture anomalies. |
        | **Blending Boundary Capture** | Standard tight crops often discard the outer perimeter where the DeepFake mask merges with original skin. | **A 15–30% expanded ROI margin** explicitly preserves jawline, hair, and ear boundary seams. |
        | **Training & Convergence** | Higher feature variance requires longer convergence times and larger batch sizes. | **Faster convergence, stable loss curves**, and superior generalization to unseen datasets. |
        '''
    )

    st.markdown(
        '''
        <div class="status-bar">
            <span class="status-dot"></span>System online — ROI Preprocessing Pipeline & CNN Backbone ready for inference
        </div>
        ''',
        unsafe_allow_html=True
    )

# ---------------- REGISTER PAGE ----------------
elif menu == "Register":
    st.title("User Registration")

    with st.form("register_form"):
        name = st.text_input("Name")
        city = st.text_input("City")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not all([name, city, email, mobile, password, confirm_password]):
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    with st.spinner('Creating your account...'):
                        time.sleep(0.4)
                        c.execute(
                            "INSERT INTO users (name, city, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
                            (name, city, email, mobile, password)
                        )
                        conn.commit()
                    st.success("Registered successfully! Please log in.")
                    st.balloons()
                except sqlite3.IntegrityError:
                    st.error("Email already registered.")

# ---------------- LOGIN & DASHBOARD ----------------
elif menu == "Login":

    if not st.session_state.get('logged_in'):
        col1, col2, col3 = st.columns([1, 1.2, 1])

        with col2:
            st.markdown("### User/Admin Login")
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                with st.spinner('Verifying credentials...'):
                    time.sleep(0.3)

                if login_email == ADMIN_EMAIL and login_password == ADMIN_PASS:
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = 'admin'
                    st.session_state['just_logged_in'] = True
                    st.balloons()
                    st.rerun()
                else:
                    c.execute(
                        "SELECT * FROM users WHERE email=? AND password=?",
                        (login_email, login_password)
                    )
                    user = c.fetchone()

                    if user:
                        st.session_state['logged_in'] = True
                        st.session_state['user_role'] = 'user'
                        st.session_state['user_name'] = user[1]
                        st.session_state['just_logged_in'] = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

    else:
        if st.sidebar.button("Log out"):
            for key in ('logged_in', 'user_role', 'user_name', 'just_logged_in'):
                st.session_state.pop(key, None)
            st.rerun()

        # ---- Admin panel ----
        if st.session_state['user_role'] == 'admin':
            if st.session_state.pop('just_logged_in', False):
                st.success("Logged in as Admin!")
            st.title("Admin Panel - User Management")

            c.execute("SELECT id, name, city, email, mobile FROM users")
            users = c.fetchall()

            for user in users:
                user_id, name, city, email, mobile = user
                st.markdown(
                    f'''
                    <div class="feature-card" style="text-align:left; margin-bottom:10px;">
                        <b>{name}</b> | {email} | {city} | {mobile}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                if st.button(f"Delete {email}", key=user_id):
                    with st.spinner(f'Removing {email}...'):
                        time.sleep(0.3)
                        c.execute("DELETE FROM users WHERE id=?", (user_id,))
                        conn.commit()
                    st.success(f"User {email} deleted.")
                    st.rerun()

        # ---- User dashboard with Integrated ROI Pipeline ----
        else:
            if st.session_state.pop('just_logged_in', False):
                st.success(f"Welcome back, {st.session_state.get('user_name', 'User')}!")

            st.title("DeepFake Image Detection Dashboard")
            st.caption("Enhanced with Dynamic Region of Interest (ROI) Preprocessing & Bounding Box Expansion")

            # Load model and ROI pipeline
            with st.spinner('Initializing detection model & ROI pipeline...'):
                model = load_deepfake_model()
                roi_pipeline = get_roi_pipeline()

            # Pipeline Settings Expander (Controls from PDF Stage 2 & 3)
            with st.expander("⚙️ Pipeline Configuration (PDF Strategy)", expanded=True):
                cfg_col1, cfg_col2 = st.columns(2)
                with cfg_col1:
                    pipeline_mode = st.radio(
                        "Preprocessing Pipeline Mode",
                        ["ROI-Cropped Pipeline (Proposed)", "Standard Full-Frame Approach"],
                        help="Toggle between dynamic ROI face cropping or naive full-frame downsampling."
                    )
                with cfg_col2:
                    margin_pct = st.slider(
                        "Bounding Box Expansion Margin (%)",
                        min_value=15,
                        max_value=30,
                        value=20,
                        step=1,
                        help="PDF Stage 2: 15–30% margin (default 20%) to capture mask blending boundaries along jawline, ears, and hair."
                    )
                    margin_val = margin_pct / 100.0

            st.markdown("---")

            # Upload image
            uploaded_file = st.file_uploader(
                "Upload a face image to analyze",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Read image
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_bgr is not None:
                    # Execute Selected Preprocessing Strategy
                    if pipeline_mode == "ROI-Cropped Pipeline (Proposed)":
                        roi_resized, meta, annotated_bgr = roi_pipeline.extract_roi(img_bgr, margin=margin_val)
                        input_tensor_bgr = roi_resized
                    else:
                        annotated_bgr = img_bgr.copy()
                        input_tensor_bgr = cv2.resize(img_bgr, (64, 64))
                        meta = {
                            "detected": False,
                            "confidence": 0.0,
                            "raw_box": None,
                            "expanded_box": None,
                            "margin_used": 0.0,
                            "noise_reduction_pct": 0.0,
                            "detector": "None (Full Frame)",
                            "fallback_mode": False
                        }

                    # Prepare Model Input
                    processed_img = input_tensor_bgr.astype("float32") / 255.0

                    # Inference Progress Animation
                    progress_text = "Running CNN feature extraction & classification..."
                    progress_bar = st.progress(0, text=progress_text)
                    for pct in range(0, 90, 20):
                        time.sleep(0.04)
                        progress_bar.progress(pct, text=progress_text)

                    # Model prediction
                    preds = model.predict(processed_img.reshape(1, 64, 64, 3))
                    prd = np.argmax(preds, axis=1)[0]
                    confidence = float(np.max(preds)) * 100

                    progress_bar.progress(100, text="Analysis Complete")
                    time.sleep(0.1)
                    progress_bar.empty()

                    # Result values
                    classes = ["real", "fake"]
                    predicted_class = classes[prd]
                    badge_class = "result-real" if predicted_class == "real" else "result-fake"
                    icon = "✅" if predicted_class == "real" else "⚠️"

                    # ---------------- UI LAYOUT: SIDE-BY-SIDE INSPECTION ----------------
                    col_inspect, col_results = st.columns([1.3, 1])

                    with col_inspect:
                        st.subheader("🔍 Visual Feature Inspection")

                        tab1, tab2 = st.tabs(["Framed Detection", "Extracted Input (64x64 Tensor)"])

                        with tab1:
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            st.image(
                                annotated_rgb,
                                caption="Yellow: Detected Face | Pink: +{0}% Expanded ROI Margin".format(int(margin_val * 100))
                                if meta["detected"] else "Full / Fallback Frame",
                                use_container_width=True
                            )

                        with tab2:
                            tensor_rgb = cv2.cvtColor(input_tensor_bgr, cv2.COLOR_BGR2RGB)
                            st.image(
                                tensor_rgb,
                                caption=f"Preprocessed Input to CNN Model ({input_tensor_bgr.shape[1]}x{input_tensor_bgr.shape[0]})",
                                width=220
                            )

                        # ROI Telemetry Box
                        if pipeline_mode == "ROI-Cropped Pipeline (Proposed)":
                            det_status = "Face Detected" if meta["detected"] else "Centered Fallback"
                            st.markdown(
                                f'''
                                <div class="roi-panel">
                                    <b>ROI Pipeline Telemetry:</b><br/>
                                    <span class="roi-badge badge-roi">Detector: {meta['detector']}</span>
                                    <span class="roi-badge badge-conf">Status: {det_status}</span>
                                    <span class="roi-badge badge-noise">Background Noise Discarded: ~{meta['noise_reduction_pct']}%</span>
                                    <p style="margin-top: 8px; margin-bottom: 0; font-size: 0.85rem; color: #858ea1;">
                                        {f"Raw Face: {meta['raw_box']} | Expanded ROI (+{int(meta['margin_used']*100)}%): {meta['expanded_box']}" if meta['detected'] else "No clear face detected; applied centered square crop safety fallback."}
                                    </p>
                                </div>
                                ''',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("Full-frame mode active: background noise (70–80%) included in inference.")

                    with col_results:
                        st.subheader("📊 Classification Result")

                        st.markdown(
                            f'''
                            <div class="result-badge {badge_class}">
                                {icon} {predicted_class.upper()}
                            </div>
                            <div class="conf-track">
                                <div class="conf-fill" style="--target-width: {confidence:.1f}%;"></div>
                            </div>
                            <p style="color:#858ea1; margin-top:6px; font-weight:600;">
                                Prediction Confidence: {confidence:.1f}%
                            </p>
                            ''',
                            unsafe_allow_html=True
                        )

                        # Gauge chart for confidence
                        st.plotly_chart(
                            make_gauge(confidence, predicted_class),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                        # Bar chart comparing both class probabilities
                        st.markdown("<p style='color:#858ea1; margin-bottom:0; font-size:0.9rem;'>Class Probabilities</p>", unsafe_allow_html=True)
                        st.plotly_chart(
                            make_probability_bar(preds),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

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
import torch

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DeepFake Face Classification",
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

# Sidebar menu
menu = st.sidebar.selectbox("Navigate", ["Home", "Register", "Login"])

# Home page
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

    /* Fade-in for the whole main block */
    section.main > div {
        animation: fadeInUp 0.7s ease-out;
    }
    @keyframes fadeInUp {
        0%   { opacity: 0; transform: translateY(18px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Typography */
    h1, h2, h3 {
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

    /* Buttons */
    .stButton > button {
        background-color: #00d2ff;
        color: #0b0e14;
        font-weight: bold;
        border-radius: 4px;
        border: none;
        transition: all 0.25s ease;
        box-shadow: 0 0 0 rgba(0, 210, 255, 0);
    }
    .stButton > button:hover {
        background-color: #00a8cc;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 6px 18px rgba(0, 210, 255, 0.35);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Text inputs get a subtle focus glow */
    .stTextInput > div > div > input {
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 2px rgba(0, 210, 255, 0.35);
    }

    /* Feature Cards */
    .feature-card {
        background-color: #191c22;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid rgba(133, 142, 161, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 24px rgba(0, 210, 255, 0.18);
        border-color: rgba(0, 210, 255, 0.5);
    }
    .feature-icon {
        font-size: 2rem;
        color: #00d2ff;
        margin-bottom: 12px;
        display: inline-block;
        animation: floatY 3s ease-in-out infinite;
    }

    /* Stagger the four home cards */
    div[data-testid="column"]:nth-of-type(1) .feature-card { animation-delay: 0.05s; }
    div[data-testid="column"]:nth-of-type(2) .feature-card { animation-delay: 0.15s; }
    div[data-testid="column"]:nth-of-type(3) .feature-card { animation-delay: 0.25s; }
    div[data-testid="column"]:nth-of-type(4) .feature-card { animation-delay: 0.35s; }

    /* Result badges */
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
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.7); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Confidence bar */
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
    @keyframes fillBar {
        to { width: var(--target-width); }
    }

    /* Status Bar */
    .status-bar {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #858ea1;
        padding-top: 20px;
        margin-top: 40px;
        border-top: 1px dashed rgba(133, 142, 161, 0.25);
        animation: fadeInUp 0.8s ease-out;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00e68c;
        margin-right: 6px;
        box-shadow: 0 0 8px #00e68c;
        animation: pulseDot 1.6s ease-in-out infinite;
    }
    @keyframes pulseDot {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }

    /* Floating Image - Fixed Corner Overlay */
    .floating-img {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 120px;
        z-index: 9999;
        animation: floatY 3s ease-in-out infinite;
        filter: drop-shadow(0 0 12px rgba(0, 210, 255, 0.4));
    }

    /* Floating Image - Inline Bobbing Effect */
    .floating-inline {
        animation: floatY 3s ease-in-out infinite;
        border-radius: 8px;
        transition: filter 0.3s ease;
        filter: drop-shadow(0 0 10px rgba(0, 210, 255, 0.35));
    }
    .floating-inline:hover {
        filter: drop-shadow(0 0 18px rgba(0, 210, 255, 0.6));
    }

    /* Shared bobbing keyframes */
    @keyframes floatY {
        0%   { transform: translatey(0px); }
        50%  { transform: translatey(-15px); }
        100% { transform: translatey(0px); }
    }

    /* Uploaded / analyzed image gets a nice frame */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid rgba(0, 210, 255, 0.25);
        transition: box-shadow 0.3s ease;
    }
    div[data-testid="stImage"] img:hover {
        box-shadow: 0 0 24px rgba(0, 210, 255, 0.3);
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------------- HOME PAGE ----------------

def get_base64(path):
    if not os.path.exists(path):
        st.error(f"Image not found at: {path}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_gauge(confidence, predicted_class):
    """Speedometer-style gauge showing model confidence."""
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
        height=250,
        font={"color": "#e1e2e4"},
    )
    return fig


def make_probability_bar(preds):
    """Horizontal bar chart comparing Real vs Fake probability."""
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
        xaxis=dict(range=[0, 100], showgrid=False, color="#858ea1", ticksuffix="%"),
        yaxis=dict(showgrid=False, color="#e1e2e4"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        font={"color": "#e1e2e4"},
    )
    return fig


if menu == 'Home':

    st.title('DeepFake Face Classification')

    st.markdown(
        'DETECT WHETHER A FACE IMAGE IS REAL OR DEEPFAKE USING ADVANCED DEEP LEARNING TECHNIQUES'
    )

    img_b64 = get_base64("image2.jpg")

    if img_b64:
        # Centered floating image
        st.markdown(
            f'''
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_b64}" class="floating-inline" width="200">
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
                    <p><small>Built with CNNs for accurate DeepFake detection</small></p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                '''
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h4>Image Analysis</h4>
                    <p><small>Advanced preprocessing and feature extraction</small></p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                '''
                <div class="feature-card">
                    <div class="feature-icon">🛡️</div>
                    <h4>High Accuracy</h4>
                    <p><small>Trained on diverse datasets for reliable classification</small></p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                '''
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h4>Real-time Prediction</h4>
                    <p><small>Optimized pipeline for instant results</small></p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.markdown(
            '''
            <div class="status-bar">
                <span class="status-dot"></span>System online — model ready for inference
            </div>
            ''',
            unsafe_allow_html=True
        )

# Register page
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

# ---------------- LOGIN (single flow, no duplicate auth) ----------------
if menu == "Login":

    if not st.session_state.get('logged_in'):

        # Center the login form using columns
        col1, col2, col3 = st.columns([1, 1.2, 1])

        with col2:
            st.markdown("### User/Admin Login")
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):

                with st.spinner('Verifying credentials...'):
                    time.sleep(0.4)

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

        # ---- User dashboard ----
        else:
            if st.session_state.pop('just_logged_in', False):
                st.success("Logged in as User!")
            st.title("User Dashboard")

            # Download model if not exists
            if not os.path.exists("Deep_model.keras"):
                import gdown

                with st.spinner('Downloading model — first run only...'):
                    gdown.download(
                        id="1MvQFRlMsv6BJh94y_7vpNMnJ_YJqI_PK",
                        output="Deep_model.keras",
                        quiet=False
                    )

            # Load the trained model
            with st.spinner('Loading model into memory...'):
                model = keras.models.load_model("Deep_model.keras")

            # Streamlit App
            st.title("DeepFake Image Detection")
            st.write("Upload an image to predict whether it is Real or Fake.")

            # Upload image
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Open image with PIL
                image = Image.open(uploaded_file).convert("RGB")

                # Display image and results side-by-side once we have a prediction
                img_col, result_col = st.columns([1, 1.2])

                with img_col:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                # Read image for OpenCV
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Check if image is loaded correctly
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, (64, 64))
                    img = img.astype("float32") / 255.0

                    # Animated progress bar while the model runs
                    progress_text = "Analyzing facial features..."
                    progress_bar = st.progress(0, text=progress_text)
                    for pct in range(0, 90, 15):
                        time.sleep(0.05)
                        progress_bar.progress(pct, text=progress_text)

                    # Prediction
                    preds = model.predict(img.reshape(1, 64, 64, 3))
                    prd = np.argmax(preds, axis=1)[0]
                    confidence = float(np.max(preds)) * 100

                    progress_bar.progress(100, text="Done")
                    time.sleep(0.15)
                    progress_bar.empty()

                    # Class names
                    classes = ["real", "fake"]
                    predicted_class = classes[prd]

                    # Show result
                    badge_class = "result-real" if predicted_class == "real" else "result-fake"
                    icon = "✅" if predicted_class == "real" else "⚠️"

                    with result_col:
                        st.markdown(
                            f'''
                            <div class="result-badge {badge_class}">
                                {icon} {predicted_class}
                            </div>
                            <div class="conf-track">
                                <div class="conf-fill" style="--target-width: {confidence:.1f}%;"></div>
                            </div>
                            <p style="color:#858ea1; margin-top:6px;">
                                Confidence: {confidence:.1f}%
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
                        st.markdown("<p style='color:#858ea1; margin-bottom:0;'>Class probabilities</p>", unsafe_allow_html=True)
                        st.plotly_chart(
                            make_probability_bar(preds),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )


# ---------------- ROI PIPELINE ----------------

import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepFakeROIPipeline:

    def __init__(self, margin=0.20, target_size=(224, 224), device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.margin = margin
        self.target_size = target_size
        self.detector = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)

    def extract_roi(self, image_bgr):

        if image_bgr is None:
            return None

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image_bgr.shape

        # Detect bounding boxes
        boxes, _ = self.detector.detect(img_rgb)

        if boxes is None:
            # Safety Fallback: Centered Crop if face detection fails
            min_dim = min(h_img, w_img)
            sx, sy = (w_img - min_dim) // 2, (h_img - min_dim) // 2
            crop = image_bgr[sy:sy+min_dim, sx:sx+min_dim]
            return cv2.resize(crop, self.target_size)

        x1, y1, x2, y2 = boxes[0].astype(int)
        w, h = x2 - x1, y2 - y1

        # Expand bounding box by configured margin
        dx, dy = int(w * self.margin), int(h * self.margin)
        x1_exp = max(0, x1 - dx)
        y1_exp = max(0, y1 - dy)
        x2_exp = min(w_img, x2 + dx)
        y2_exp = min(h_img, y2 + dy)

        roi = image_bgr[y1_exp:y2_exp, x1_exp:x2_exp]
        return cv2.resize(roi, self.target_size)


# PyTorch Dataset Integration

class DeepFakeDataset(Dataset):

    def __init__(self, image_paths, labels, pipeline, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.pipeline = pipeline
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        # Dynamic ROI extraction
        roi = self.pipeline.extract_roi(img)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=roi)
            roi = augmented['image']

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return roi, label

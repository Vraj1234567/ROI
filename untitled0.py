"""
DeepFake Detection API — ROI Pipeline Backend

Serves the trained Keras model behind a small HTTP API so the interactive
HTML frontend can get real predictions instead of the in-browser mock.

Endpoints
---------
POST /predict
    multipart/form-data:
        image   - the uploaded image file (jpg/png)
        mode    - "roi" or "full"          (default: "roi")
        margin  - integer percent 15-30    (default: 20)

    Returns JSON shaped to match what the frontend's runPipeline()/classify()
    already produce, e.g.:

    {
      "meta": {
        "detected": true,
        "confidence": 97.4,
        "rawBox": [120, 84, 340, 360],
        "expBox": [92, 56, 368, 388],
        "margin": 0.2,
        "noise": 71.3,
        "detector": "MTCNN (CPU)",
        "fallback": false
      },
      "cls": {
        "label": "fake",
        "confidence": 88.2,
        "probReal": 11.8,
        "probFake": 88.2
      }
    }

Run
---
    pip install flask flask-cors opencv-python-headless numpy tensorflow pillow
    python app.py
    # -> http://localhost:5000/predict
"""

import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # allow the HTML page (served from a different origin) to call this API

MODEL_PATH = "Deep_model.keras"
CLASSES = ["real", "fake"]


# ---------------------------------------------------------------------------
# ROI Pipeline (same logic as the original Streamlit app's
# DeepFakeROIPipeline: detect -> expand margin -> clamp -> fallback)
# ---------------------------------------------------------------------------
class DeepFakeROIPipeline:
    def __init__(self, margin=0.20, target_size=(64, 64)):
        self.margin = margin
        self.target_size = target_size
        self.detector_type = "None"
        self.detector = None

        try:
            import torch
            from facenet_pytorch import MTCNN

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detector = MTCNN(
                keep_all=False, select_largest=True, post_process=False, device=device
            )
            self.detector_type = f"MTCNN ({device.upper()})"
        except Exception:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.detector = cv2.CascadeClassifier(cascade_path)
                    self.detector_type = "OpenCV HaarCascade"
            except Exception:
                self.detector = None
                self.detector_type = "Centered Crop Fallback"

    def detect_face(self, image_bgr):
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
                faces = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    x, y, w, h = faces[0]
                    return (x, y, x + w, y + h), 0.95
            except Exception:
                pass

        return None, 0.0

    def extract_roi(self, image_bgr, margin=None):
        if image_bgr is None:
            return None, {}

        if margin is None:
            margin = self.margin

        h_img, w_img = image_bgr.shape[:2]
        total_pixels = h_img * w_img

        box, conf = self.detect_face(image_bgr)

        if box is None:
            min_dim = min(h_img, w_img)
            sx, sy = (w_img - min_dim) // 2, (h_img - min_dim) // 2
            crop = image_bgr[sy : sy + min_dim, sx : sx + min_dim]
            roi = cv2.resize(crop, self.target_size)

            meta = {
                "detected": False,
                "confidence": 0.0,
                "rawBox": None,
                "expBox": [sx, sy, sx + min_dim, sy + min_dim],
                "margin": 0.0,
                "noise": round((1.0 - (min_dim * min_dim) / total_pixels) * 100, 1),
                "detector": self.detector_type,
                "fallback": True,
            }
            return roi, meta

        x1, y1, x2, y2 = box
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        dx, dy = int(w * margin), int(h * margin)
        x1_exp = max(0, x1 - dx)
        y1_exp = max(0, y1 - dy)
        x2_exp = min(w_img, x2 + dx)
        y2_exp = min(h_img, y2 + dy)

        roi = image_bgr[y1_exp:y2_exp, x1_exp:x2_exp]
        roi_resized = cv2.resize(roi, self.target_size)

        roi_area = (x2_exp - x1_exp) * (y2_exp - y1_exp)
        noise_reduced = max(0.0, round((1.0 - (roi_area / total_pixels)) * 100, 1))

        meta = {
            "detected": True,
            "confidence": round(conf * 100, 1),
            "rawBox": [int(x1), int(y1), int(x2), int(y2)],
            "expBox": [int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)],
            "margin": margin,
            "noise": noise_reduced,
            "detector": self.detector_type,
            "fallback": False,
        }

        return roi_resized, meta


# ---------------------------------------------------------------------------
# Model + pipeline are loaded once at startup
# ---------------------------------------------------------------------------
_pipeline = None
_model = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DeepFakeROIPipeline(margin=0.20, target_size=(64, 64))
    return _pipeline


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Place the trained model file next to app.py."
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name must be 'image')."}), 400

    file = request.files["image"]
    mode = request.form.get("mode", "roi")
    margin_pct = int(request.form.get("margin", 20))
    margin = max(0.15, min(0.30, margin_pct / 100.0))

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if mode == "roi":
        pipeline = get_pipeline()
        roi_resized, meta = pipeline.extract_roi(img_bgr, margin=margin)
        input_tensor_bgr = roi_resized
    else:
        input_tensor_bgr = cv2.resize(img_bgr, (64, 64))
        meta = {
            "detected": False,
            "confidence": 0.0,
            "rawBox": None,
            "expBox": None,
            "margin": 0.0,
            "noise": 0.0,
            "detector": "None (Full Frame)",
            "fallback": False,
        }

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    processed = input_tensor_bgr.astype("float32") / 255.0
    preds = model.predict(processed.reshape(1, 64, 64, 3), verbose=0)

    prob_real = float(preds[0][0]) * 100
    prob_fake = float(preds[0][1]) * 100
    predicted_idx = int(np.argmax(preds, axis=1)[0])
    label = CLASSES[predicted_idx]
    confidence = float(np.max(preds)) * 100

    return jsonify(
        {
            "meta": meta,
            "cls": {
                "label": label,
                "confidence": round(confidence, 1),
                "probReal": round(prob_real, 1),
                "probFake": round(prob_fake, 1),
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

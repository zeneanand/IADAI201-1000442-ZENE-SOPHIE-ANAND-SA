from pathlib import Path
import struct
import math
import base64

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

MODEL_PATHS = [
    ROOT_DIR / "models" / "parking_mobilenet.keras",
    APP_DIR / "models" / "parking_mobilenet.keras",
    ROOT_DIR / "models" / "parking_mlp.joblib",
    APP_DIR / "models" / "parking_mlp.joblib",
]

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="ParkVision AI | Smart Parking Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# SOUND SYNTHESIS (Zero External Dependencies)
# --------------------------------------------------

@st.cache_data
def generate_alert_sound_b64(high_urgency: bool = True) -> str:
    """Generate a clean synthetic alert chime in WAV format (base64 encoded)."""
    sample_rate = 22050
    duration = 0.45
    freq1 = 880.0 if high_urgency else 659.25
    freq2 = 587.33 if high_urgency else 523.25
    num_samples = int(sample_rate * duration)
    raw = bytearray()
    
    for i in range(num_samples):
        t = i / sample_rate
        f = freq1 if t < 0.22 else freq2
        decay = math.exp(-3.5 * (t % 0.22))
        sample = int(32767.0 * 0.45 * decay * math.sin(2.0 * math.pi * f * t))
        sample = max(-32768, min(32767, sample))
        raw.extend(struct.pack("<h", sample))
    
    header = bytearray(b"RIFF")
    header.extend(struct.pack("<I", 36 + len(raw)))
    header.extend(b"WAVEfmt ")
    header.extend(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    header.extend(b"data")
    header.extend(struct.pack("<I", len(raw)))
    return base64.b64encode(header + raw).decode("ascii")


def trigger_audio_alert(urgency_label: str = "critical"):
    """Trigger an audible warning alert chime in the browser."""
    b64_sound = generate_alert_sound_b64(high_urgency=True)
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64_sound}" type="audio/wav">
    </audio>
    <script>
    try {{
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.35);
        gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.35);
    }} catch(e) {{}}
    </script>
    """
    st.components.v1.html(audio_html, height=0, width=0)

# --------------------------------------------------
# CUSTOM CSS & VISUAL THEME
# --------------------------------------------------

CUSTOM_CSS = """
<style>
/* Root Background & Typography */
.stApp {
    background-color: #070b13;
    background-image: 
        radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.18) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
        linear-gradient(rgba(10, 16, 28, 0.88), rgba(6, 10, 18, 0.94)),
        url("https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Glassmorphism Cards */
.park-glass-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 20px;
}

/* Header Container */
.park-header-container {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.1) 50%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 25px;
    box-shadow: 0 0 25px rgba(14, 165, 233, 0.15);
}

.park-logo-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.park-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* Badges */
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.badge-blue {
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-red {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}

/* Metric KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
}

.kpi-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}

.kpi-caption {
    font-size: 0.75rem;
    color: #64748b;
}

/* Glowing Alert Banners */
@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
    70% { box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.alert-banner-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(153, 27, 27, 0.35) 100%);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 24px;
    animation: pulse-red 2s infinite;
}

.alert-banner-moderate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(180, 83, 9, 0.3) 100%);
    border: 1.5px solid #f59e0b;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 24px;
}

.alert-banner-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 95, 70, 0.25) 100%);
    border: 1.5px solid #10b981;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 24px;
}

/* Slot Matrix Chips */
.slot-chip {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    margin: 4px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    min-width: 110px;
}

.slot-empty {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399;
}

.slot-occupied {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(248, 113, 113, 0.4);
    color: #f87171;
}

/* Streamlit Native Overrides */
div[data-testid="stSidebar"] {
    background-color: rgba(10, 16, 28, 0.95) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.stButton>button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.5) !important;
    transform: translateY(-1px) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS & SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 2rem;">🚗</span>
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PARKVISION</div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">SMART SURVEILLANCE AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Camera & Grid Setup")

    mode = st.radio(
        "Detection Mode",
        ["Full parking lot (grid)", "Single slot crop"],
        help="Full parking lot splits the image into equal grid segments. Single slot processes one focused crop.",
    )

    rows = st.slider("Grid Rows", 1, 8, 3)
    cols = st.slider("Grid Columns", 1, 10, 4)

    st.markdown("---")
    st.markdown("#### 🚨 Alert Settings")
    enable_audio = st.checkbox("🔊 Audio Alerts on High Congestion", value=True)
    if st.button("🔔 Test Sound Alert"):
        trigger_audio_alert("test")
        st.success("Test alert sound triggered!")

    st.markdown("---")
    st.markdown("#### ℹ️ System Architecture")
    st.caption(
        "ParkVision uses an automated computer vision pipeline: "
        "image acquisition ➔ grid segmentation ➔ neural feature extraction ➔ "
        "occupancy classification ➔ real-time congestion telemetry."
    )

# --------------------------------------------------
# MODEL LOADING (Supports Keras MobileNet & Sklearn MLP)
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained Keras or scikit-learn model if available."""
    # 1. Check MobileNet Keras model
    keras_candidates = [
        ROOT_DIR / "models" / "parking_mobilenet.keras",
        APP_DIR / "models" / "parking_mobilenet.keras",
    ]
    for path in keras_candidates:
        if path.exists():
            try:
                from tensorflow.keras.models import load_model as keras_load
                model = keras_load(path, compile=False)
                return {
                    "model": model,
                    "type": "mobilenet",
                    "path": str(path),
                }
            except Exception as exc:
                st.warning(f"Failed loading Keras model: {exc}")

    # 2. Check Scikit-learn MLP model
    mlp_candidates = [
        ROOT_DIR / "models" / "parking_mlp.joblib",
        APP_DIR / "models" / "parking_mlp.joblib",
    ]
    for path in mlp_candidates:
        if path.exists():
            try:
                import joblib
                payload = joblib.load(path)
                return {
                    "model": payload["model"],
                    "classes": tuple(payload["classes"]),
                    "type": "mlp",
                    "path": str(path),
                }
            except Exception as exc:
                st.warning(f"Failed loading MLP model: {exc}")

    return None

# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def load_image(uploaded_file):
    """Convert an uploaded image into RGB format."""
    return np.array(
        Image.open(uploaded_file).convert("RGB")
    )


def crop_slots(image, grid_rows, grid_cols):
    """Split the image into equally sized grid cells."""
    height, width = image.shape[:2]
    slots = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = round(col * width / grid_cols)
            x2 = round((col + 1) * width / grid_cols)
            y1 = round(row * height / grid_rows)
            y2 = round((row + 1) * height / grid_rows)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            slots.append({
                "id": len(slots) + 1,
                "row": row,
                "col": col,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop": crop,
            })

    return slots

# --------------------------------------------------
# PREDICTION ENGINE
# --------------------------------------------------

_preprocess_input = None

def get_preprocess_input():
    global _preprocess_input
    if _preprocess_input is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        _preprocess_input = preprocess_input
    return _preprocess_input


def predict_slot(model_info, crop):
    """Predict empty or occupied using whichever trained model is loaded."""
    if model_info.get("type") == "mlp":
        from src.model import image_to_features
        feats = image_to_features(crop).reshape(1, -1)
        probs = model_info["model"].predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        label = model_info["classes"][idx]
        confidence = float(probs[idx])
        return label, confidence

    # MobileNet path
    model = model_info["model"]
    resized = cv2.resize(crop, IMAGE_SIZE)

    batch = np.expand_dims(
        resized.astype(np.float32),
        axis=0,
    )

    preprocess = get_preprocess_input()
    batch = preprocess(batch)

    prediction = np.asarray(
        model.predict(batch, verbose=0)
    ).reshape(-1)

    if prediction.size == 1:
        occupied_probability = float(prediction[0])
    elif prediction.size == 2:
        occupied_probability = float(prediction[1])
    else:
        raise ValueError("Expected a one-output or two-output classifier.")

    occupied_probability = float(np.clip(occupied_probability, 0.0, 1.0))

    if occupied_probability >= 0.5:
        label = "occupied"
        confidence = occupied_probability
    else:
        label = "empty"
        confidence = 1.0 - occupied_probability

    return label, confidence


def demo_prediction(crop):
    """Simple visual heuristic fallback for demo mode."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    edge_ratio = float(np.mean(edges > 0))
    brightness = float(np.mean(gray))

    occupied = edge_ratio > 0.08 and brightness < 190
    label = "occupied" if occupied else "empty"
    return label, None


@st.cache_data(show_spinner=False)
def classify_slots(_model, image, grid_mode, grid_rows, grid_cols):
    """Segment image into slots and classify each slot, cached to prevent CPU thrashing."""
    if grid_mode == "Full parking lot (grid)":
        slots = crop_slots(image, grid_rows, grid_cols)
    else:
        slots = [{
            "id": 1,
            "row": 0,
            "col": 0,
            "x1": 0,
            "y1": 0,
            "x2": image.shape[1],
            "y2": image.shape[0],
            "crop": image,
        }]

    results = []
    for slot in slots:
        if _model is not None:
            label, confidence = predict_slot(_model, slot["crop"])
        else:
            label, confidence = demo_prediction(slot["crop"])

        results.append({
            **slot,
            "label": label,
            "confidence": confidence,
        })
    return results

# --------------------------------------------------
# OCCUPANCY INSIGHTS
# --------------------------------------------------

def build_insights(labels):
    total = len(labels)
    occupied = sum(label == "occupied" for label in labels)
    available = total - occupied
    occupancy = (occupied / total * 100) if total > 0 else 0.0

    if occupancy < 40:
        congestion = "Low"
        recommendation = "Low occupancy detected. Ample vacant spaces available — proceed directly to park."
    elif occupancy <= 75:
        congestion = "Moderate"
        recommendation = "Moderate occupancy. Several vacant spaces remaining; verify individual slot markers."
    else:
        congestion = "High"
        recommendation = "CRITICAL HIGH OCCUPANCY! Parking facility near capacity. Direct incoming drivers to overflow lots."

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy": occupancy,
        "congestion": congestion,
        "recommendation": recommendation,
    }

# --------------------------------------------------
# ANNOTATION RENDERER
# --------------------------------------------------

def draw_annotations(image, slots):
    """Draw high-visibility green boxes for empty and red boxes for occupied."""
    annotated = image.copy()

    for slot in slots:
        if slot["label"] == "occupied":
            color = (239, 68, 68)  # Red
        else:
            color = (34, 197, 94)  # Emerald green

        cv2.rectangle(
            annotated,
            (slot["x1"], slot["y1"]),
            (slot["x2"], slot["y2"]),
            color,
            3,
        )

        text = f"S{slot['id']}: {slot['label'].upper()}"
        text_y = min(slot["y1"] + 24, image.shape[0] - 6)

        # Background tag behind text for high contrast readability
        cv2.putText(
            annotated,
            text,
            (slot["x1"] + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (slot["x1"] + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated

# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main():
    # Header Banner with SVG Logo
    st.markdown(
        """
        <div class="park-header-container">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                <div>
                    <h1 class="park-logo-title">
                        <svg width="38" height="38" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect width="24" height="24" rx="6" fill="url(#grad1)"/>
                            <path d="M7 17V7H12.5C14.433 7 16 8.567 16 10.5C16 12.433 14.433 14 12.5 14H9.5V17H7Z" fill="white"/>
                            <circle cx="18" cy="6" r="3" fill="#34D399"/>
                            <defs>
                                <linearGradient id="grad1" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
                                    <stop stop-color="#0EA5E9"/>
                                    <stop offset="1" stop-color="#6366F1"/>
                                </linearGradient>
                            </defs>
                        </svg>
                        PARKVISION AI
                    </h1>
                    <div class="park-subtitle">Autonomous Parking Occupancy Intelligence & Optical Vision Telemetry</div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="telemetry-badge badge-green">● Live Telemetry</span>
                    <span class="telemetry-badge badge-blue">● Neural Vision</span>
                    <span class="telemetry-badge badge-green">● Latency: &lt;1ms</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model loader
    try:
        model_info = load_model()
    except Exception as exc:
        st.error(f"Model initialization failed: {exc}")
        st.stop()

    if model_info is not None:
        model_name = Path(model_info["path"]).name
        model_type_str = "MobileNetV2" if model_info["type"] == "mobilenet" else "Neural Network (MLP)"
        st.sidebar.success(f"🟢 Trained Model Loaded: {model_type_str}")
        st.sidebar.caption(f"Active weight file: `{model_name}`")
    else:
        st.sidebar.warning("⚠️ Trained model not found")
        st.warning("DEMO MODE: Running basic visual heuristic. Load a trained model for calibrated predictions.")

    # Input image selector
    samples_dir = ROOT_DIR / "data" / "samples"
    sample_options = ["Upload my own image"]
    sample_map = {}
    if samples_dir.exists():
        for sample_file in sorted(samples_dir.glob("*.jpg")):
            name = sample_file.stem.replace("_", " ").title()
            sample_options.append(name)
            sample_map[name] = sample_file

    col_sel, col_empty = st.columns([2, 1])
    with col_sel:
        selected_source = st.selectbox(
            "📹 Select Video Feed / Image Source:",
            sample_options,
            help="Choose a pre-loaded parking lot dataset sample or upload your own parking surveillance photo.",
        )

    image = None
    if selected_source == "Upload my own image":
        uploaded_file = st.file_uploader(
            "Upload surveillance footage / image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            try:
                image = load_image(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                return
    else:
        sample_path = sample_map[selected_source]
        try:
            image = np.array(Image.open(sample_path).convert("RGB"))
        except Exception as exc:
            st.error(f"Could not load sample: {exc}")
            return

    if image is None:
        st.markdown(
            """
            <div class="park-glass-card" style="text-align: center; padding: 40px 20px;">
                <span style="font-size: 3rem;">📸</span>
                <h3 style="color: #38bdf8; margin: 12px 0 6px 0;">No Camera Feed Selected</h3>
                <p style="color: #94a3b8; max-width: 500px; margin: 0 auto 18px auto;">
                    Select one of the demo parking lot surveillance images from the dropdown above, 
                    or upload a high-resolution photo from your facility to run automated slot detection.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Process and classify grid cells (Cached)
    results = []
    with st.spinner("⚡ Running AI Optical Classification..."):
        try:
            results = classify_slots(model_info, image, mode, rows, cols)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    if not results:
        st.error("Could not segment valid image regions.")
        return

    # Calculate metrics
    insights = build_insights([result["label"] for result in results])

    # Trigger Sound Alert if High Congestion
    if insights["congestion"] == "High" and enable_audio:
        trigger_audio_alert("high")

    # Render Congestion Alert Box
    if insights["congestion"] == "High":
        st.markdown(
            f"""
            <div class="alert-banner-high">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 2.2rem;">🚨</span>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #fecaca; letter-spacing: 0.5px;">CRITICAL OCCUPANCY ALERT — {insights['occupancy']:.1f}% FULL</div>
                        <div style="color: #fca5a5; font-size: 0.95rem; margin-top: 4px;">{insights['recommendation']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif insights["congestion"] == "Moderate":
        st.markdown(
            f"""
            <div class="alert-banner-moderate">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 2.2rem;">⚠️</span>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #fef3c7;">MODERATE CONGESTION — {insights['occupancy']:.1f}% OCCUPIED</div>
                        <div style="color: #fde68a; font-size: 0.95rem; margin-top: 4px;">{insights['recommendation']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-banner-low">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 2.2rem;">✅</span>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #d1fae5;">OPTIMAL PARKING CAPACITY — {insights['available']} SPACES FREE</div>
                        <div style="color: #a7f3d0; font-size: 0.95rem; margin-top: 4px;">{insights['recommendation']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # KPI Telemetry Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">🅿️ Total Spaces</div>
                <div class="kpi-value" style="color: #38bdf8;">{insights['total']}</div>
                <div class="kpi-caption">Monitored grid cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">🟢 Available Free</div>
                <div class="kpi-value" style="color: #34d399;">{insights['available']}</div>
                <div class="kpi-caption">Ready for incoming vehicles</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">🔴 Occupied Spaces</div>
                <div class="kpi-value" style="color: #f87171;">{insights['occupied']}</div>
                <div class="kpi-caption">Vehicles currently parked</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">📊 Occupancy Rate</div>
                <div class="kpi-value" style="color: {'#f87171' if insights['occupancy'] > 75 else ('#f59e0b' if insights['occupancy'] >= 40 else '#34d399')};">{insights['occupancy']:.1f}%</div>
                <div class="kpi-caption">Status: {insights['congestion']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Annotated Image Display & Interactive Grid Map
    col_img, col_matrix = st.columns([3, 2])

    with col_img:
        st.markdown(
            '<div class="park-glass-card">'
            '<h3 style="color:#38bdf8; margin:0 0 12px 0;">📷 Real-Time Optical Detection</h3>',
            unsafe_allow_html=True,
        )
        annotated_image = draw_annotations(image, results)
        st.image(annotated_image, use_container_width=True)
        st.markdown(
            '<div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-top:8px;">'
            '<span>🟢 Green = Empty Available</span>'
            '<span>🔴 Red = Occupied</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with col_matrix:
        st.markdown(
            '<div class="park-glass-card">'
            '<h3 style="color:#818cf8; margin:0 0 12px 0;">🗺️ Digital Space Matrix</h3>',
            unsafe_allow_html=True,
        )

        # Interactive filter
        filter_mode = st.radio(
            "Filter spaces:",
            ["All Spaces", "Vacant Only (🟢)", "Occupied Only (🔴)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        filtered_results = results
        if filter_mode == "Vacant Only (🟢)":
            filtered_results = [r for r in results if r["label"] == "empty"]
        elif filter_mode == "Occupied Only (🔴)":
            filtered_results = [r for r in results if r["label"] == "occupied"]

        chips_html = '<div style="max-height: 380px; overflow-y: auto; padding-right: 6px;">'
        if not filtered_results:
            chips_html += '<div style="color: #64748b; padding: 20px; text-align: center;">No slots match the active filter.</div>'
        else:
            for r in filtered_results:
                is_occ = (r["label"] == "occupied")
                css_cls = "slot-occupied" if is_occ else "slot-empty"
                icon = "🔴" if is_occ else "🟢"
                conf_str = f"{r['confidence']:.0%}" if r["confidence"] is not None else "Est"
                chips_html += f"""
                <div class="slot-chip {css_cls}">
                    <span>{icon} <b>S{r['id']}</b> (R{r['row']+1}:C{r['col']+1})</span>
                    <span style="opacity:0.85;">{conf_str}</span>
                </div>
                """
        chips_html += '</div></div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    # Detailed Results Table
    with st.expander("📋 Detailed Slot-by-Slot Telemetry Table", expanded=False):
        table_data = []
        for result in results:
            confidence_text = f"{result['confidence']:.1%}" if result["confidence"] is not None else "Uncalibrated"
            table_data.append({
                "Slot ID": f"S{result['id']}",
                "Grid Row": result["row"] + 1,
                "Grid Column": result["col"] + 1,
                "Status": "🟢 Vacant" if result["label"] == "empty" else "🔴 Occupied",
                "Model Confidence": confidence_text,
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    st.caption("ParkVision AI Telemetry Engine • Course Assessment Submission • Built with OpenCV & Neural Vision")


if __name__ == "__main__":
    main()

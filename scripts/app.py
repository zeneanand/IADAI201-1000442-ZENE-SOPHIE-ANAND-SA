
from pathlib import Path

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
    page_title="ParkVision",
    page_icon="🚗",
    layout="wide",
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🚗 ParkVision")
    st.write("AI-powered parking space detection")
    st.divider()

    mode = st.radio(
        "Image type",
        ["Full parking lot (grid)", "Single slot crop"],
    )

    rows = st.slider("Grid rows", 1, 8, 3)
    cols = st.slider("Grid columns", 1, 10, 4)

    st.markdown("### How it works")
    st.write(
        "The image is divided into grid cells. "
        "Each cell is classified as empty or occupied."
    )

# --------------------------------------------------
# MODEL LOADING
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
# PREDICTION
# --------------------------------------------------

_preprocess_input = None


def get_preprocess_input():
    global _preprocess_input
    if _preprocess_input is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        _preprocess_input = preprocess_input
    return _preprocess_input


def predict_slot(model_info, crop):
    """
    Predict empty or occupied using whichever trained model is loaded.
    """
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
        # Assumes output order: empty, occupied.
        occupied_probability = float(prediction[1])

    else:
        raise ValueError(
            "Expected a one-output or two-output classifier."
        )

    occupied_probability = float(
        np.clip(occupied_probability, 0.0, 1.0)
    )

    if occupied_probability >= 0.5:
        label = "occupied"
        confidence = occupied_probability
    else:
        label = "empty"
        confidence = 1.0 - occupied_probability

    return label, confidence


def demo_prediction(crop):
    """
    Simple visual heuristic for demo mode only.
    This is NOT a trained or reliable parking classifier.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    edge_ratio = float(np.mean(edges > 0))
    brightness = float(np.mean(gray))

    occupied = edge_ratio > 0.08 and brightness < 190

    label = "occupied" if occupied else "empty"

    # No calibrated confidence is available in demo mode.
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

    occupied = sum(
        label == "occupied" for label in labels
    )

    available = total - occupied

    occupancy = (
        occupied / total * 100
        if total > 0
        else 0.0
    )

    if occupancy < 40:
        congestion = "Low"
        recommendation = (
            "Low estimated occupancy. Check the predicted "
            "empty slots before parking."
        )

    elif occupancy <= 75:
        congestion = "Moderate"
        recommendation = (
            "Moderate estimated occupancy. Some spaces may "
            "be available; check individual predictions."
        )

    else:
        congestion = "High"
        recommendation = (
            "High estimated occupancy. Verify the predictions "
            "before looking for another parking area."
        )

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy": occupancy,
        "congestion": congestion,
        "recommendation": recommendation,
    }


# --------------------------------------------------
# ANNOTATED IMAGE
# --------------------------------------------------

def draw_annotations(image, slots):
    """Draw green boxes for empty and red boxes for occupied."""
    annotated = image.copy()

    for slot in slots:
        if slot["label"] == "occupied":
            color = (230, 50, 50)
        else:
            color = (30, 180, 60)

        cv2.rectangle(
            annotated,
            (slot["x1"], slot["y1"]),
            (slot["x2"], slot["y2"]),
            color,
            3,
        )

        text = f"S{slot['id']}: {slot['label']}"

        text_y = min(
            slot["y1"] + 22,
            image.shape[0] - 5,
        )

        cv2.putText(
            annotated,
            text,
            (slot["x1"] + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main():
    st.title("🚗 ParkVision")
    st.caption(
        "Parking occupancy analysis using computer vision"
    )

    st.markdown(
        "**Input:** Parking-lot image  \n"
        "**Output:** Slot-wise empty or occupied predictions  \n"
        "**Approach:** Grid-based crop classification with MobileNetV2"
    )

    # Load trained model, if available.
    try:
        model_info = load_model()

    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    if model_info is not None:
        model_name = Path(model_info["path"]).name
        model_type_str = "MobileNetV2" if model_info["type"] == "mobilenet" else "Neural Network (MLP)"
        st.sidebar.success(f"Trained model loaded: {model_type_str}")
        st.caption(
            f"Active model: `{model_name}`"
        )

    else:
        st.sidebar.warning("Trained model not found")
        st.warning(
            "DEMO MODE: No trained model was found. "
            "The fallback uses a basic visual heuristic and "
            "is not reliable for real parking detection."
        )

    # Image source selector: upload or choose demo lot.
    samples_dir = ROOT_DIR / "data" / "samples"
    sample_options = ["Upload my own image"]
    sample_map = {}
    if samples_dir.exists():
        for sample_file in sorted(samples_dir.glob("*.jpg")):
            name = sample_file.stem.replace("_", " ").title()
            sample_options.append(name)
            sample_map[name] = sample_file

    selected_source = st.selectbox(
        "Select image source:",
        sample_options,
    )

    image = None
    if selected_source == "Upload my own image":
        uploaded_file = st.file_uploader(
            "Upload a parking image",
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
            st.info(f"Loaded demo lot: **{selected_source}**")
        except Exception as exc:
            st.error(f"Could not load sample: {exc}")
            return

    if image is None:
        st.info("Upload a parking image or select a demo lot above to begin.")

        st.subheader("System overview")
        st.write(
            "1. Upload or select a demo parking image.  \n"
            "2. Divide the image into a grid.  \n"
            "3. Classify each slot.  \n"
            "4. Review occupancy metrics and annotated results."
        )
        return

    st.subheader("Parking image")
    st.image(image, use_container_width=True)

    # Generate and classify grid cells (cached to prevent CPU spikes on UI reruns).
    results = []
    with st.spinner("Analysing parking slots..."):
        try:
            results = classify_slots(model_info, image, mode, rows, cols)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    if not results:
        st.error("Could not create valid image regions.")
        return

    # Calculate insights.
    insights = build_insights([
        result["label"] for result in results
    ])

    st.divider()
    st.subheader("📊 Parking insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total slots",
        insights["total"],
    )

    col2.metric(
        "Occupied",
        insights["occupied"],
    )

    col3.metric(
        "Available",
        insights["available"],
    )

    col4.metric(
        "Occupancy",
        f"{insights['occupancy']:.1f}%",
    )

    st.write(
        f"**Congestion level:** {insights['congestion']}"
    )

    st.info(insights["recommendation"])

    # Annotate parking image.
    annotated_image = draw_annotations(
        image,
        results,
    )

    st.subheader("🅿️ Annotated parking image")

    st.image(
        annotated_image,
        use_container_width=True,
    )

    st.caption(
        "Green = predicted empty | Red = predicted occupied"
    )

    # Detailed table.
    table_data = []

    for result in results:
        if result["confidence"] is None:
            confidence_text = "Not calibrated"
        else:
            confidence_text = (
                f"{result['confidence']:.1%}"
            )

        table_data.append({
            "Slot": f"S{result['id']}",
            "Row": result["row"] + 1,
            "Column": result["col"] + 1,
            "Prediction": result["label"].title(),
            "Confidence": confidence_text,
        })

    st.subheader("Slot-by-slot results")

    st.dataframe(
        pd.DataFrame(table_data),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Predictions depend on image quality, grid alignment, "
        "training data, and model accuracy. Verify results "
        "before using them for real parking decisions."
    )


if __name__ == "__main__":
    main()

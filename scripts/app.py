
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
        "The parking image is divided into slots. "
        "Each slot is classified as empty or occupied."
    )

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained Keras model if it exists."""
    for path in MODEL_PATHS:
        if path.exists():
            try:
                from tensorflow.keras.models import load_model
                return load_model(path, compile=False), str(path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load model at {path}: {exc}"
                ) from exc

    return None, None


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def load_image(uploaded_file):
    """Convert uploaded image to an RGB NumPy array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def crop_slots(image, grid_rows, grid_cols):
    """Divide an image into equal-sized grid cells."""
    height, width = image.shape[:2]
    slots = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = round(col * width / grid_cols)
            x2 = round((col + 1) * width / grid_cols)
            y1 = round(row * height / grid_rows)
            y2 = round((row + 1) * height / grid_rows)

            crop = image[y1:y2, x1:x2]

            if crop.size:
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

def predict_slot(model, crop):
    """
    Predict a slot using the trained model.

    Expected output:
    - Two-class model: index 0 = empty, index 1 = occupied
    - One-output model: sigmoid probability for occupied
    """
    resized = cv2.resize(crop, IMAGE_SIZE)
    batch = np.expand_dims(resized.astype("float32"), axis=0)

    # Standard MobileNetV2 preprocessing.
    # The model must have been trained with compatible preprocessing.
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    batch = preprocess_input(batch)

    prediction = np.asarray(model.predict(batch, verbose=0))
    prediction = prediction.reshape(-1)

    if prediction.size == 1:
        occupied_probability = float(prediction[0])
        occupied_probability = float(
            np.clip(occupied_probability, 0, 1)
        )
    elif prediction.size == 2:
        # Assumes model class order is [empty, occupied].
        occupied_probability = float(prediction[1])
    else:
        raise ValueError(
            "The model must output one sigmoid probability "
            "or two class probabilities."
        )

    label = (
        "occupied"
        if occupied_probability >= 0.5
        else "empty"
    )

    confidence = (
        occupied_probability
        if label == "occupied"
        else 1 - occupied_probability
    )

    return label, confidence


def demo_prediction(crop):
    """
    Demonstration-only heuristic when no trained model exists.
    This is NOT a reliable parking classifier.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    edge_ratio = float(np.mean(edges > 0))
    brightness = float(np.mean(gray))

    # Arbitrary visual heuristic; do not use for real decisions.
    occupied = edge_ratio > 0.08 and brightness < 190

    label = "occupied" if occupied else "empty"
    return label, 0.0


# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------

def build_insights(labels):
    total = len(labels)
    occupied = sum(label == "occupied" for label in labels)
    available = total - occupied
    occupancy = (occupied / total * 100) if total else 0.0

    if occupancy < 40:
        congestion = "Low"
        recommendation = "Slots may be available. Check the displayed predictions."
    elif occupancy <= 75:
        congestion = "Moderate"
        recommendation = "Parking availability may be limited. Check individual slots."
    else:
        congestion = "High"
        recommendation = "The lot may be busy. Verify the predictions before choosing another area."

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy": occupancy,
        "congestion": congestion,
        "recommendation": recommendation,
    }


def draw_annotations(image, slots):
    """Draw green boxes for empty slots and red for occupied slots."""
    annotated = image.copy()

    for slot in slots:
        color = (
            (220, 50, 50)
            if slot["label"] == "occupied"
            else (30, 180, 60)
        )

        cv2.rectangle(
            annotated,
            (slot["x1"], slot["y1"]),
            (slot["x2"], slot["y2"]),
            color,
            3,
        )

        cv2.putText(
            annotated,
            f"S{slot['id']}: {slot['label']}",
            (slot["x1"] + 5, min(slot["y1"] + 22, image.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():
    st.title("🚗 ParkVision")
    st.caption(
        "Parking occupancy analysis using image classification"
    )

    st.markdown(
        "**Input:** Parking-lot image  \n"
        "**Output:** Slot-wise empty or occupied predictions  \n"
        "**Approach:** Grid-based crop classification with MobileNetV2"
    )

    try:
        model, model_path = load_model()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    if model is not None:
        st.sidebar.success("Trained model loaded")
        st.caption(f"Model: `{Path(model_path).name}`")
    else:
        st.sidebar.warning("Trained model not found")
        st.warning(
            "DEMO MODE: No trained model was found. "
            "The visual heuristic is not a validated AI model and "
            "its predictions are not reliable. Add your trained "
            "models/parking_mobilenet.keras file for actual inference."
        )

    uploaded_file = st.file_uploader(
        "Upload a parking image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a parking-lot image to begin.")
        st.markdown("### System overview")
        st.write(
            "1. Load the image.  \n"
            "2. Divide the parking area into a grid.  \n"
            "3. Classify each crop.  \n"
            "4. Display slot labels and occupancy insights."
        )
        return

    try:
        image = load_image(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return

    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    if mode == "Full parking lot (grid)":
        slots = crop_slots(image, rows, cols)
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

    if not slots:
        st.error("No valid image regions could be created.")
        return

    results = []

    with st.spinner("Classifying parking slots..."):
        for slot in slots:
            try:
                if model is not None:
                    label, confidence = predict_slot(
                        model, slot["crop"]
                    )
                else:
                    label, confidence = demo_prediction(
                        slot["crop"]
                    )

                results.append({
                    **slot,
                    "label": label,
                    "confidence": confidence,
                })

            except Exception as exc:
                st.error(
                    f"Prediction failed for slot {slot['id']}: {exc}"
                )
                st.stop()

    insights = build_insights(
        [item["label"] for item in results]
    )

    st.divider()
    st.subheader("Parking insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total slots", insights["total"])
    c2.metric("Occupied", insights["occupied"])
    c3.metric("Available", insights["available"])
    c4.metric("Occupancy", f"{insights['occupancy']:.1f}%")

    st.write(f"**Congestion:** {insights['congestion']}")
    st.info(insights["recommendation"])

    if model is not None:
        annotated = draw_annotations(image, results)
        st.subheader("Annotated parking lot")
        st.image(annotated, use_container_width=True)
        st.caption("Green = predicted empty · Red = predicted occupied")
    else:
        st.subheader("Grid preview — demo heuristic")
        st.caption(
            "These labels are illustrative only, not validated predictions."
        )
        preview = draw_annotations(image, results)
        st.image(preview, use_container_width=True)

    table = pd.DataFrame([
        {
            "Slot": f"S{item['id']}",
            "Row": item["row"] + 1,
            "Column": item["col"] + 1,
            "Prediction": item["label"].title(),
            "Confidence": (
                f"{item['confidence']:.1%}"
                if model is not None
                else "Not calibrated"
            ),
        }
        for item in results
    ])

    st.subheader("Slot-by-slot results")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide"
)

MODEL_PATH = Path("models/parking_model.keras")
IMG_SIZE = (224, 224)

st.title("🅿️ ParkVision AI")
st.caption("AI-powered smart parking occupancy dashboard")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.sidebar.header("Parking Configuration")
rows = st.sidebar.number_input("Number of rows", 1, 20, 3)
cols = st.sidebar.number_input("Slots per row", 1, 20, 6)
threshold = st.sidebar.slider(
    "Occupied confidence threshold", 0.1, 0.9, 0.5
)

uploaded_file = st.file_uploader(
    "Upload a parking-lot image",
    type=["jpg", "jpeg", "png"]
)

def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

def split_slots(image, rows, cols):
    width, height = image.size
    crops, boxes = [], []

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * width / cols)
            x2 = int((c + 1) * width / cols)
            y1 = int(r * height / rows)
            y2 = int((r + 1) * height / rows)

            crops.append(image.crop((x1, y1, x2, y2)))
            boxes.append((x1, y1, x2, y2))

    return crops, boxes

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded parking image")
    st.image(image, use_container_width=True)

    crops, boxes = split_slots(image, int(rows), int(cols))
    batch = np.stack([preprocess(crop) for crop in crops])

    with st.spinner("Analysing parking slots..."):
        probabilities = model.predict(batch, verbose=0).reshape(-1)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    results = []

    for i, (prob, box) in enumerate(
        zip(probabilities, boxes), start=1
    ):
        occupied = float(prob) >= threshold
        status = "Occupied" if occupied else "Empty"
        color = (220, 40, 40) if occupied else (20, 160, 70)

        draw.rectangle(box, outline=color, width=4)
        draw.text(
            (box[0] + 5, box[1] + 5),
            f"Slot {i}: {status} ({prob:.0%})",
            fill=color
        )

        results.append({
            "Slot": i,
            "Status": status,
            "Occupied probability": round(float(prob), 4)
        })

    total = len(results)
    occupied_count = sum(
        item["Status"] == "Occupied" for item in results
    )
    available_count = total - occupied_count
    occupancy = occupied_count / total * 100

    if occupancy < 40:
        congestion = "Low"
        recommendation = "Slots available — proceed."
    elif occupancy <= 75:
        congestion = "Moderate"
        recommendation = "Parking is moderately occupied."
    else:
        congestion = "High"
        recommendation = "Parking highly occupied — try another area."

    if available_count == 0:
        recommendation = "Parking appears full — try another area."

    st.subheader("Slot-wise predictions")
    st.image(annotated, use_container_width=True)
    st.caption("Green = Empty | Red = Occupied")

    a, b, c, d = st.columns(4)
    a.metric("Total slots", total)
    b.metric("Occupied", occupied_count)
    c.metric("Available", available_count)
    d.metric("Occupancy", f"{occupancy:.1f}%")

    st.subheader("Parking insights")
    st.write(f"**Congestion level:** {congestion}")
    st.info(recommendation)

    df = pd.DataFrame(results)
    st.subheader("Slot results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV results",
        df.to_csv(index=False).encode("utf-8"),
        "parkvision_results.csv",
        "text/csv"
    )


import streamlit as st

with st.sidebar:
    st.title("ParkVision")
    st.write("Parking Space Detection")

**Input:** parking lot image  
**Output:** slot-wise Occupied / Empty
**Approach:** crop-based classification  
Each slot is cropped from a **fixed grid** and classified as Occupied or Empty.
**Model:** MobileNetV2 when `models/parking_mobilenet.keras` is present; otherwise the bundled neural-net classifier.
**Why not YOLO here?**  
YOLO fits irregular layouts. Painted campus/mall bays work as a grid, which is lighter to train and deploy.
            """
        )
        mode = st.radio("Image type", ["Full parking lot (grid)", "Single slot crop"])
        rows = st.slider("Grid rows", 1, 8, 3)
        cols = st.slider("Grid columns", 1, 10, 4)
        st.markdown("#### Challenges handled")
        st.write("Shadows, wet/rain streaks, brightness changes, and partial cars are covered by augmentation and synthetic weather variants.")
    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    backend = "MobileNetV2" if getattr(model, "backend", "") == "mobilenet" else "Neural net classifier"
    st.sidebar.success(f"Loaded: {backend}")
    uploaded = st.file_uploader("Upload a parking image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    sample_name = None
    if SAMPLES.exists():
        sample_files = sorted(p.name for p in SAMPLES.glob("*.jpg"))
        if sample_files:
            sample_name = st.selectbox("Or try a demo lot", ["—"] + sample_files)
    image = None
    if uploaded is not None:
        image = load_image(uploaded)
    elif sample_name and sample_name != "—":
        image = load_image(SAMPLES / sample_name)
    if image is None:
        st.info("Upload an image or pick a demo parking lot to see occupancy insights.")
        st.markdown(
            """
### System design
| Piece | Choice |
| --- | --- |
| Problem | Smart-city parking occupancy from camera images |
| Layout | Fixed painted slots (grid). Irregular lots need YOLO / polygon maps |
| Model | MobileNetV2 transfer learning, 224×224 slot crops |
| Split | 70% train / 15% validation / 15% test |
| Insights | Occupancy % → Low (<40%) / Moderate (40–75%) / High (>75%) |
            """
        )
        return
    if mode.startswith("Full"):
        slots = predict_parking_lot(model, image, rows, cols)
        annotated = draw_annotations(image, slots)
        insights = build_insights([s["label"] for s in slots])
        left, right = st.columns([1.35, 1])
        with left:
            st.subheader("Annotated parking lot")
            st.image(annotated, use_container_width=True)
            st.caption("Green = empty · Red = occupied")
        with right:
            st.subheader("Parking insights")
            metric_cards(insights)
            st.markdown(
                f"**Congestion:** `{insights.congestion}`"
            )
            st.markdown(
                f"<div style='padding:14px 16px;border-radius:12px;background:{insights.color}22;"
                f"border:1px solid {insights.color};font-size:1.05rem'><b>{insights.recommendation}</b></div>",
                unsafe_allow_html=True,
            )
            table = pd.DataFrame(
                [
                    {
                        "Slot": f"S{s['id']}",
                        "Row": s["row"] + 1,
                        "Col": s["col"] + 1,
                        "Status": s["label"].title(),
                        "Confidence": s["confidence"],
                    }
                    for s in slots
                ]
            )
            st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        label, conf = predict_slot(model, image)
        insights = build_insights([label])
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded slot crop", use_container_width=True)
        with right:
            st.metric("Prediction", label.title())
            st.metric("Confidence", f"{conf:.1%}")
            metric_cards(insights)
            st.write(insights.recommendation)
if __name__ == "__main__":
    main()

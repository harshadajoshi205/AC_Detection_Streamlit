
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AC Detection",
    page_icon="❄️",
    layout="wide"
)

st.title("Indoor & Outdoor AC Detection")
st.write("AI-powered AC detection using a custom YOLO26n model.")

@st.cache_resource
def load_model():
    return YOLO("weights.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload an AC image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting AC units..."):
        results = model.predict(
            source=np.array(image),
            conf=confidence,
            device="cpu",
            verbose=False
        )

    result = results[0]

    with col2:
        st.subheader("Detection Result")
        annotated_image = result.plot()
        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

    st.subheader("Detected Objects")

    if len(result.boxes) == 0:
        st.warning("No AC units detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            class_name = result.names[class_id]

            st.write(
                f"**{class_name}** — Confidence: {score:.1%}"
            )

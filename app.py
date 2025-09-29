import streamlit as st
from PIL import Image
import numpy as np

# Defensive import untuk ultralytics dan cv2
try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Ultralytics not found. Check your environment or requirements.txt.")
    st.stop()

# Load model safely
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

st.title("YOLOv8 Object Detection 🚀")
st.markdown("Upload an image and let the model detect objects in it.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run prediction
        results = model.predict(image, conf=0.25)
        output_array = results[0].plot()  # returns numpy array with boxes

        # Convert numpy array to PIL image
        output_image = Image.fromarray(output_array.astype(np.uint8))
        st.image(output_image, caption="Detected Objects", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

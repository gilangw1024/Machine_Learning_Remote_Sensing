import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("YOLOv8 Object Detection 🚀")

# Load model
model = YOLO("best.pt")  # atau "yolov8n.pt"

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(image, conf=0.25)
    boxes = results[0].plot()  # returns numpy array with boxes

    st.image(boxes, caption="Detected Objects", use_column_width=True)

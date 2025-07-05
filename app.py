import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
from PIL import Image

# Download model if not present
def download_model():
    model_path = "models/cnn_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        file_id = "1aJ6m3IJ22L9a-Faawok3C-1AbhU4-AmW"  # Your Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    return model_path

# Load model
model_path = download_model()
model = load_model(model_path)

# Image preprocessing
def preprocess_image(img):
    img = img.resize((128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.set_page_config(page_title="Forgery Detection App", layout="centered")
st.title("📝 Forgery Detection in ID Cards and Signatures")

st.write("Upload an image of an ID card or signature to check if it is **Original** or **Forged**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    label = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    result = "Original" if label == 0 else "Forged"

    st.subheader("Prediction Result:")
    st.success(f"**{result}** (Confidence: {confidence:.2f}%)")

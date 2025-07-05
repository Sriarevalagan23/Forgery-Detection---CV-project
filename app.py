import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your model (ensure it's in the same directory or use relative path)
model = load_model("/Users/srinidhi/Desktop/forgery_detection/models/forgery_cnn_model.h5")  # Make sure this path is correct

# Preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("Forgery Detection Web App")
st.write("Upload an image to detect whether it's **Original** or **Forged**.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    img = preprocess_image(image)
    prediction = model.predict(img)
    label = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    result = "Original" if label == 0 else "Forged"

    st.subheader("Prediction Result")
    st.markdown(f"**Result:** {result}")
    st.markdown(f"**Confidence:** {round(confidence, 2)}%")

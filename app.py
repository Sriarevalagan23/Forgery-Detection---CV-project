from flask import Flask, render_template, request
from random import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import os
import gdown



app = Flask(__name__)

def download_model():
    model_path = "models/cnn_model.h5"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        file_id = "1aJ6m3IJ22L9a-Faawok3C-1AbhU4-AmW"  # Your Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    return model_path
    
model_path = download_model()
model = load_model("models/cnn_model.h5")
app.config['UPLOAD_FOLDER'] = 'uploads'


def preprocess_image(path):
    img = load_img(path, target_size=(128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template("index.html", random=random)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    label = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    result = "Original" if label == 0 else "Forged"

    return render_template('result.html', result=result, confidence=round(confidence, 2), filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)

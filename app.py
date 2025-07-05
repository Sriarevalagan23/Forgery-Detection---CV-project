from flask import Flask, render_template, request
from random import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model('/Users/srinidhi/Desktop/forgery_detection/forgery_cnn_model.h5')

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

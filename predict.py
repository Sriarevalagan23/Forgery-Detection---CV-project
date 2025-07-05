import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    print(f"Reading from folder: {folder}")
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
            images.append(img)
            labels.append(label)
    return images, labels

# Load images
original_folder = '/Users/srinidhi/Desktop/forgery_detection/dataset/original'
forged_folder = '/Users/srinidhi/Desktop/forgery_detection/dataset/forged'

X_original, y_original = load_images_from_folder(original_folder, 0)
X_forged, y_forged = load_images_from_folder(forged_folder, 1)

# Combine and normalize data
X = np.array(X_original + X_forged, dtype="float32") / 255.0
y = np.array(y_original + y_forged)

# One-hot encode for training
from tensorflow.keras.utils import to_categorical
y_cat = to_categorical(y, num_classes=2)

# Split the data
X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = train_test_split(
    X, y_cat, y, test_size=0.2, random_state=42
)

# Load trained model
model = load_model('/Users/srinidhi/Desktop/forgery_detection/models/cnn_model.h5')

# Evaluate model on one-hot encoded labels
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\nModel Evaluation:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict classes
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate metrics in percentage
acc = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='binary') * 100
recall = recall_score(y_test, y_pred, average='binary') * 100
f1 = f1_score(y_test, y_pred, average='binary') * 100

print("\nPerformance Metrics (%):")
print(f"Accuracy   : {acc:.2f}%")
print(f"Precision  : {precision:.2f}%")
print(f"Recall     : {recall:.2f}%")
print(f"F1-Score   : {f1:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

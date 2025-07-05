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
model = load_model('/Users/srinidhi/Desktop/forgery_detection/forgery_cnn_model.h5')

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

import pickle
import matplotlib.pyplot as plt

# Assuming you already have 'history' from model.fit()
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Accuracy Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = load_model("/Users/srinidhi/Desktop/forgery_detection/forgery_cnn_model.h5")

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Original", "Forged"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Original", "Forged"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

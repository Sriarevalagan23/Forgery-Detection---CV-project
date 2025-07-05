import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set paths
original_path = "/Users/srinidhi/Desktop/forgery_detection/dataset/original"
forged_path = "/Users/srinidhi/Desktop/forgery_detection/dataset/forged"

# Read and preprocess images
def load_images_from_folder(folder, label, image_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Make it 3-channel
            images.append(img)
            labels.append(label)
    return images, labels

print("Reading original images...")
original_images, original_labels = load_images_from_folder(original_path, 0)
print("Reading forged images...")
forged_images, forged_labels = load_images_from_folder(forged_path, 1)

# Combine data
X = np.array(original_images + forged_images, dtype='float32') / 255.0
y = np.array(original_labels + forged_labels)

print("Total images:", len(X))
print("Total labels:", len(y))

# One-hot encode labels
y_cat = to_categorical(y, 2)

# Split data
X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = train_test_split(
    X, y_cat, y, test_size=0.2, stratify=y, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# Build improved CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=16),
    epochs=30,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop]
)

# Save model
model.save("forgery_cnn_model.h5")
print("Model saved as forgery_cnn_model.h5")

import pickle

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


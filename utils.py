import cv2
import numpy as np
import os

def preprocess_image(image_path, size=(128, 128)):
    """
    Reads and preprocesses an image: resize, grayscale, normalize.

    Args:
        image_path (str): Path to the image.
        size (tuple): Desired size (width, height).

    Returns:
        numpy.ndarray: Flattened feature vector.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")

    # Resize to fixed size
    image = cv2.resize(image, size)

    # Normalize pixel values to 0–1
    image = image / 255.0

    # Flatten to 1D array
    return image.flatten()

def load_dataset(original_folder, forged_folder):
    X = []
    y = []

    print("Reading from folder:", original_folder)
    for file in os.listdir(original_folder):
        path = os.path.join(original_folder, file)
        if os.path.isfile(path):
            features = preprocess_image(path)
            X.append(features)
            y.append(0)  # Label 0 for original

    print("Reading from folder:", forged_folder)
    for file in os.listdir(forged_folder):
        path = os.path.join(forged_folder, file)
        if os.path.isfile(path):
            features = preprocess_image(path)
            X.append(features)
            y.append(1)  # Label 1 for forged

    return np.array(X), np.array(y)

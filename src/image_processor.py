# custom_inception.py
import os
import io
import json
import requests
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import preprocess_input

# ------------------------------
# Config / Paths
# ------------------------------
MODEL_PATH = "models/inceptionv3_finetuned.keras"
LABELS_PATH = "models/labels.json"
IMG_SIZE = (299, 299)

# ------------------------------
# 1. Load Custom Model
# ------------------------------
def load_model():
    """
    Load your trained InceptionV3 model and label mapping.
    Returns:
        model: tf.keras.Model
        idx_to_class: dict mapping class index -> class name
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        data = json.load(f)
    class_to_idx = data["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class

# ------------------------------
# 2. Image Loading Utilities
# ------------------------------
<<<<<<< HEAD
def load_image_from_file(filepath: str, target_size=(299, 299)):
    """Load and preprocess image from local file"""
    # Handle different possible image paths
    possible_paths = [
        filepath,  # Direct path
        "data/newData/" + filepath,  # Original path
        "data/kaggle_data/dataset/" + filepath,  # Kaggle dataset path
        "data/kaggle_data/" + filepath  # Alternative Kaggle path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            img = image.load_img(path, target_size=target_size)
            img_array = image.img_to_array(img)
            return np.expand_dims(img_array, axis=0)
    
    # If no valid path found, raise an error with all attempted paths
    raise FileNotFoundError(f"Image not found. Tried paths: {possible_paths}")
=======
def load_image_from_file(filepath: str, target_size=IMG_SIZE):
    """
    Load an image from a local file and preprocess for InceptionV3.
    Returns a numpy array with batch dimension: (1, H, W, 3)
    """
    img = Image.open("data/newData/" + filepath).convert("RGB")
    img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0,0,0))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)
>>>>>>> 239b0ba0748e865a4c9c21a8e2ffc409d026a978

def load_image_from_url(url, target_size=IMG_SIZE):
    """
    Load an image from a URL and preprocess for InceptionV3.
    Returns a numpy array with batch dimension: (1, H, W, 3)
    """
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0,0,0))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

# ------------------------------
# 3. Prediction Function
# ------------------------------
def classify_image(model, idx_to_class, img_array):
    """
    Run prediction using the trained model and return:
        - predicted class name
        - probability vector for all classes
    """
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    pred_class_idx = preds.argmax(axis=1)[0]
    pred_class_name = idx_to_class[pred_class_idx]
    return pred_class_name, preds[0]

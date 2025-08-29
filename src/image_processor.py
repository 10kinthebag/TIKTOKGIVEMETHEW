import os
import io
import requests
import pandas as pd
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image

# ------------------------------
# 1. Load InceptionV3 model
# ------------------------------
def load_model():
    model = InceptionV3(weights="imagenet")
    return model

# ------------------------------
# 2. Image Loading Utilities
# ------------------------------
def load_image_from_file(filepath: str, target_size=(299, 299)):
    """Load and preprocess image from local file"""
    img = image.load_img("data/newData/" + filepath, target_size=target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

def load_image_from_url(url, target_size=(299, 299)):
    """Load and preprocess image from URL"""
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

# ------------------------------
# 3. Prediction + Mapping
# ------------------------------
# Define your allowed categories
ALLOWED_DESCRIPTIONS = {
    "taste": ["pizza", "cheeseburger", "hotdog", "sushi", "ice_cream"],
    "menu": ["restaurant", "plate", "cup", "bottle", "fork"],
    "outdoor_atmosphere": ["patio", "park_bench", "picnic_table", "garden", "tent"],
    "indoor_atmosphere": ["dining_table", "couch", "lamp", "restaurant_indoor", "bar"]
}

def classify_image(model, img_array):
    """Run prediction and map to allowed categories or 'Other'"""
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]

    # Try to map prediction to allowed descriptions
    for _, label, prob in decoded:
        if label.lower() in ALLOWED_DESCRIPTIONS:
            return ALLOWED_DESCRIPTIONS[label.lower()]
    return "Other"

# ------------------------------
# 4. Apply Policy to Dataset
# ------------------------------
def classify_from_csv(csv_path, output_csv="classified_output.csv"):
    """
    Takes CSV with column 'image_path' (can be URL or file path).
    Outputs new CSV with 'classification' column.
    """
    df = pd.read_csv(csv_path)
    model = load_model()
    
    results = []
    for idx, row in df.iterrows():
        path = row['image_path']
        try:
            if path.startswith("http"):
                img_array = load_image_from_url(path)
            else:
                img_array = load_image_from_file(path)
            
            classification = classify_image(model, img_array)
        except Exception as e:
            classification = f"Error: {str(e)}"
        
        results.append(classification)
    
    df['classification'] = results
    df.to_csv(output_csv, index=False)
    return df

# ------------------------------
# Example Run (comment out in production)
# ------------------------------
if __name__ == "__main__":
    # Example CSV: image_path column contains either URLs or file paths
    data = {
        "image_path": [
            "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",  # dog
            "https://upload.wikimedia.org/wikipedia/commons/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg",  # pizza
            "local_image.jpg"  # local file example
        ]
    }
    test_csv = "images.csv"
    pd.DataFrame(data).to_csv(test_csv, index=False)
    
    result_df = classify_from_csv(test_csv)
    print(result_df)


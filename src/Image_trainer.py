# train_inception_v3.py
import os
import io
import json
import math
import requests
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS_FROZEN = 15        # train top head only
EPOCHS_FINE_TUNE = 30     # unfreeze some base layers
FINE_TUNE_AT = 249        # unfreeze from this layer index in InceptionV3
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "inceptionv3_finetuned.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data loader (URLs or files)
# -----------------------------
def load_image_any(path: str, target_size=IMG_SIZE) -> np.ndarray:
    """Load image from local path or URL and return float32 array [H,W,3]."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")

    if str(path).lower().startswith("http"):
        resp = requests.get(path, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(path).convert("RGB")

    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    return arr

class ImageSequence(tf.keras.utils.Sequence):
    """Simple Keras Sequence that loads images on the fly from URLs or files."""
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, batch_size=BATCH_SIZE, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.class_to_idx = class_to_idx
        self.augment = augment

        # Keras preprocessing/augmentation as a small model
        aug_layers = []
        if augment:
            aug_layers += [
                layers.RandomContrast(0.1),
                layers.RandomTranslation(0.1, 0.1),

            ]
        self.augmenter = tf.keras.Sequential(aug_layers)

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch = self.df.iloc[start:start + self.batch_size]
        xs = []
        ys = []
        for _, row in batch.iterrows():
            path = row["image_path"]
            label = row["label"]  # one of: taste, menu, outdoor_atmosphere, indoor_atmosphere
            try:
                img = load_image_any(path)
                xs.append(img)
                ys.append(self.class_to_idx[label])
            except Exception:
                # Skip bad rows; keep batch shape by inserting black image & "dummy" label
                xs.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
                ys.append(self.class_to_idx[next(iter(self.class_to_idx))])

        x = np.stack(xs, axis=0)
        x = preprocess_input(x)  # InceptionV3 preprocessing

        # Apply Keras augmentation (if enabled)
        if self.augment and len(x) > 0:
            x = self.augmenter(x, training=True)

        y = tf.keras.utils.to_categorical(ys, num_classes=len(self.class_to_idx))
        return x, y

# -----------------------------
# Model
# -----------------------------
def build_model(num_classes: int) -> tf.keras.Model:
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # start frozen

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base

# -----------------------------
# Training
# -----------------------------
def build_dataframe_from_folders(root_dir: str) -> pd.DataFrame:
    """
    Scans a directory structure like:
        root_dir/
            class1/
                img1.jpg
                img2.png
            class2/
                img3.jpg
    and builds a DataFrame with columns [image_path, label].
    """
    rows = []
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                rows.append({"image_path": fpath, "label": label})
    df = pd.DataFrame(rows)
    return df

def train(input_path: str, output_model_path: str = MODEL_PATH, labels_out: str = LABELS_PATH):
    """
    input_path can be either:
      - a CSV file with columns [image_path,label]
      - a directory containing subfolders per class
    """
    if os.path.isdir(input_path):
        print(f"Building dataset from folder structure: {input_path}")
        df = build_dataframe_from_folders(input_path)
    else:
        print(f"Loading dataset from CSV: {input_path}")
        df = pd.read_csv(input_path)

    df = df.dropna(subset=["image_path", "label"]).copy()
    ...


    # Classes
    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(labels_out, "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)

    # Split
    df = df.sample(frac=1.0, random_state=42)
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    train_seq = ImageSequence(df_train, class_to_idx, augment=True)
    val_seq   = ImageSequence(df_val,   class_to_idx, augment=False)
    test_seq  = ImageSequence(df_test,  class_to_idx, augment=False)

    # Build & compile
    model, base = build_model(num_classes=len(classes))
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cbs = [
        callbacks.ModelCheckpoint(output_model_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.2, monitor="val_loss", mode="min"),
    ]

    # Stage 1: train head
    model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS_FROZEN, callbacks=cbs)

    # Stage 2: fine-tune upper layers of base
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= FINE_TUNE_AT)

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS_FINE_TUNE, callbacks=cbs)

    # Evaluate
    eval_metrics = model.evaluate(test_seq, verbose=0)
    print(f"Test metrics: {model.metrics_names} -> {eval_metrics}")

    # Save final (also saved best via checkpoint)
    model.save(output_model_path)
    print(f"Saved model to {output_model_path} and labels to {labels_out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Training input: either a CSV file or a directory with subfolders per class")
    ap.add_argument("--out", default=MODEL_PATH, help="Output model path")
    ap.add_argument("--labels_out", default=LABELS_PATH, help="Output labels json")
    args = ap.parse_args()
    train(args.input, args.out, args.labels_out)


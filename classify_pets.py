"""
classify_pets.py — Cats vs Dogs classifier using MobileNetV2 transfer learning.
Saved with TF 2.15 / Keras 2.15 — loaded via tf.keras for compatibility.
Preprocessing: resize to 160×160, rescale to [0, 1].
"""

import os
import json
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Use tf.keras (Keras 2 compat) to load the pets model which was saved with Keras 2.15
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from PIL import Image

MODEL_PATH_H5    = "saved_model/pets_classification_model_fixed.h5"
MODEL_PATH_KERAS = "saved_model/pets_classification_model.keras"
LABELS_PATH      = "saved_model/pets_class_names.json"
IMG_SIZE         = (160, 160)
labels           = {0: "Cat", 1: "Dog"}

if os.path.exists(MODEL_PATH_H5):
    model = tf.keras.models.load_model(MODEL_PATH_H5, compile=False)
    print(f"[classify_pets] Loaded h5 model")
elif os.path.exists(MODEL_PATH_KERAS):
    model = tf.keras.models.load_model(MODEL_PATH_KERAS, compile=False)
    print(f"[classify_pets] Loaded keras model")
else:
    raise FileNotFoundError("Pets model not found. Run train_pets_model.py first.")

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        raw = json.load(f)
    _name_map = {"cats_set": "Cat", "dogs_set": "Dog", "cat": "Cat", "dog": "Dog"}
    labels = {v: _name_map.get(k.lower(), k.capitalize()) for k, v in raw.items()}

print(f"[classify_pets] Ready | input: {IMG_SIZE} | classes: {labels}")


def _preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_pet_image(img_path: str) -> dict:
    try:
        arr  = _preprocess(img_path)
        pred = model.predict(arr, verbose=0)
        idx  = int(np.argmax(pred[0]))
        return {"label": labels[idx], "confidence": round(float(pred[0][idx]) * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

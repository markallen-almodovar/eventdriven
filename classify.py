"""
classify.py — Gender classifier using MobileNetV2 transfer learning.
Uses keras 3 directly to load the model (saved with keras 3.12.1).
Preprocessing: resize to model input size, rescale to [0, 1].
"""

import os
import json
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
keras.utils.set_random_seed(42)

from PIL import Image

MODEL_PATH  = "saved_model/gender_classification_model.keras"
LABELS_PATH = "saved_model/class_names.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Run fine_tune_model.py first.")

model = keras.saving.load_model(MODEL_PATH, compile=False)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
else:
    labels = {0: "Female", 1: "Male"}

_input_shape = model.input_shape
IMG_SIZE = (_input_shape[1], _input_shape[2])

print(f"[classify] Model loaded | input: {IMG_SIZE} | classes: {labels}")


def _preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image(img_path: str) -> dict:
    try:
        arr  = _preprocess(img_path)
        pred = model.predict(arr, verbose=0)
        idx  = int(np.argmax(pred[0]))
        return {"label": labels[idx], "confidence": round(float(pred[0][idx]) * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

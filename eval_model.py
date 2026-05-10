"""
eval_model.py — Evaluate the baseline cats_dogs_model.keras on the validation set.

This evaluates the simple CNN model from training.py.
For the MobileNetV2 model, use evaluate_pets_model.py instead.

Usage:
  venv_new\\Scripts\\python.exe eval_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODEL_PATH   = "cats_dogs_model.keras"
VAL_DIR      = "dogs_cats_split/validation"
IMAGE_SIZE   = (150, 150)
BATCH_SIZE   = 16

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Run training.py first."
    )

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────

loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nValidation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")

# ──────────────────────────────────────────────
# PREDICTIONS
# ──────────────────────────────────────────────

predictions = model.predict(val_generator, verbose=1)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nClassification Report:")
print(report)

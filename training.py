"""
training.py — Simple CNN baseline for Cats vs Dogs classification.

This is a lightweight baseline model (not MobileNetV2).
For production use, run train_pets_model.py instead which uses
MobileNetV2 transfer learning for much better accuracy.

Usage:
  venv_new\\Scripts\\python.exe training.py
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

TRAIN_DIR  = "dogs_cats_split/train"
VAL_DIR    = "dogs_cats_split/validation"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 10

# ──────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# ──────────────────────────────────────────────
# MODEL — simple CNN
# ──────────────────────────────────────────────

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),  # binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────

model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ],
)

# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────

model.save("cats_dogs_model.keras")
print("\nBaseline model saved to cats_dogs_model.keras")
print("For better accuracy, run: venv_new\\Scripts\\python.exe train_pets_model.py")

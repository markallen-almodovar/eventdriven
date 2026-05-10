"""
compare_models.py — Compare the baseline CNN vs the MobileNetV2 fine-tuned model
on the test set.

Usage:
  venv_new\\Scripts\\python.exe compare_models.py

Update FINE_MODEL_PATH to point to your fine-tuned model once trained.
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

BASELINE_MODEL_PATH = "cats_dogs_model.keras"
FINE_MODEL_PATH     = "saved_model/pets_classification_model.keras"

TEST_DIR   = "dogs_cats_split/test"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def evaluate_binary_model(model, generator, name: str):
    """Evaluate a binary-output (sigmoid) model."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    loss, acc = model.evaluate(generator, verbose=1)
    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc * 100:.2f}%")

    preds  = model.predict(generator, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = generator.classes
    labels = list(generator.class_indices.keys())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))


def evaluate_categorical_model(model, test_dir: str, img_size, batch_size: int, name: str):
    """Evaluate a categorical-output (softmax) model using tf.data."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=42,
    )
    class_names = test_ds.class_names

    norm = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc * 100:.2f}%")

    all_preds, all_labels = [], []
    for images, labels_batch in test_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(labels_batch.numpy(), axis=1))

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ──────────────────────────────────────────────
# BASELINE MODEL (binary CNN)
# ──────────────────────────────────────────────

if os.path.exists(BASELINE_MODEL_PATH):
    print(f"Loading baseline model: {BASELINE_MODEL_PATH}")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH, compile=False)
    baseline_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    evaluate_binary_model(baseline_model, test_gen, "Baseline CNN (cats_dogs_model.keras)")
else:
    print(f"Baseline model not found at '{BASELINE_MODEL_PATH}'. Run training.py first.")

# ──────────────────────────────────────────────
# FINE-TUNED MOBILENETV2 MODEL (categorical)
# ──────────────────────────────────────────────

if os.path.exists(FINE_MODEL_PATH):
    print(f"\nLoading fine-tuned model: {FINE_MODEL_PATH}")
    fine_model = tf.keras.models.load_model(FINE_MODEL_PATH, compile=False)
    _shape = fine_model.input_shape
    fine_img_size = (_shape[1], _shape[2])
    evaluate_categorical_model(
        fine_model, TEST_DIR, fine_img_size, BATCH_SIZE,
        "MobileNetV2 Fine-Tuned (pets_classification_model.keras)"
    )
else:
    print(f"\nFine-tuned model not found at '{FINE_MODEL_PATH}'. Run train_pets_model.py first.")

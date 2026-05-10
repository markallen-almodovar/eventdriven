"""
convert_models.py — Convert .keras (zip) format models to .h5 format.

TF 2.15 on Windows hangs when loading .keras format due to a bug where it
prints the entire model config JSON to stderr, overflowing the process buffer.
H5 format loads cleanly without this issue.

Usage:
  venv_new\\Scripts\\python.exe convert_models.py
"""

import os
import sys
import json
import zipfile
import tempfile
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

MODELS = [
    ("saved_model/gender_classification_model.keras", "saved_model/gender_classification_model.h5"),
    ("saved_model/pets_classification_model.keras",   "saved_model/pets_classification_model.h5"),
]

def rebuild_mobilenetv2_model(num_classes=2, img_size=(160, 160)):
    """Rebuild the MobileNetV2 + head architecture from scratch."""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    base = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights=None)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)


def convert(src, dst):
    if not os.path.exists(src):
        print(f"  SKIP: {src} not found")
        return False
    if os.path.exists(dst):
        print(f"  SKIP: {dst} already exists")
        return True

    print(f"  Converting {src} → {dst}")

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(tmpdir)

        weights_path = os.path.join(tmpdir, "model.weights.h5")

        # Rebuild architecture and load weights by name
        model = rebuild_mobilenetv2_model(num_classes=2)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        model.save(dst)
        print(f"  Saved: {dst}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


print("Converting models to H5 format...")
for src, dst in MODELS:
    convert(src, dst)

print("\nDone. Run the ML server with:")
print("  venv_new\\Scripts\\python.exe -m uvicorn mlserver:app --port 8000")

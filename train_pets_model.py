"""
train_pets_model.py — Train a Cats vs Dogs classifier using MobileNetV2
transfer learning.  Saves the final model to saved_model/pets_classification_model.keras.

GPU + CPU split:
  - Automatically uses GPU if CUDA is available (RTX 3050 etc.)
  - Falls back to CPU-only if no GPU is detected
  - Uses tf.distribute.MirroredStrategy for multi-device training when possible

Dataset layout (already present):
  dogs_cats_split/
    train/cats_set/ + dogs_set/
    validation/cats_set/ + dogs_set/
    test/cats_set/ + dogs_set/

Two-phase training:
  Phase 1 — Train only the new classification head (MobileNetV2 base frozen)
  Phase 2 — Unfreeze top layers of MobileNetV2 and fine-tune end-to-end

Usage:
  venv_new\\Scripts\\python.exe train_pets_model.py

  # Resume from Phase 2 only (if Phase 1 already done):
  venv_new\\Scripts\\python.exe train_pets_model.py --phase2-only
"""

import os
import sys
import json
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ──────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--phase2-only", action="store_true",
                    help="Skip Phase 1 and load checkpoint for Phase 2 fine-tuning only")
args = parser.parse_args()

# ──────────────────────────────────────────────
# DEVICE SETUP — GPU + CPU split
# ──────────────────────────────────────────────

gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

print(f"\n{'='*60}")
print(f"  Device Detection")
print(f"{'='*60}")
print(f"  GPUs available : {len(gpus)}")
for g in gpus:
    print(f"    {g.name}")
print(f"  CPUs available : {len(cpus)}")
for c in cpus:
    print(f"    {c.name}")

if gpus:
    # Allow GPU memory growth — prevents TF from grabbing all VRAM at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # MirroredStrategy: splits batches across all available GPUs (and CPU as fallback)
    # On a single GPU system this still gives you GPU acceleration
    strategy = tf.distribute.MirroredStrategy()
    print(f"\n  Strategy       : MirroredStrategy ({strategy.num_replicas_in_sync} replica(s))")
    DEVICE_NOTE = f"GPU ({len(gpus)} device(s))"
else:
    # No GPU — use default CPU strategy but split work across logical CPU cores
    # Create 2 logical CPUs to simulate parallel execution on multi-core CPU
    try:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ]
        )
        logical_cpus = tf.config.list_logical_devices("CPU")
        print(f"\n  Logical CPUs   : {len(logical_cpus)} (split for parallel data loading)")
    except RuntimeError:
        pass  # already configured

    strategy = tf.distribute.MirroredStrategy(["CPU:0"])
    print(f"\n  Strategy       : CPU-only MirroredStrategy")
    DEVICE_NOTE = "CPU-only (install CUDA 11.8 + cuDNN 8.6 for GPU acceleration)"

print(f"  Note           : {DEVICE_NOTE}")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Scale batch size by number of replicas for distributed training
BASE_BATCH   = 32
BATCH_SIZE   = BASE_BATCH * strategy.num_replicas_in_sync
IMAGE_SIZE   = (160, 160)
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 30
UNFREEZE_FROM = 100   # unfreeze MobileNetV2 layers from this index onward

TRAIN_DIR = "dogs_cats_split/train"
VAL_DIR   = "dogs_cats_split/validation"
TEST_DIR  = "dogs_cats_split/test"
SAVE_DIR  = "saved_model"
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "pets_best_checkpoint.keras")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "pets_classification_model.keras")
LABELS_PATH = os.path.join(SAVE_DIR, "pets_class_names.json")

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Batch size      : {BATCH_SIZE} (base {BASE_BATCH} × {strategy.num_replicas_in_sync} replica(s))")
print(f"Image size      : {IMAGE_SIZE}")
print(f"Phase 1 epochs  : {PHASE1_EPOCHS}")
print(f"Phase 2 epochs  : {PHASE2_EPOCHS}")

# ──────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────

AUTOTUNE = tf.data.AUTOTUNE


def build_dataset(directory: str, training: bool):
    """
    Load images and apply augmentation if training.
    Normalisation: Rescaling(1/255) → [0, 1].  Must match classify_pets.py.
    """
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=training,
        seed=42,
    )
    names = raw_ds.class_names

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ])
        ds = raw_ds.map(
            lambda x, y: (augmentation(normalization(x), training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = raw_ds.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=AUTOTUNE,
        )

    return ds.cache().prefetch(AUTOTUNE), names


print("\nLoading datasets...")
train_ds, class_names = build_dataset(TRAIN_DIR, training=True)
val_ds,   _           = build_dataset(VAL_DIR,   training=False)
num_classes = len(class_names)

print(f"Classes           : {class_names}")
print(f"Training batches  : {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}\n")

# Save class labels immediately so they're available even if training is interrupted
class_indices = {name: idx for idx, name in enumerate(class_names)}
with open(LABELS_PATH, "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"Labels saved      : {LABELS_PATH}  →  {class_indices}\n")

# ──────────────────────────────────────────────
# MODEL — built inside strategy scope for distribution
# ──────────────────────────────────────────────

def build_model(num_classes: int, base_trainable: bool = False):
    """Build MobileNetV2 + classification head."""
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=outputs), base_model


# ──────────────────────────────────────────────
# PHASE 1 — Train classification head only
# ──────────────────────────────────────────────

if not args.phase2_only:
    # ── Phase 1: train head only ──────────────────────────────────────────────
    print("=== Phase 1: Training classification head (base frozen) ===\n")

    with strategy.scope():
        model, _ = build_model(num_classes, base_trainable=False)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.summary(line_length=90)

    callbacks_p1 = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    history1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_p1,
    )

    best_p1_acc = max(history1.history["val_accuracy"])
    print(f"\nPhase 1 complete — best val_accuracy: {best_p1_acc * 100:.2f}%")

else:
    # ── Resume from saved Phase 1 checkpoint ─────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: No checkpoint found at '{CHECKPOINT_PATH}'.")
        print("Run without --phase2-only first to complete Phase 1.")
        sys.exit(1)

    print(f"Loading Phase 1 checkpoint: {CHECKPOINT_PATH}")
    with strategy.scope():
        model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)

    print("Phase 1 skipped — resuming from checkpoint for Phase 2.\n")

# ──────────────────────────────────────────────
# PHASE 2 — Unfreeze top layers and fine-tune
# ──────────────────────────────────────────────

print("\n=== Phase 2: Fine-tuning top MobileNetV2 layers ===\n")

with strategy.scope():
    # Unfreeze top portion of the model layers.
    # MobileNetV2 layers are indices 1–153 in the flat model; head is 154+.
    # We freeze layers 0..UNFREEZE_FROM-1 and unfreeze the rest up to the head.
    # The head layers (GlobalAveragePooling onward) are always trainable.
    mobilenet_layer_names = {
        "Conv1", "bn_Conv1", "Conv1_relu",  # stem
    }
    # Identify MobileNetV2 body layers: everything before global_average_pooling2d
    head_start_names = {"global_average_pooling2d", "batch_normalization", "dense",
                        "dropout", "dense_1", "dropout_1", "dense_2"}

    mobilenet_layers = [l for l in model.layers
                        if l.name not in head_start_names
                        and not l.name.startswith("input")]

    total_mb = len(mobilenet_layers)
    freeze_count = min(UNFREEZE_FROM, total_mb)

    for i, layer in enumerate(mobilenet_layers):
        layer.trainable = (i >= freeze_count)

    unfrozen = sum(1 for l in mobilenet_layers if l.trainable)
    print(f"MobileNetV2 layers : {total_mb} total, {freeze_count} frozen, {unfrozen} unfrozen")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

callbacks_p2 = [
    EarlyStopping(
        monitor="val_loss", patience=7,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-7, verbose=1,
    ),
    ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

history2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_p2,
)

# ──────────────────────────────────────────────
# SAVE FINAL MODEL
# ──────────────────────────────────────────────

model.save(FINAL_MODEL_PATH)
print(f"\nFinal model saved : {FINAL_MODEL_PATH}")

# ──────────────────────────────────────────────
# FINAL EVALUATION
# ──────────────────────────────────────────────

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n{'='*60}")
print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Final Validation Loss     : {val_loss:.4f}")
print(f"  Device used               : {DEVICE_NOTE}")
print(f"{'='*60}")

if os.path.exists(TEST_DIR):
    test_ds, _ = build_dataset(TEST_DIR, training=False)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n  Test Set Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Set Loss     : {test_loss:.4f}")

print("\nTraining complete!")
print(f"Run evaluate_pets_model.py for detailed metrics and confusion matrix.")

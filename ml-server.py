"""
ml-server.py — FastAPI server for AI classification.

Endpoints:
  GET  /                    — health check
  GET  /health              — detailed health (model info, classes)

  Gender Classification:
  POST /gender/predict      — upload face image, returns Male/Female
  POST /gender/upload       — alias for /gender/predict

  Pets Classification:
  POST /pets/predict        — upload pet image, returns Cat/Dog
  POST /pets/upload         — alias for /pets/predict

  Legacy (backward compat):
  POST /upload              — redirects to gender classifier
  POST /predict             — redirects to gender classifier
"""

# ── Suppress ALL TF/Keras noise before any import ─────────────────────────────
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"
os.environ["PYTHONWARNINGS"]              = "ignore"
os.environ["TF_KERAS_SERIALIZATION_DEBUG"] = "0"

# Swap stderr for /dev/null while models load so the Keras JSON config
# dump never reaches the terminal / process buffer.
# We restore it right after so uvicorn logs still appear normally.
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import uuid, glob

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────
# LOAD CLASSIFIERS  (stderr still silenced here)
# ──────────────────────────────────────────────

from classify import classify_image, labels as gender_labels, IMG_SIZE as gender_img_size

_pets_available = False
_pets_error_msg = ""
try:
    from classify_pets import classify_pet_image, labels as pets_labels, IMG_SIZE as pets_img_size
    _pets_available = True
except FileNotFoundError as e:
    _pets_error_msg = str(e)
    pets_labels   = {0: "Cat", 1: "Dog"}
    pets_img_size = (160, 160)

# ── Restore stderr so uvicorn/FastAPI logs are visible ────────────────────────
sys.stderr.close()
sys.stderr = _real_stderr

print(f"[ml-server] Gender model ready  | classes: {gender_labels}")
if _pets_available:
    print(f"[ml-server] Pets model ready    | classes: {pets_labels}")
else:
    print(f"[ml-server] WARNING: Pets model not available — {_pets_error_msg}")

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = FastAPI(title="AI Classification API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

for stale in glob.glob(os.path.join(UPLOAD_DIR, "*")):
    try:
        os.remove(stale)
    except OSError:
        pass

ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_MB    = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ──────────────────────────────────────────────
# HEALTH ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "AI Classification API is running",
        "version": "3.0.0",
        "endpoints": {"gender": "/gender/predict", "pets": "/pets/predict"},
        "pets_model_ready": _pets_available,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "gender": {
                "model":      "gender_classification_model.keras",
                "input_size": list(gender_img_size),
                "classes":    gender_labels,
                "ready":      True,
            },
            "pets": {
                "model":      "pets_classification_model.keras",
                "input_size": list(pets_img_size),
                "classes":    pets_labels,
                "ready":      _pets_available,
                "error":      _pets_error_msg if not _pets_available else None,
            },
        },
    }


# ──────────────────────────────────────────────
# GENDER ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/gender/predict")
async def predict_gender(file: UploadFile = File(...)):
    return await _classify_file(file, classifier="gender")

@app.post("/gender/upload")
async def upload_gender_image(file: UploadFile = File(...)):
    return await _classify_file(file, classifier="gender")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    return await _classify_file(file, classifier="gender")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await _classify_file(file, classifier="gender")


# ──────────────────────────────────────────────
# PETS ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/pets/predict")
async def predict_pet(file: UploadFile = File(...)):
    if not _pets_available:
        raise HTTPException(status_code=503, detail=f"Pets model not ready. {_pets_error_msg}")
    return await _classify_file(file, classifier="pets")

@app.post("/pets/upload")
async def upload_pet_image(file: UploadFile = File(...)):
    return await predict_pet(file)


# ──────────────────────────────────────────────
# SHARED HANDLER
# ──────────────────────────────────────────────

async def _classify_file(file: UploadFile, classifier: str = "gender"):
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB} MB.")

    file_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    try:
        with open(file_path, "wb") as buf:
            buf.write(contents)

        result = classify_pet_image(file_path) if classifier == "pets" else classify_image(file_path)

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

"""
mlserver.py — FastAPI server for AI classification.

Endpoints:
  GET  /                    — health check
  GET  /health              — model info + classes

  Gender:
  POST /gender/predict      — face image → Male/Female + confidence
  POST /gender/upload       — alias

  Pets:
  POST /pets/predict        — pet image → Cat/Dog + confidence
  POST /pets/upload         — alias

  Legacy:
  POST /upload              — gender (backward compat)
  POST /predict             — gender (backward compat)
"""

import os
import sys
import uuid
import glob
import asyncio

# Suppress TF noise before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── App setup (models loaded in startup event, not at import time) ────────────

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
for _f in glob.glob(os.path.join(UPLOAD_DIR, "*")):
    try:
        os.remove(_f)
    except OSError:
        pass

ALLOWED_EXT    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SIZE_MB    = 5
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

# These are populated in the startup event
_classify_image    = None
_classify_pet      = None
_gender_labels     = {}
_pets_labels       = {0: "Cat", 1: "Dog"}
_gender_img_size   = (160, 160)
_pets_img_size     = (160, 160)
_pets_available    = False
_pets_error_msg    = ""
_models_ready      = False


@app.on_event("startup")
async def load_models():
    """Load TF models in a thread so uvicorn starts immediately."""
    global _classify_image, _classify_pet, _gender_labels, _pets_labels
    global _gender_img_size, _pets_img_size, _pets_available, _pets_error_msg, _models_ready

    loop = asyncio.get_event_loop()

    def _load():
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)

        from classify import (
            classify_image, labels as g_labels, IMG_SIZE as g_size
        )
        nonlocal_vars = {
            "classify_image": classify_image,
            "g_labels": g_labels,
            "g_size": g_size,
            "pets_available": False,
            "pets_error": "",
            "classify_pet": None,
            "p_labels": {0: "Cat", 1: "Dog"},
            "p_size": (160, 160),
        }
        try:
            from classify_pets import (
                classify_pet_image, labels as p_labels, IMG_SIZE as p_size
            )
            nonlocal_vars["classify_pet"]    = classify_pet_image
            nonlocal_vars["p_labels"]        = p_labels
            nonlocal_vars["p_size"]          = p_size
            nonlocal_vars["pets_available"]  = True
        except FileNotFoundError as e:
            nonlocal_vars["pets_error"] = str(e)
        return nonlocal_vars

    print("[mlserver] Loading models in background thread...")
    result = await loop.run_in_executor(None, _load)

    _classify_image  = result["classify_image"]
    _classify_pet    = result["classify_pet"]
    _gender_labels   = result["g_labels"]
    _gender_img_size = result["g_size"]
    _pets_labels     = result["p_labels"]
    _pets_img_size   = result["p_size"]
    _pets_available  = result["pets_available"]
    _pets_error_msg  = result["pets_error"]
    _models_ready    = True

    print(f"[mlserver] Gender model ready | classes: {_gender_labels}")
    if _pets_available:
        print(f"[mlserver] Pets model ready   | classes: {_pets_labels}")
    else:
        print(f"[mlserver] WARNING: Pets model not available — {_pets_error_msg}")

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "AI Classification API is running",
        "version": "3.0.0",
        "models_ready": _models_ready,
        "endpoints": {"gender": "/gender/predict", "pets": "/pets/predict"},
        "pets_model_ready": _pets_available,
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if _models_ready else "loading",
        "models": {
            "gender": {
                "model":      "gender_classification_model.keras",
                "input_size": list(_gender_img_size),
                "classes":    _gender_labels,
                "ready":      _models_ready,
            },
            "pets": {
                "model":      "pets_classification_model.keras",
                "input_size": list(_pets_img_size),
                "classes":    _pets_labels,
                "ready":      _pets_available,
                "error":      _pets_error_msg if not _pets_available else None,
            },
        },
    }

# ── Gender endpoints ──────────────────────────────────────────────────────────

@app.post("/gender/predict")
async def predict_gender(file: UploadFile = File(...)):
    return await _handle(file, "gender")

@app.post("/gender/upload")
async def upload_gender(file: UploadFile = File(...)):
    return await _handle(file, "gender")

@app.post("/upload")
async def legacy_upload(file: UploadFile = File(...)):
    return await _handle(file, "gender")

@app.post("/predict")
async def legacy_predict(file: UploadFile = File(...)):
    return await _handle(file, "gender")

# ── Pets endpoints ────────────────────────────────────────────────────────────

@app.post("/pets/predict")
async def predict_pet(file: UploadFile = File(...)):
    if not _pets_available:
        raise HTTPException(status_code=503, detail=f"Pets model not ready. {_pets_error_msg}")
    return await _handle(file, "pets")

@app.post("/pets/upload")
async def upload_pet(file: UploadFile = File(...)):
    return await predict_pet(file)

# ── Shared handler ────────────────────────────────────────────────────────────

async def _handle(file: UploadFile, classifier: str):
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models are still loading. Please wait a moment.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )

    data = await file.read()
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_SIZE_MB} MB.")

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    try:
        with open(path, "wb") as f:
            f.write(data)

        result = _classify_pet(path) if classifier == "pets" else _classify_image(path)

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(path):
            os.remove(path)

"""
start_ml_server.py — Starts the ML FastAPI server.
Runs uvicorn programmatically with the mlserver app.
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn

if __name__ == "__main__":
    print("[start_ml_server] Starting uvicorn on port 8000...")
    uvicorn.run("mlserver:app", host="0.0.0.0", port=8000, log_level="info")

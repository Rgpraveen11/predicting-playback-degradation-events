from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Playback Degradation Prediction API",
    description="API for predicting streaming playback degradation risk.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"

# Use the same preprocessing and model pipeline
# that was trained on your project dataset.
model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


class SessionInput(BaseModel):
    mobile_carrier_network: str = "Alpha_4G"
    signal_strength_dbm: float = Field(default=-85, ge=-150, le=0)
    cell_tower_congestion_ratio: float = Field(default=0.70, ge=0, le=1)
    concurrent_regional_viewers_k: float = Field(default=300, ge=0)
    cdn_edge_latency_ms: float = Field(default=80, ge=0)
    requested_bitrate_mbps: float = Field(default=8, ge=0)
    available_bandwidth_mbps: float = Field(default=12, ge=0)
    device_cpu_load_pct: float = Field(default=85, ge=0, le=100)


@app.get("/")
def home():
    return {
        "message": "Playback Degradation Prediction API is running",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(session: SessionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained model not found. Train and save your "
                "existing preprocessing/model pipeline first."
            ),
        )

    # These column names must match the names and format
    # used during training in app.py.
    row = pd.DataFrame([{
        "Mobile_Carrier_Network": session.mobile_carrier_network,
        "Signal_Strength_dBm": session.signal_strength_dbm,
        "Cell_Tower_Congestion_Ratio":
            session.cell_tower_congestion_ratio,
        "Concurrent_Regional_Viewers_k":
            session.concurrent_regional_viewers_k,
        "CDN_Edge_Latency_ms": session.cdn_edge_latency_ms,
        "Requested_Bitrate_Mbps": session.requested_bitrate_mbps,
        "Available_Bandwidth_Mbps": session.available_bandwidth_mbps,
        "Device_CPU_Load_Pct": session.device_cpu_load_pct,
    }])

    try:
        probability = float(model.predict_proba(row)[0][1])
        prediction = int(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check that the saved model "
                "pipeline uses the same input columns and preprocessing. "
                f"Details: {exc}"
            ),
        ) from exc

    return {
        "predicted_class": prediction,
        "classification": (
            "Degradation risk" if prediction == 1 else "Lower risk"
        ),
        "degradation_probability": round(probability, 4),
        "degradation_probability_percent": round(
            probability * 100, 2
        ),
    }
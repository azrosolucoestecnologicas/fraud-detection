"""
Fraud Detection — Inference API
================================
FastAPI service that loads the champion model from MLflow Model Registry
and exposes a REST endpoint for real-time fraud scoring.

Endpoints:
    GET  /health    → liveness + model status
    POST /predict   → batch fraud probability
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Union

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
MODEL_URI: str = os.getenv("MODEL_URI", "models:/fraud-xgb@champion")

FEATURE_NAMES: list[str] = [
    "amount", "hour", "dow", "channel", "international", "new_merchant",
    "acct_age_days", "txn_count_24h", "txn_amount_24h", "distance_km",
    "device_change", "ip_risk",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

model = None


# ──────────────────────────────────────────────
# App Lifecycle
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global model
    mlflow.set_tracking_uri(TRACKING_URI)
    log.info("Loading model from %s", MODEL_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    log.info("Model loaded successfully")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for financial transactions (XGBoost + MLflow)",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Batch of transactions — list of arrays or list of dicts."""
    x: list[Union[list[float], dict]] = Field(
        ...,
        examples=[[
            [50.0, 10, 1, 0, 0, 0, 800, 1, 50.0, 0.5, 0, 0.05],
            [3500.0, 3, 6, 1, 1, 1, 30, 8, 4200.0, 2800.0, 1, 0.92],
        ]],
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_uri: str
    model_loaded: bool


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_uri=MODEL_URI,
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, detail="Model not loaded yet")

    if not req.x:
        raise HTTPException(422, detail="Empty transaction list")

    if isinstance(req.x[0], dict):
        df = pd.DataFrame(req.x)
    else:
        if any(len(row) != len(FEATURE_NAMES) for row in req.x):
            raise HTTPException(
                422,
                detail=f"Each row must have {len(FEATURE_NAMES)} features: {FEATURE_NAMES}",
            )
        df = pd.DataFrame(req.x, columns=FEATURE_NAMES)

    preds = model.predict(df)
    return PredictResponse(
        predictions=[round(float(p), 6) for p in preds],
        count=len(preds),
    )

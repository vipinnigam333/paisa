"""
api.py
------
FastAPI service that serves the trained model as a prediction endpoint.

Design choices worth knowing (and being able to explain in an interview):
- The API takes a ticker + lookback window, fetches recent data itself,
  builds the same features used at training time, and returns a
  direction + probability. This mirrors how the model would be called
  in a real scheduled-inference or trading-signal setting.
- The model is loaded once at startup, not per-request (avoids repeated
  disk I/O and deserialization cost).
- /health is separate from / so a load balancer / k8s readiness probe
  can check liveness without triggering a full prediction.

Run locally:
    uvicorn src.api:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl "http://localhost:8000/predict?ticker=AAPL"
"""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_data
from src.features import build_features, FEATURE_COLUMNS

MODEL_PATH = "models/model.joblib"

model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_bundle["data"] = joblib.load(MODEL_PATH)
        print(f"[api] Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[api] WARNING: no model found at {MODEL_PATH}. Run `python -m src.train` first.")
        model_bundle["data"] = None
    yield
    model_bundle.clear()


app = FastAPI(
    title="Stock Direction Prediction API",
    description="Serves next-day directional predictions for a given ticker.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    ticker: str
    as_of_date: str
    predicted_direction: str  # "UP" or "DOWN"
    probability_up: float
    is_synthetic_data: bool
    disclaimer: str = (
        "Educational project only. Not financial advice. "
        "Model accuracy is only marginally better than a naive baseline; "
        "do not use this to make real trading decisions."
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle.get("data") is not None}


@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str = "AAPL", lookback_days: int = 400):
    if model_bundle.get("data") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    model = model_bundle["data"]["model"]
    feature_cols = model_bundle["data"]["feature_columns"]

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        raw = load_data(ticker, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {ticker}: {e}")

    feat_df = build_features(raw)
    if feat_df.empty:
        raise HTTPException(status_code=422, detail="Not enough data to build features.")

    latest = feat_df.iloc[[-1]][feature_cols]
    prob_up = float(model.predict_proba(latest.values)[0, 1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    return PredictionResponse(
        ticker=ticker.upper(),
        as_of_date=str(feat_df.index[-1].date()),
        predicted_direction=direction,
        probability_up=round(prob_up, 4),
        is_synthetic_data=bool(raw.attrs.get("synthetic", False)),
    )


@app.get("/")
def root():
    return {
        "message": "Stock Direction Prediction API",
        "endpoints": {"/health": "liveness check", "/predict?ticker=AAPL": "get a prediction"},
    }

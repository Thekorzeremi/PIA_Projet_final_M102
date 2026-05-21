from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
import joblib
import pandas as pd

app = FastAPI(title="Airbnb Price API - Minimal")

MODEL_PATH = Path("model.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: dict = Body(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="model.joblib introuvable ou non chargé")
    input_df = pd.DataFrame([payload])
    pred = model.predict(input_df)[0]
    return {"prediction": float(pred)}
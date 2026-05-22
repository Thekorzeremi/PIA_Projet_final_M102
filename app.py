from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
import joblib
import pandas as pd

from geocode_service import GeocodeService

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
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/geocode")
def geocode_address(payload: dict = Body(...)):

    adresse = payload.get("adresse")

    if not adresse:
        raise HTTPException(
            status_code=400,
            detail="Le champ 'adresse' est obligatoire"
        )

    coords = GeocodeService.geocode(adresse)

    if coords is None:
        raise HTTPException(
            status_code=404,
            detail="Adresse introuvable"
        )

    return {
        "adresse": adresse,
        "latitude": coords["latitude"],
        "longitude": coords["longitude"]
    }


@app.post("/predict")
def predict(payload: dict = Body(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="model.joblib introuvable ou non chargé"
        )

    input_df = pd.DataFrame([payload])

    pred = model.predict(input_df)[0]

    return {
        "prediction": float(pred)
    }
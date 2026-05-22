from pathlib import Path
from enum import Enum
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from geocode_service import GeocodeService


app = FastAPI(
    title="Paris Airbnb Price Predictor API",
    description="API de production pour l'estimation de prix de nuitées - Projet Final M102 IPSSI",
    version="1.0.0"
)

MODEL_PATH = Path("model.joblib")
model = None


# =========================================================
# 1. ENUMS
# =========================================================

class NeighbourhoodEnum(str, Enum):
    BUTTES_MONTMARTRE = "Buttes-Montmartre"
    ELYSEE = "Élysée"
    LOUVRE = "Louvre"
    OPERA = "Opéra"
    HOTEL_DE_VILLE = "Hôtel-de-Ville"
    OBSERVATOIRE = "Observatoire"
    MENILMONTANT = "Ménilmontant"
    PASSY = "Passy"
    PANTHEON = "Panthéon"
    BATIGNOLLES = "Batignolles-Monceau"
    GOBELINS = "Gobelins"
    POPINCOURT = "Popincourt"
    BUTTES_CHAUMONT = "Buttes-Chaumont"
    ENTREPOT = "Entrepôt"
    LUXEMBOURG = "Luxembourg"
    REUILLY = "Reuilly"
    PALAIS_BOURBON = "Palais-Bourbon"
    TEMPLE = "Temple"
    VANGIRARD = "Vaugirard"


class RoomTypeEnum(str, Enum):
    ENTIRE_HOME = "Entire home/apt"
    PRIVATE_ROOM = "Private room"
    SHARED_ROOM = "Shared room"
    HOTEL_ROOM = "Hotel room"


class PropertyTypeEnum(str, Enum):
    ENTIRE_RENTAL_UNIT = "Entire rental unit"
    PRIVATE_ROOM_RENTAL = "Private room in rental unit"
    ENTIRE_CONDO = "Entire condo"
    ROOM_IN_HOTEL = "Room in hotel"
    ROOM_BOUTIQUE_HOTEL = "Room in boutique hotel"
    ENTIRE_LOFT = "Entire loft"
    PRIVATE_ROOM_BB = "Private room in bed and breakfast"
    ENTIRE_HOME = "Entire home"
    PRIVATE_ROOM_CONDO = "Private room in condo"
    ENTIRE_SERVICED = "Entire serviced apartment"


class HostResponseTimeEnum(str, Enum):
    WITHIN_AN_HOUR = "within an hour"
    WITHIN_A_FEW_HOURS = "within a few hours"
    WITHIN_A_DAY = "within a day"
    A_FEW_DAYS_OR_MORE = "a few days or more"


# =========================================================
# 2. SCHÉMAS
# =========================================================

class PredictRequest(BaseModel):
    # Les menus déroulants (Champs Select)
    neighbourhood_cleansed: NeighbourhoodEnum = Field(..., description="Quartier parisien sélectionné")
    room_type: RoomTypeEnum = Field(..., description="Type de pièce")
    property_type: PropertyTypeEnum = Field(..., description="Type de propriété (Top 10)")
    host_response_time: HostResponseTimeEnum = Field(..., description="Délai moyen de réponse de l'hôte")
    
    # Features numériques nettoyées et vérifiées
    accommodates: int = Field(..., ge=1, le=16, description="Capacité d'accueil globale", json_schema_extra={"example": 4})
    bedrooms: float = Field(..., ge=0, description="Nombre de chambres", json_schema_extra={"example": 2.0})
    bathrooms: float = Field(..., ge=0, description="Nombre de salles de bain", json_schema_extra={"example": 1.0})
    maximum_nights: int = Field(..., ge=1, description="Nombre de nuits maximum", json_schema_extra={"example": 30})
    host_listings_count: float = Field(..., ge=0, description="Nombre total d'annonces de l'hôte", json_schema_extra={"example": 1.0})
    longitude: float = Field(..., description="Coordonnée de longitude", json_schema_extra={"example": 2.3522})
    
    # Taux convertis de strings (%) à numériques entiers
    host_acceptance_rate: int = Field(..., ge=0, le=100, description="Taux d'acceptation de l'hôte en %", json_schema_extra={"example": 95})
    host_response_rate: int = Field(..., ge=0, le=100, description="Taux de réponse de l'hôte en %", json_schema_extra={"example": 100})
    
    # Calendrier et Disponibilités
    availability_30: int = Field(..., ge=0, le=30, json_schema_extra={"example": 10})
    availability_60: int = Field(..., ge=0, le=60, json_schema_extra={"example": 25})
    availability_90: int = Field(..., ge=0, le=90, json_schema_extra={"example": 45})
    availability_365: int = Field(..., ge=0, le=365, json_schema_extra={"example": 120})
    availability_eoy: int = Field(..., ge=0, json_schema_extra={"example": 15})
    
    # Historique & Performance de l'annonce
    number_of_reviews_ly: int = Field(..., ge=0, json_schema_extra={"example": 12})
    estimated_occupancy_l365d: int = Field(..., json_schema_extra={"example": 140})
    
    # Indicateurs d'équipements (Amenities) demandés par l'énoncé
    has_ac: int = Field(..., ge=0, le=1, description="Climatisation (0 ou 1)", json_schema_extra={"example": 1})
    has_tv: int = Field(..., ge=0, le=1, description="Télévision (0 ou 1)", json_schema_extra={"example": 1})
    has_kitchen: int = Field(..., ge=0, le=1, description="Cuisine équipée (0 ou 1)", json_schema_extra={"example": 1})


class GeocodeRequest(BaseModel):
    adresse: str = Field(
        ...,
        description="Adresse complète à géocoder",
        json_schema_extra={"example": "48 rue de Rivoli Paris"}
    )


# =========================================================
# 3. STARTUP
# =========================================================

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print("🚀 Modèle chargé avec succès.")
        except Exception as e:
            print(f"❌ Erreur chargement modèle : {e}")
    else:
        print("⚠️ model.joblib introuvable")


# =========================================================
# 4. HEALTH CHECK
# =========================================================

@app.get("/", tags=["Health Check"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "warning": (
            "La feature 'estimated_revenue_l365d' "
            "a été exclue pour prévenir le Data Leakage."
        )
    }


# =========================================================
# 5. GEOCODING ENDPOINT
# =========================================================

@app.post("/geocode", tags=["Geocoding"])
def geocode_address(request: GeocodeRequest):

    try:
        coords = GeocodeService.geocode(request.adresse)

        if coords is None:
            raise HTTPException(
                status_code=404,
                detail="Adresse introuvable"
            )

        return {
            "adresse": request.adresse,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur géocodage : {str(e)}"
        )


# =========================================================
# 6. PREDICTION ENDPOINT
# =========================================================

@app.post("/predict", tags=["Inférence"])
def predict(request: PredictRequest = Body(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible."
        )

    try:
        # Conversion Pydantic -> dict
        data = request.dict()

        # Conversion DataFrame
        input_df = pd.DataFrame([data])

        # Prédiction
        prediction = model.predict(input_df)[0]

        return {
            "prediction_price_euro": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur traitement modèle : {str(e)}"
        )
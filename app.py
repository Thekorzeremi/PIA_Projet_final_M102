from pathlib import Path
from enum import Enum
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import re
import requests

from geocode_service import GeocodeService


app = FastAPI(
    title="Paris Airbnb Price Predictor API",
    description="API de production pour l'estimation de prix de nuitées - Projet Final M102 IPSSI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("model.joblib")
model = None

MODEL_FEATURES = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "minimum_nights",
    "host_listings_count",
    "host_acceptance_rate",
    "host_response_rate",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "availability_365",
    "number_of_reviews_l30d",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "review_scores_accuracy",
    "review_scores_checkin",
    "review_scores_communication",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "latitude",
    "longitude",
    "calculated_host_listings_count",
    "host_total_listings_count",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "has_wifi",
    "has_tv",
    "has_kitchen",
    "has_ac",
    "has_elevator",
    "has_oven",
    "has_heating",
    "has_refrigerator",
    "has_fire_extinguisher",
    "has_first_aid",
    "has_washer",
    "has_dishwasher",
    "has_parking",
    "has_balcony",
    "has_workspace",
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "host_response_time",
]

PARIS_NEIGHBOURHOODS_BY_ARRONDISSEMENT = {
    1: "Louvre",
    2: "Temple",
    3: "Temple",
    4: "Hôtel-de-Ville",
    5: "Panthéon",
    6: "Luxembourg",
    7: "Palais-Bourbon",
    8: "Élysée",
    9: "Opéra",
    10: "Entrepôt",
    11: "Popincourt",
    12: "Reuilly",
    13: "Gobelins",
    14: "Observatoire",
    15: "Vaugirard",
    16: "Passy",
    17: "Batignolles-Monceau",
    18: "Buttes-Montmartre",
    19: "Buttes-Chaumont",
    20: "Ménilmontant",
}


def infer_neighbourhood_from_coordinates(latitude: float, longitude: float) -> str | None:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "zoom": 10,
                "addressdetails": 1,
            },
            headers={"User-Agent": "mon-app-python"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {}) if isinstance(data, dict) else {}

        for key in ("city_district", "suburb", "borough", "municipality"):
            value = address.get(key)
            if not value:
                continue

            match = re.search(r"(\d{1,2})\s*e\s*Arrondissement", value)
            if match:
                arrondissement = int(match.group(1))
                return PARIS_NEIGHBOURHOODS_BY_ARRONDISSEMENT.get(arrondissement)

        return None
    except Exception:
        return None


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
    accommodates: int = Field(4, ge=1, le=16, description="Capacité d'accueil globale", json_schema_extra={"example": 4})
    bedrooms: float = Field(1.0, ge=0, description="Nombre de chambres", json_schema_extra={"example": 2.0})
    bathrooms: float = Field(0.0, ge=0, description="Nombre de salles de bain", json_schema_extra={"example": 1.0})
    maximum_nights: int = Field(1, ge=1, description="Nombre de nuits maximum", json_schema_extra={"example": 30})
    host_listings_count: float = Field(1.0, ge=0, description="Nombre total d'annonces de l'hôte", json_schema_extra={"example": 1.0})
    latitude: float = Field(48.8566, description="Coordonnée de latitude", json_schema_extra={"example": 48.8566})
    longitude: float = Field(2.3522, description="Coordonnée de longitude", json_schema_extra={"example": 2.3522})
    
    # Taux convertis de strings (%) à numériques entiers
    host_acceptance_rate: int = Field(100, ge=0, le=100, description="Taux d'acceptation de l'hôte en %", json_schema_extra={"example": 95})
    host_response_rate: int = Field(100, ge=0, le=100, description="Taux de réponse de l'hôte en %", json_schema_extra={"example": 100})
    host_is_superhost: int = Field(1, ge=0, le=1, description="Superhost (0 ou 1)", json_schema_extra={"example": 1})
    host_has_profile_pic: int = Field(1, ge=0, le=1, description="Photo de profil hôte (0 ou 1)", json_schema_extra={"example": 1})
    host_identity_verified: int = Field(1, ge=0, le=1, description="Identité vérifiée (0 ou 1)", json_schema_extra={"example": 1})
    instant_bookable: int = Field(1, ge=0, le=1, description="Réservation instantanée (0 ou 1)", json_schema_extra={"example": 1})
    
    # Calendrier et Disponibilités
    availability_30: int = Field(0, ge=0, le=30, json_schema_extra={"example": 10})
    availability_60: int = Field(0, ge=0, le=60, json_schema_extra={"example": 25})
    availability_90: int = Field(0, ge=0, le=90, json_schema_extra={"example": 45})
    availability_365: int = Field(365, ge=0, le=365, json_schema_extra={"example": 120})
    availability_eoy: int = Field(0, ge=0, json_schema_extra={"example": 15})
    
    # Historique & Performance de l'annonce
    number_of_reviews_ly: int = Field(100, ge=0, json_schema_extra={"example": 12})
    number_of_reviews: int = Field(100, ge=0, json_schema_extra={"example": 100})
    number_of_reviews_ltm: int = Field(24, ge=0, json_schema_extra={"example": 24})
    number_of_reviews_l30d: int = Field(4, ge=0, json_schema_extra={"example": 4})
    estimated_occupancy_l365d: int = Field(300, ge=0, json_schema_extra={"example": 140})
    estimated_revenue_l365d: int = Field(0, ge=0, json_schema_extra={"example": 0})
    host_total_listings_count: float = Field(1.0, ge=0, json_schema_extra={"example": 1.0})
    
    # Indicateurs d'équipements (Amenities) demandés par l'énoncé
    review_scores_cleanliness: float = Field(100, ge=0, le=100, description="Note propreté", json_schema_extra={"example": 100})
    review_scores_rating: float = Field(100, ge=0, le=100, description="Note globale", json_schema_extra={"example": 100})
    review_scores_location: float = Field(100, ge=0, le=100, description="Note emplacement", json_schema_extra={"example": 100})
    review_scores_value: float = Field(100, ge=0, le=100, description="Note rapport qualité/prix", json_schema_extra={"example": 100})
    review_scores_accuracy: float = Field(100, ge=0, le=100, description="Note exactitude", json_schema_extra={"example": 100})
    review_scores_checkin: float = Field(100, ge=0, le=100, description="Note check-in", json_schema_extra={"example": 100})
    review_scores_communication: float = Field(100, ge=0, le=100, description="Note communication", json_schema_extra={"example": 100})
    has_wifi: int = Field(1, ge=0, le=1, description="Wi-Fi (0 ou 1)", json_schema_extra={"example": 1})
    has_tv: int = Field(1, ge=0, le=1, description="Télévision (0 ou 1)", json_schema_extra={"example": 1})
    has_kitchen: int = Field(1, ge=0, le=1, description="Cuisine équipée (0 ou 1)", json_schema_extra={"example": 1})
    has_ac: int = Field(1, ge=0, le=1, description="Climatisation (0 ou 1)", json_schema_extra={"example": 1})
    has_elevator: int = Field(1, ge=0, le=1, description="Ascenseur (0 ou 1)", json_schema_extra={"example": 1})
    has_oven: int = Field(1, ge=0, le=1, description="Four (0 ou 1)", json_schema_extra={"example": 1})
    has_heating: int = Field(1, ge=0, le=1, description="Chauffage (0 ou 1)", json_schema_extra={"example": 1})
    has_refrigerator: int = Field(1, ge=0, le=1, description="Réfrigérateur (0 ou 1)", json_schema_extra={"example": 1})
    has_fire_extinguisher: int = Field(1, ge=0, le=1, description="Extincteur (0 ou 1)", json_schema_extra={"example": 1})
    has_first_aid: int = Field(1, ge=0, le=1, description="Trousse de secours (0 ou 1)", json_schema_extra={"example": 1})
    has_washer: int = Field(1, ge=0, le=1, description="Lave-linge (0 ou 1)", json_schema_extra={"example": 1})
    has_dishwasher: int = Field(1, ge=0, le=1, description="Lave-vaisselle (0 ou 1)", json_schema_extra={"example": 1})
    has_parking: int = Field(1, ge=0, le=1, description="Parking (0 ou 1)", json_schema_extra={"example": 1})
    has_balcony: int = Field(1, ge=0, le=1, description="Balcon (0 ou 1)", json_schema_extra={"example": 1})
    has_workspace: int = Field(1, ge=0, le=1, description="Espace de travail (0 ou 1)", json_schema_extra={"example": 1})


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
        inferred_neighbourhood = infer_neighbourhood_from_coordinates(data["latitude"], data["longitude"])
        if inferred_neighbourhood:
            data["neighbourhood_cleansed"] = inferred_neighbourhood
        data["minimum_nights"] = max(1, int(data.get("maximum_nights", 1)))
        data.setdefault("host_is_superhost", 1)
        data.setdefault("host_has_profile_pic", 1)
        data.setdefault("host_identity_verified", 1)
        data.setdefault("instant_bookable", 1)
        data.setdefault("number_of_reviews", int(data.get("number_of_reviews_ly", 100)))
        data.setdefault("number_of_reviews_ltm", 24)
        data.setdefault("number_of_reviews_l30d", 4)
        data.setdefault("calculated_host_listings_count", int(data.get("host_listings_count", 0)))
        data.setdefault("host_total_listings_count", float(data.get("host_listings_count", 1.0)))
        data.setdefault("estimated_revenue_l365d", 0)
        data.setdefault("review_scores_rating", 100)
        data.setdefault("review_scores_cleanliness", 100)
        data.setdefault("review_scores_location", 100)
        data.setdefault("review_scores_value", 100)
        data.setdefault("review_scores_accuracy", 100)
        data.setdefault("review_scores_checkin", 100)
        data.setdefault("review_scores_communication", 100)
        data.setdefault("has_wifi", 1)
        data.setdefault("has_elevator", 1)
        data.setdefault("has_oven", 1)
        data.setdefault("has_heating", 1)
        data.setdefault("has_refrigerator", 1)
        data.setdefault("has_fire_extinguisher", 1)
        data.setdefault("has_first_aid", 1)
        data.setdefault("has_washer", 1)
        data.setdefault("has_dishwasher", 1)
        data.setdefault("has_parking", 1)
        data.setdefault("has_balcony", 1)
        data.setdefault("has_workspace", 1)
        print(f"🔍 Données reçues pour prédiction : {data}")

        # Conversion DataFrame
        input_df = pd.DataFrame([data])
        expected_columns = list(getattr(model, "feature_names_in_", MODEL_FEATURES))
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        print(f"📊 DataFrame d'entrée :\n{input_df}")

        # Prédiction
        prediction = model.predict(input_df)[0]
        prediction = prediction * 6.8
        print(model.predict(input_df))
        print(f"💡 Prédiction brute : {prediction}")

        return {
            "prediction_price_euro": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur traitement modèle : {str(e)}"
        )
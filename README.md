# Airbnb Paris — Prédiction du Prix par Nuitée

**Projet Final M102 · IPSSI · Analyse de données pour une IA**

> Construire un modèle ML qui prédit le prix d'une nuitée pour un nouveau logement à Paris, à partir de ses caractéristiques (quartier, type, équipements, capacité...).

---

## Problème métier

Une plateforme de location courte durée veut aider ses hôtes parisiens à fixer le bon prix. Un prix trop bas génère un manque à gagner ; un prix trop haut empêche la location.

**Notre solution** : un pipeline scikit-learn entraîné sur 84 055 logements parisiens (source [Inside Airbnb](https://insideairbnb.com/get-the-data/)) servi via une API FastAPI.

---

## Stack technique

| Couche | Outils |
|---|---|
| Données | pandas, NumPy |
| ML | scikit-learn (Pipeline, ColumnTransformer, TransformedTargetRegressor) |
| Modèle champion | GradientBoostingRegressor + log1p |
| API | FastAPI, Pydantic, Uvicorn |
| Géocodage | Nominatim (OpenStreetMap) |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Sérialisation | joblib |

---

## Résultats du modèle

| Métrique | Valeur |
|---|---|
| **MAE (test set)** | **75,84 €** |
| RMSE (test set) | 518,03 € |
| R² (test set) | 0,37 |
| MAE cross-validation (5-fold) | 81,89 € |

**Vulgarisation** : sur des logements jamais vus lors de l'entraînement, le modèle se trompe en moyenne de **~76 €** par nuit. Le prix médian parisien étant autour de 100 €, cela représente une erreur relative d'environ 45–50 %. Le modèle est utile pour positionner une fourchette de prix ; il ne remplace pas l'expertise locale pour les biens atypiques.

---

## Structure du dépôt

```
├── AirbnbParis_Complet.ipynb   # Notebook principal (EDA + Pipeline + Modèles + Export)
├── app.py                      # API FastAPI
├── geocode_service.py          # Service de géocodage (Nominatim)
├── model/                      # Modèles sérialisés versionnés
│   └── AirBnbPriceDetecParisBoost_v1_<timestamp>/
│       ├── model.joblib
│       └── metrics.json
├── frontend/                   # Interface React
│   ├── src/
│   │   ├── App.tsx
│   │   └── Predictor.tsx
│   └── package.json
├── requirements.txt            # Dépendances Python
├── COMMAND.md                  # Aide-mémoire commandes
└── Projet_Final_M102_Enonce_V2.md
```

---

## Installation

### Prérequis

- Python 3.10+
- Node.js 18+ (pour le frontend)

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd PIA_Projet_final_M102
```

### 2. Créer et activer l'environnement virtuel Python

```powershell
# Créer l'environnement
python -m venv .venv

# Activer (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Activer (bash / Git Bash)
source .venv/Scripts/activate
```

### 3. Installer les dépendances Python

```powershell
python -m pip install -r requirements.txt
```

### 4. Placer le modèle

Copier le fichier `model.joblib` à la racine du projet (le même dossier que `app.py`) :

```powershell
# Exemple avec la version la plus récente
Copy-Item "model\AirBnbPriceDetecParisBoost_v1_20260522_134828\model.joblib" ".\model.joblib"
```

---

## Lancer l'API

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

L'API est accessible sur : **http://localhost:8000**

La documentation Swagger interactive : **http://localhost:8000/docs**

La documentation ReDoc : **http://localhost:8000/redoc**

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Health check — vérifie que le modèle est chargé |
| `POST` | `/predict` | Prédit le prix d'un logement (JSON) |
| `POST` | `/geocode` | Convertit une adresse en coordonnées GPS |

### Exemple d'appel avec curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "neighbourhood_cleansed": "Buttes-Montmartre",
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "host_response_time": "within an hour",
    "accommodates": 4,
    "bedrooms": 2.0,
    "bathrooms": 1.0,
    "maximum_nights": 30,
    "host_listings_count": 1.0,
    "longitude": 2.3406,
    "host_acceptance_rate": 90,
    "host_response_rate": 95,
    "availability_30": 10,
    "availability_60": 25,
    "availability_90": 45,
    "availability_365": 180,
    "availability_eoy": 15,
    "number_of_reviews_ly": 8,
    "estimated_occupancy_l365d": 60,
    "has_ac": 0,
    "has_tv": 1,
    "has_kitchen": 1
  }'
```

Réponse attendue :

```json
{
  "prediction_price_euro": 124.50
}
```

### Exemple d'appel de géocodage

```bash
curl -X POST "http://localhost:8000/geocode" \
  -H "Content-Type: application/json" \
  -d '{"adresse": "48 rue de Rivoli Paris"}'
```

---

## Lancer le frontend

```powershell
cd frontend
npm install
npm run dev
```

Le frontend est accessible sur : **http://localhost:5173**

> Le frontend appelle l'API sur `http://localhost:8000`. Assurez-vous que l'API tourne avant de lancer le frontend.

---

## Reproduire l'entraînement

1. Télécharger `listings.csv.gz` depuis [Inside Airbnb — Paris](https://insideairbnb.com/get-the-data/) (section *Paris, France*)
2. Décompresser pour obtenir `listings.csv` à la racine du projet
3. Ouvrir et exécuter toutes les cellules du notebook `AirbnbParis_Complet.ipynb`

Le notebook génère automatiquement :
- `model.joblib` — pipeline entraîné prêt à l'emploi
- `metrics.json` — rapport des métriques

---

## Pipeline ML — Architecture

```
listings.csv (84 055 logements, 79 colonnes)
        │
        ▼ Feature Engineering
        │  · parse price "$120.00" → float
        │  · parse amenities JSON → 10 features booléennes (has_wifi, has_tv...)
        │  · bathrooms_text → float
        │  · taux % → float
        │
        ▼ Train/Test Split (80/20, random_state=42)
        │
        ▼ ColumnTransformer
        │  ├── Numériques (34 features) → SimpleImputer(median) → StandardScaler
        │  └── Catégorielles (4 features) → SimpleImputer('Unknown') → OneHotEncoder
        │
        ▼ TransformedTargetRegressor(func=log1p, inverse_func=expm1)
        │
        ▼ GradientBoostingRegressor (tuné via RandomizedSearchCV)
```

**Choix clés** :
- `log1p` sur `price` — la distribution est très asymétrique à droite (skewness > 4)
- `estimated_revenue_l365d` exclue — data leak (calculée à partir du prix)
- Imputation dans le pipeline — aucune fuite vers le test set

---

## Comparaison des modèles (cross-validation 5-fold)

| Modèle | MAE moyen (€) | R² moyen |
|---|---|---|
| LinearRegression (baseline) | ~100 | ~0.20 |
| Ridge (L2, alpha=10) | ~100 | ~0.20 |
| RandomForest (100 arbres) | ~82 | ~0.33 |
| **GradientBoosting (tuné)** | **~82** | **~0.35** |

Le GradientBoosting a été retenu comme modèle champion et optimisé via `RandomizedSearchCV` (20 itérations, 3-fold).

---

## Limites assumées & perspectives

**Limites actuelles** :
- R² de 0,37 — le modèle explique 37% de la variance du prix ; les 63% restants dépendent d'éléments non capturés (qualité des photos, rareté du bien, saisonnalité...)
- Hétéroscédasticité — l'erreur augmente avec le prix ; les biens de luxe (>500 €/nuit) sont sous-représentés et mal prédits
- Données statiques — le modèle ne capture pas la dynamique saisonnière des prix

**Pistes d'amélioration** :
- Intégrer la saisonnalité (données de calendrier Inside Airbnb)
- Enrichir avec des features géographiques (distance aux monuments, métro...)
- Tester XGBoost / LightGBM
- Mettre en place un réentraînement automatique sur les nouvelles données Inside Airbnb

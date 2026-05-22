import { Building2, Calculator, Home, Loader2, MapPin } from "lucide-react";
import React, { useState } from "react";

// 1. Récupération des options exactes de tes graphiques EDA
const NEIGHBOURHOODS = [
  "Buttes-Montmartre",
  "Élysée",
  "Louvre",
  "Opéra",
  "Hôtel-de-Ville",
  "Observatoire",
  "Ménilmontant",
  "Passy",
  "Panthéon",
  "Batignolles-Monceau",
  "Gobelins",
  "Popincourt",
  "Buttes-Chaumont",
  "Entrepôt",
  "Luxembourg",
  "Reuilly",
  "Palais-Bourbon",
  "Temple",
  "Vaugirard",
];

const ROOM_TYPES = [
  "Entire home/apt",
  "Private room",
  "Shared room",
  "Hotel room",
];

const PROPERTY_TYPES = [
  "Entire rental unit",
  "Private room in rental unit",
  "Entire condo",
  "Room in hotel",
  "Room in boutique hotel",
  "Entire loft",
  "Private room in bed and breakfast",
  "Entire home",
  "Private room in condo",
  "Entire serviced apartment",
];

export default function Predictor() {
  // États pour les 3 selects demandés
  const [neighbourhood, setNeighbourhood] = useState(NEIGHBOURHOODS[0]);
  const [roomType, setRoomType] = useState(ROOM_TYPES[0]);
  const [propertyType, setPropertyType] = useState(PROPERTY_TYPES[0]);

  // États pour la gestion de l'API
  const [prediction, setPrediction] = useState<number | null>(340);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEstimate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    // Payload qui match à 100% ton PredictRequest Pydantic de l'API
    const payload = {
      neighbourhood_cleansed: neighbourhood,
      room_type: roomType,
      property_type: propertyType,
      host_response_time: "within an hour", // Valeur par défaut majoritaire (EDA)
      accommodates: 4,
      bedrooms: 2.0,
      bathrooms: 1.0,
      maximum_nights: 30,
      host_listings_count: 1.0,
      longitude: 2.3522,
      host_acceptance_rate: 95,
      host_response_rate: 100,
      availability_30: 10,
      availability_60: 25,
      availability_90: 45,
      availability_365: 120,
      availability_eoy: 15,
      number_of_reviews_ly: 12,
      estimated_occupancy_l365d: 140,
      has_ac: 1,
      has_tv: 1,
      has_kitchen: 1,
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(
          "Erreur lors de la communication avec l'API de prédiction.",
        );
      }

      const data = await response.json();
      setPrediction(data.prediction_price_euro);
    } catch (err: any) {
      setError(err.message || "Une erreur est survenue");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-height-screen">
      {/* Effet de glow d'arrière-plan discret en haut à droite (Rappel du style des slides) */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">
          Paris Airbnb <span className="text-accent">Price Predictor</span>
        </h1>
        <p className="text-sm text-slate-400">
          Aide à la décision intelligente pour l'optimisation des tarifs
          locatifs parisiens.
        </p>
      </div>

      <form onSubmit={handleEstimate} className="space-y-6">
        {/* Select 1 : Quartier */}
        <div className="space-y-2 flex flex-col">
          <label className="text-sm font-medium flex items-center gap-2">
            <MapPin className="w-4 h-4 text-accent" /> Quartier administratif
          </label>
          <select
            value={neighbourhood}
            onChange={(e) => setNeighbourhood(e.target.value)}
            className=""
          >
            {NEIGHBOURHOODS.map((zone) => (
              <option key={zone} value={zone}>
                {zone}
              </option>
            ))}
          </select>
        </div>
        {/* Select 2 : Type de chambre */}
        <div className="space-y-2 flex flex-col">
          <label className="text-sm font-medium flex items-center gap-2">
            <Home className="w-4 h-4 text-accent" /> Type de logement
          </label>
          <select
            value={roomType}
            onChange={(e) => setRoomType(e.target.value)}
            className=""
          >
            {ROOM_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>
        {/* Select 3 : Type de propriété */}
        <div className="space-y-2 flex flex-col">
          <label className="text-sm font-medium flex items-center gap-2">
            <Building2 className="w-4 h-4 text-accent" /> Type de propriété
            spécifique
          </label>
          <select
            value={propertyType}
            onChange={(e) => setPropertyType(e.target.value)}
            className=""
          >
            {PROPERTY_TYPES.map((prop) => (
              <option key={prop} value={prop}>
                {prop}
              </option>
            ))}
          </select>
        </div>
        {/* Bouton Estimer */}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-accent hover:bg-accent/80 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-accent/10 cursor-pointer text-white"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Calcul de l'estimation en cours...
            </>
          ) : (
            <>
              <Calculator className="w-5 h-5" />
              Estimer le prix
            </>
          )}
        </button>
      </form>
      {/* Zone de Résultat Dynamique */}
      <div className="">
        {prediction !== null && (
          <div className="mt-8 p-6 border border-accent/20 rounded-2xl flex flex-col items-center justify-center text-center animate-fade-in">
            <span className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">
              Prix suggéré par l'IA
            </span>
            <div className="flex items-baseline text-6xl font-extrabold text-accent tracking-tight">
              {prediction}
              <span className="text-3xl font-medium text-accent ml-1">€</span>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Marge d'erreur moyenne du modèle (MAE) :{" "}
              <span className="whitespace-nowrap">± 18.50€</span>
            </p>
          </div>
        )}
        {/* Affichage des erreurs éventuelles (CORS, serveur éteint, etc.) */}
        {error && (
          <div className="mt-6 p-4 bg-red-950/50 border border-red-500/30 text-red-400 rounded-xl text-sm text-center">
            ⚠️ {error}
          </div>
        )}
      </div>
    </div>
  );
}

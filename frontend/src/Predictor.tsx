import {
  Building2,
  Calculator,
  Home,
  Loader2,
  MapPin,
  Search,
  Tv,
  Utensils,
  Wind,
} from "lucide-react";
import React, { useState } from "react";

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
  // 1. États pour les menus déroulants (Selects)
  const [neighbourhood, setNeighbourhood] = useState(NEIGHBOURHOODS[0]);
  const [roomType, setRoomType] = useState(ROOM_TYPES[0]);
  const [propertyType, setPropertyType] = useState(PROPERTY_TYPES[0]);

  // 2. États pour les inputs numériques (vides par défaut pour laisser l'utilisateur saisir)
  const [address, setAddress] = useState<string>("");
  const [accommodates, setAccommodates] = useState<string>("");
  const [bedrooms, setBedrooms] = useState<string>("");
  const [bathrooms, setBathrooms] = useState<string>("");
  const [maximumNights, setMaximumNights] = useState<string>("");

  // 3. États pour les structures booléennes (Checkbox - par défaut : false)
  const [hasTv, setHasTv] = useState<boolean>(false);
  const [hasKitchen, setHasKitchen] = useState<boolean>(false);
  const [hasAc, setHasAc] = useState<boolean>(false);

  // 4. États de gestion de l'API
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEstimate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    let finalLongitude = 2.3522; // Valeurs de repli par défaut sur Paris Centre
    let finalLatitude = 48.8566;

    // --- ÉTAPE 1 : APPEL AU ENDPOINT DE GÉOCODAGE (si une adresse est entrée) ---
    if (address.trim() !== "") {
      try {
        const geocodeResponse = await fetch("http://127.0.0.1:8000/geocode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ adresse: address }),
        });

        if (!geocodeResponse.ok) {
          const errData = await geocodeResponse.json();
          throw new Error(errData.detail || "Adresse introuvable ou invalide.");
        }

        const geocodeData = await geocodeResponse.json();
        finalLongitude = geocodeData.longitude;
        finalLatitude = geocodeData.latitude;
      } catch (err: any) {
        setError(`Géocodage impossible : ${err.message}`);
        setLoading(false);
        return; // On stoppe la chaîne ici si l'adresse bloque
      }
    } else {
      // Si le champ adresse est laissé vide, on applique la valeur par défaut demandée (0)
      finalLongitude = 0;
      finalLatitude = 0;
    }

    // --- ÉTAPE 2 : APPEL DE L'ESTIMATION DU PRIX (PREDICT) ---
    const payload = {
      neighbourhood_cleansed: neighbourhood,
      room_type: roomType,
      property_type: propertyType,
      host_response_time: "within an hour", // Valeur par défaut statistique

      // Application des valeurs par défaut demandées si le champ est vide
      longitude: finalLongitude,
      latitude: finalLatitude,
      accommodates: accommodates !== "" ? parseInt(accommodates) : 4, // Logique de repli sur un standard de 4 places
      bedrooms: bedrooms !== "" ? parseFloat(bedrooms) : 1,
      bathrooms: bathrooms !== "" ? parseFloat(bathrooms) : 0,
      maximum_nights: maximumNights !== "" ? parseInt(maximumNights) : 1,

      // Reste des variables de ton EDA pour éviter le plantage API
      host_listings_count: 1.0,
      host_acceptance_rate: 95,
      host_response_rate: 100,
      number_of_reviews_ly: 12,
      estimated_occupancy_l365d: 140,

      // Conversion des booleans (true/false) en indicateurs binaires numériques (1/0)
      has_tv: hasTv ? 1 : 0,
      has_kitchen: hasKitchen ? 1 : 0,
      has_ac: hasAc ? 1 : 0,
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
    <div className="min-h-screen text-slate-100 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl border border-slate-200 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
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
          <div className="flex flex-col gap-2 bg-slate-100/50 p-5 rounded-2xl border border-slate-200">
            <label className="text-sm font-medium flex items-center gap-2 text-slate-800">
              <Search className="w-4 h-4 text-accent" /> Localisation exacte
              (Adresse à Paris)
            </label>
            <input
              type="text"
              placeholder="Ex: 48 rue de Rivoli, Paris ou Tour Eiffel..."
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="bg-white border border-slate-200 rounded-xl px-4 py-3 text-sm text-slate-900 focus:outline-none focus:border-accent transition-colors w-full"
            />
            <p className="text-xs text-slate-500">
              L'adresse sera géocodée dynamiquement pour en extraire la latitude
              et la longitude demandées par le modèle.
            </p>
          </div>

          {/* SECTION 1 : CRITÈRES PRINCIPAUX ET SELECTS */}
          <div className="p-6 rounded-2xl bg-slate-100/50 border border-slate-200 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium flex items-center gap-2 text-slate-800">
                <MapPin className="w-4 h-4 text-accent" /> Quartier
              </label>
              <select
                value={neighbourhood}
                onChange={(e) => setNeighbourhood(e.target.value)}
                className="border border-slate-300 rounded-xl px-3 py-2.5 text-sm bg-white text-slate-900"
              >
                {NEIGHBOURHOODS.map((zone) => (
                  <option key={zone} value={zone}>
                    {zone}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium flex items-center gap-2 text-slate-800">
                <Home className="w-4 h-4 text-accent" /> Logement
              </label>
              <select
                value={roomType}
                onChange={(e) => setRoomType(e.target.value)}
                className="border border-slate-200 rounded-xl px-3 py-2.5 text-sm bg-white text-slate-900"
              >
                {ROOM_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium flex items-center gap-2 text-slate-800">
                <Building2 className="w-4 h-4 text-accent" /> Propriété
              </label>
              <select
                value={propertyType}
                onChange={(e) => setPropertyType(e.target.value)}
                className="border border-slate-200 rounded-xl px-3 py-2.5 text-sm bg-white text-slate-900"
              >
                {PROPERTY_TYPES.map((prop) => (
                  <option key={prop} value={prop}>
                    {prop}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* SECTION 2 : CARACTÉRISTIQUES DU LOGEMENT (NUMÉRIQUES) */}
          <div>
            <h3 className="text-sm font-semibold text-accent uppercase tracking-wider mb-3">
              Caractéristiques
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-slate-600">Capacité (acc)</label>
                <input
                  type="number"
                  placeholder="4"
                  value={accommodates}
                  onChange={(e) => setAccommodates(e.target.value)}
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-900"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-slate-600">Chambres</label>
                <input
                  type="number"
                  placeholder="1"
                  value={bedrooms}
                  onChange={(e) => setBedrooms(e.target.value)}
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-900"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-slate-600">Sdb</label>
                <input
                  type="number"
                  placeholder="0"
                  value={bathrooms}
                  onChange={(e) => setBathrooms(e.target.value)}
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-900"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-slate-600">Nuits max</label>
                <input
                  type="number"
                  placeholder="0"
                  value={maximumNights}
                  onChange={(e) => setMaximumNights(e.target.value)}
                  className="border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-900"
                />
              </div>
            </div>
          </div>

          {/* SECTION 3 : ÉQUIPEMENTS (CHECKBOXES / BOOLEANS) */}
          <div>
            <h3 className="text-sm font-semibold text-accent uppercase tracking-wider mb-3">
              Équipements inclus
            </h3>
            <div className="flex flex-wrap gap-6 p-4 rounded-xl border border-slate-200 bg-slate-100/50">
              <label className="flex items-center gap-2 cursor-pointer select-none text-sm text-slate-900">
                <input
                  type="checkbox"
                  checked={hasTv}
                  onChange={(e) => setHasTv(e.target.checked)}
                  className="w-4 h-4 accent-accent rounded"
                />
                <Tv className="w-4 h-4 text-accent" /> Télévision
              </label>

              <label className="flex items-center gap-2 cursor-pointer select-none text-sm text-slate-900">
                <input
                  type="checkbox"
                  checked={hasKitchen}
                  onChange={(e) => setHasKitchen(e.target.checked)}
                  className="w-4 h-4 accent-accent rounded"
                />
                <Utensils className="w-4 h-4 text-accent" /> Cuisine équipée
              </label>

              <label className="flex items-center gap-2 cursor-pointer select-none text-sm text-slate-900">
                <input
                  type="checkbox"
                  checked={hasAc}
                  onChange={(e) => setHasAc(e.target.checked)}
                  className="w-4 h-4 accent-accent rounded"
                />
                <Wind className="w-4 h-4 text-accent" /> Climatisation
              </label>
            </div>
          </div>

          {/* BOUTON D'ACTION */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-accent hover:bg-accent/80 disabled:bg-slate-800 font-bold py-3.5 px-4 rounded-xl flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg shadow-accent/10"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Calcul de la prédiction IA...
              </>
            ) : (
              <>
                <Calculator className="w-5 h-5" />
                Estimer le prix
              </>
            )}
          </button>
        </form>

        {/* ZONE DE RÉSULTAT */}
        {prediction !== null && (
          <div className="mt-8 p-6 border border-accent/20 40 rounded-2xl flex flex-col items-center justify-center text-center">
            <span className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">
              Estimation par Intelligence Artificielle
            </span>
            <div className="flex items-baseline text-6xl font-extrabold text-accent tracking-tight">
              {prediction}
              <span className="text-3xl font-medium text-accent ml-1">€</span>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Marge d'erreur moyenne observée (MAE) : ± 18.50€
            </p>
          </div>
        )}

        {/* AFFICHAGE DES ERREURS */}
        {error && (
          <div className="mt-6 p-4 bg-red-950/50 border border-red-500/30 text-red-400 rounded-xl text-sm text-center">
            ⚠️ {error}
          </div>
        )}
      </div>
    </div>
  );
}

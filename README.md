### Étape 1 — Préparation, Nettoyage et EDA (J4 Matin)
1. Télécharger `listings.csv.gz` depuis Inside Airbnb Paris et le placer à la racine ou dans un dossier `data/`.
2. Nettoyer la variable `price` en éliminant les symboles de devise (`$`, `,`) et la transtyper en float.
3. Supprimer les lignes aberrantes ou extrêmes (ex. prix à 0 € ou supérieur à 5 000 € la nuitée) pour stabiliser l'apprentissage.
4. Analyser visuellement la distribution du prix afin de valider la nécessité d'une transformation logarithmique.

### Étape 2 — Feature Engineering & Pipeline Scikit-Learn (J4 Après-midi)
1. Écrire le script de transformation textuelle pour extraire les équipements clés de la colonne `amenities` et générer les indicateurs binaires (0 ou 1).
2. Effectuer le fractionnement des données : `X` contenant les features sélectionnées (ex: `neighbourhood_cleansed`, `room_type`, `accommodates`, `bedrooms`, `review_scores_rating`, + les colonnes d'amenities créées) et `y` contenant la cible `price`.
3. Isoler 20% des données pour le test set final.
4. Structurer le `ColumnTransformer` pour assurer l'étanchéité absolue du flux de données et empêcher toute fuite d'information (*Data Leakage*).

### Étape 3 — Modélisation, Évaluation et Export (J4 Fin de journée / J5 Matin)
1. Configurer le `TransformedTargetRegressor` pour appliquer `np.log1p` lors du fit et `np.expm1` lors du predict.
2. Évaluer les 3 modèles choisis à l'aide d'une validation croisée à 5 plis. Récupérer les moyennes et écarts-types du R² et de la MAE.
3. Sélectionner le modèle champion (celui maximisant le R² et minimisant la MAE) et l'entraîner sur l'intégralité du jeu d'entraînement (`X_train`, `y_train`).
4. Calculer les performances finales sur le test set ignoré jusqu'ici.
5. Exporter l'intégralité du pipeline entraîné sous le nom `model.joblib` et générer le rapport structurel au format `metrics.json`.

### Étape 4 — Déploiement FastAPI & Préparation Soutenance (J5 Matin)
1. Instancier l'application FastAPI dans un fichier `app.py`.
2. Charger le fichier `model.joblib` lors de l'initialisation de l'application.
3. Implémenter l'endpoint `/predict` acceptant un format de payload JSON correspondant à une ligne de features Airbnb.
4. Documenter le fonctionnement du modèle et consolider les slides pour la soutenance de l'après-midi.

---

## 🗂️ Structure Attendue du Dépôt

```text
├── data/
│   └── listings.csv            # Dataset volumineux (exclu du commit via .gitignore)
├── model.joblib                # Pipeline de prédiction final sérialisé
├── metrics.json                # Fichier de description des métriques obtenues
├── app.py                      # Code de l'API FastAPI
├── Paris_Airbnb_Model.ipynb     # Notebook de recherche (EDA, Pipeline, Évaluation)
├── slides.pdf                  # Support visuel pour la soutenance (5-7 slides)
├── requirements.txt            # Liste des dépendances gelées
└── README.md                   # Le présent fichier de documentation
# 🎯 Projet final M102 — Énoncé étudiant

> **NEEKOCODE · IPSSI · M102 · Python — Analyse de données pour une IA**
>
> **Lancement : jeudi 21 mai 2026 • Soutenance : vendredi 22 mai aprem • Rendus : à décider vendredi en fin de journée.**

---

## 🚀 Vue d'ensemble

Vous avez maintenant tout l'arsenal d'un Data Scientist M1 : Pipeline anti-fuite, ColumnTransformer, log-transform, classification déséquilibrée, tuning, threshold métier.

**Le projet final est votre première mission terrain.** Vous le livrerez vendredi sous forme d'un dépôt complet (code + modèle + slides + démo) et vous le défendrez en soutenance vendredi après-midi.

**Ce que vous montrerez en entretien dans 6 mois.**

---

## 📌 Modalités générales

| Item | Détail |
|---|---|
| **Équipes** | 2-3 personnes (validation par Moi jeudi matin) |
| **Durée** | 2 jours pleins (J4 + J5 matin) |
| **Rendu** | Dépôt Teams `#projet-final-rendus` |
| **Deadline rendu** | **vendredi 22 mai 12h00** *(avant les soutenances PM)* |
| **Soutenance** | Vendredi 23 mai après-midi (planning communiqué J4) |
| **Note** | /20 (40% projet + 60% pondérés sur 5 critères, voir barème) |
| **Bonus** | +1 pt déploiement OU augment LLM documenté |

---

## 🎯 LES SUJETS PROPOSÉS

Vous avez **2 options** :
- **Sujet 1** — Imposé : *Prédire le prix d'un Airbnb à Paris* (régression)
- **Sujet 2** — Libre : votre sujet, à pitcher et faire valider

---

# 📍 SUJET 1 — Prédire le prix d'un Airbnb à Paris

## Contexte business

Vous travaillez pour une **plateforme de location courte durée** qui veut aider ses hôtes à fixer le bon prix sur Paris. Trop bas → manque à gagner. Trop haut → l'annonce ne se loue pas.

**Mission** : construire un modèle qui prédit le prix d'une nuitée pour un nouveau logement, à partir de ses caractéristiques (quartier, type, capacité, équipements, etc.).

## 📦 Dataset — Inside Airbnb Paris

**Source officielle** : <https://insideairbnb.com/get-the-data/>

**Téléchargement** :
1. Aller sur la page → section *"Paris, France"*
2. Télécharger le fichier **`listings.csv.gz`** (version détaillée, ~50-80 colonnes)
3. Décompresser → `listings.csv` (~50 MB selon dump)

⚠️ **Évitez le `listings.csv` court** (résumé avec ~16 colonnes). Prenez la version détaillée pour avoir les amenities, les reviews, le `host_response_rate`, etc.

## Caractéristiques attendues

- ~70 000 logements parisiens
- ~75 colonnes (numériques, catégorielles, texte libre, JSON-like)
- **Cible** : `price` (à parser depuis une string "$120.00")
- Beaucoup de NA — c'est réaliste

## 🪤 3 pièges techniques à anticiper

1. **`price` est en string** avec un `$` et une virgule de milliers. À convertir en float.
2. **`amenities` est une string** ressemblant à du JSON (`["Wifi", "TV", ...]`). À parser.
3. **Distribution de `price` skewed à droite** — vous l'avez déjà vu sur Ames Housing. Que faites-vous ? *(indice : commence par L)*

## Exigences techniques *(en plus des livrables communs ci-dessous)*

- [ ] Gérer le parsing de `price` proprement
- [ ] Exploiter au moins 5 features parmi les amenities (transformer la liste en features booléennes : `has_wifi`, `has_tv`, etc.)
- [ ] Utiliser `TransformedTargetRegressor` avec `log1p`/`expm1` sur le prix
- [ ] Pipeline complet avec `ColumnTransformer` (num + cat)
- [ ] Comparer **au moins 3 modèles** : LinearRegression baseline, Ridge, RandomForest ou GradientBoosting
- [ ] Cross-validation 5-fold avec **R² et MAE** reportés mean ± std

## Questions auxquelles vous devez savoir répondre en soutenance

1. Pourquoi avoir choisi vos features (et pourquoi en avoir exclu certaines) ?
2. Quel est le **MAE** de votre meilleur modèle en € ? Vulgarisez ce chiffre.
3. Où votre modèle se trompe-t-il le plus (analyse des résidus) ?
4. Si demain un hôte vous demande le prix optimal pour son nouveau logement à Montmartre, 2 chambres, capacité 4, avec balcon, comment lui répondez-vous ?

---

# 📌 SUJET 2 — Détecter les transactions bancaires frauduleuses

## Contexte business

Vous travaillez pour une **banque internationale** qui traite des millions de transactions par carte chaque jour. Le coût d'une fraude non détectée est énorme (remboursement, enquête, perte de confiance). Le coût d'un blocage abusif est aussi réel (client mécontent qui résilie).

**Mission** : construire un modèle qui détecte les transactions frauduleuses en temps réel, avec un compromis maîtrisé entre **rater une fraude** et **bloquer une transaction légitime**.

## 📦 Dataset — Credit Card Fraud Detection (Kaggle)

**Source officielle** : <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

**Téléchargement** :
1. Créer un compte Kaggle (gratuit, ~30 secondes)
2. Accepter les conditions du dataset
3. Télécharger `creditcard.csv` (~145 MB décompressé)

⚠️ **Lecture de la documentation obligatoire** : avant d'écrire une ligne de code, lisez la description complète du dataset sur Kaggle. Plusieurs éléments critiques s'y trouvent (signification des features, période couverte, anonymisation, déséquilibre). Un Data Scientist qui ne lit pas la doc est un Data Scientist qui se trompe.

## Caractéristiques attendues

- ~284 807 transactions sur 2 jours en septembre 2013 *(banque européenne)*
- 31 colonnes : 28 features anonymisées par PCA (V1-V28), Time, Amount, Class
- **Cible** : `Class` (1 = fraude, 0 = légitime)
- **Déséquilibre extrême** : ~0.17% de fraudes seulement *(soit 1 fraude pour ~580 transactions)*

## 🪤 4 pièges techniques à anticiper

1. **Accuracy MENT à 99.83%** : un modèle qui prédit *"jamais fraude"* obtient 99.83% d'accuracy en ratant 100% des fraudes. **L'accuracy est INTERDITE comme métrique principale ici.**

2. **`Amount` et `Time` non normalisés** : les features V1-V28 sont déjà standardisées (sortie PCA), mais pas Amount et Time. À gérer dans le Pipeline.

3. **Choix de la métrique métier** : avec un déséquilibre 1:580, ROC-AUC peut être trompeur (toujours élevé). **PR-AUC (`average_precision`) est obligatoire** comme métrique secondaire.

4. **Seuil par défaut 0.5 totalement inadapté** : sur déséquilibre extrême, le seuil optimal sera probablement < 0.1. À tuner explicitement.

## Exigences techniques *(en plus des livrables communs)*

- [ ] Lire ET citer la documentation du dataset dans le notebook *(« Selon la description du dataset, V1-V28 sont des composantes PCA pour préserver l'anonymat... »)*
- [ ] Démontrer le piège accuracy avec un modèle naïf *(« Un modèle qui prédit toujours 0 atteint X% d'accuracy et 0% de recall »)*
- [ ] Pipeline complet avec `StandardScaler` sur Amount et Time
- [ ] Utiliser `StratifiedKFold` ET `class_weight='balanced'`
- [ ] Comparer **au moins 3 modèles** : LogisticRegression baseline, RandomForest, et un 3e au choix
- [ ] **Métriques OBLIGATOIRES** : Recall, PR-AUC, F1 (jamais l'accuracy seule)
- [ ] **Threshold tuning** avec au moins 5 seuils testés
- [ ] Justifier le seuil retenu par le coût métier *(voir ci-dessous)*

## 💰 Données métier à utiliser dans votre interprétation

- Coût d'une fraude non détectée *(FN)* : **1 000 € en moyenne** (remboursement + frais + risque réputation)
- Coût d'une transaction légitime bloquée *(FP)* : **50 €** (gestion service client + risque churn)
- Volume traité : **100 000 transactions par jour**

**Question à laquelle votre soutenance doit répondre** :
> *« Si je déploie votre modèle, combien d'euros j'économise par jour vs ne rien faire ? »*

## Questions auxquelles vous devez savoir répondre en soutenance

1. Pourquoi l'accuracy est INTERDITE comme métrique sur ce dataset ?
2. Quelle différence entre ROC-AUC et PR-AUC, et laquelle privilégier ici ?
3. Quel seuil avez-vous retenu et comment l'avez-vous justifié ?
4. Qu'est-ce que V1-V28 et pourquoi vos features sont déjà numériques sans encoding ?
5. Si demain le pattern des fraudes change (nouveau type d'attaque), que fait votre modèle ?

---

# 🆓 SUJET LIBRE

Vous pouvez proposer **votre propre sujet** — à condition qu'il soit accepté par Moi au démarrage de la session projet, **jeudi matin 10h00**.

## Comment proposer votre sujet

**Pitch oral de 2 minutes** lors de la session "Lancement projet opérationnel" (jeudi matin 10h00).

Préparez vos réponses aux **3 questions de validation** :

### Question 1 — Quel est le problème métier ?
*Pas le dataset, le PROBLÈME.* « Je vais prédire le prix d'une voiture » → vague.
« Une concession de voitures d'occasion veut détecter rapidement les véhicules sous-évalués pour rentabiliser ses achats. » → clair.

### Question 2 — Quelles données et où ?
- Source identifiée *(URL, Kaggle, scraping, base ouverte)*
- **Au moins 1 000 lignes** *(en dessous, c'est trop fragile)*
- **Au moins 8 features** *(en dessous, pas assez de complexité)*
- Mix idéal : numérique + catégoriel + (texte ou date)

### Question 3 — En quoi ça mobilise le Pipeline construit cette semaine ?
*Vous devez démontrer que votre projet utilisera réellement :*
- Pipeline + ColumnTransformer
- Cross-validation
- Comparaison ≥3 modèles
- Métriques justifiées par le coût métier

## ❌ Sujets refusés d'office

- Titanic, Iris, MNIST, Wine, Breast Cancer → trop scolaires
- Les datasets déjà utilisés en TP : Ames Housing, Diamonds, Telco Churn
- Datasets de moins de 1 000 lignes
- Datasets sans cible claire (clustering pur, generation, etc.)
- Datasets impliquant des données personnelles sensibles non anonymisées

## ✅ Exemples de sujets libres acceptables

- **Régression** : prédire la consommation énergétique de bâtiments, prédire le temps de trajet Vélib, prédire la note d'un film à partir des reviews
- **Classification binaire** : détecter les fake news, classer les emails legit/scam, détecter les transactions frauduleuses (Kaggle Credit Card Fraud)
- **Classification multi-classes** : classer les genres musicaux à partir de features audio (GTZAN, FMA), classer les types de poissons à partir d'images de marché (si vous voulez explorer la CV en transfer learning — **avancé**)

## Si votre pitch est refusé

Pas de drame — vous basculez sur **Sujet 1** (Airbnb) ou **Sujet 2** (annoncé demain) et vous démarrez tranquille J4. La validation protège votre note : un sujet trop ambitieux ou trop flou vous fait perdre des points.

---

# 📦 LIVRABLES COMMUNS À TOUS LES SUJETS

Quel que soit le sujet choisi, **5 livrables techniques attendus** :

### 1. Le modèle entraîné — `model.joblib`
Exporté avec `joblib.dump(pipeline, 'model.joblib')`. Doit être rechargeable et fonctionner en `predict()`.

### 2. Les métriques — `metrics.json`
```json
{
  "model": "RandomForestRegressor",
  "metric_main": {"name": "MAE", "value": 24.5, "unit": "€"},
  "metric_secondary": {"name": "R²", "value": 0.78},
  "cv_results": {"mean": 0.78, "std": 0.03},
  "test_size": 0.2,
  "random_state": 42
}
```

### 3. Le README — `README.md`
Structure attendue :
- Description du problème métier
- Stack technique utilisée
- Comment installer (`uv venv`, `uv pip install -r requirements.txt`)
- Comment exécuter (le notebook + une cellule "démo")
- Résultats principaux (1 tableau, 1-2 graphes embarqués)
- Limites assumées + perspectives

### 4. Le notebook principal — `<NomProjet>.ipynb`
Structure attendue :
1. EDA (visualisations claires, observations écrites)
2. Pipeline (avec anti-fuite explicite)
3. Comparaison ≥3 modèles
4. Évaluation finale sur test set
5. Démo de prédiction sur un cas concret

### 5. Les slides de soutenance — `slides.pdf` (5-7 slides max)
- Slide 1 : Problème métier en 1 phrase
- Slide 2 : Données + EDA principales
- Slide 3 : Architecture (votre Pipeline)
- Slide 4 : Résultats + comparaison modèles
- Slide 5 : Démo (un exemple chiffré)
- Slide 6 : Limites + perspectives
- Slide 7 (optionnel) : Bonus

---

# 🎁 BONUS (+1 pt — capé à 20)

Choisir **UNE** des 2 pistes :

### Bonus A — Déploiement
Wrapper le modèle dans une **API FastAPI** minimaliste avec un endpoint `/predict` :
```python
@app.post("/predict")
def predict(features: PredictRequest):
    pred = model.predict(features.to_dataframe())
    return {"prediction": float(pred[0])}
```
À démontrer en soutenance (via `curl` ou Swagger).

### Bonus B — Augment LLM documenté
Utiliser un LLM (Claude / GPT) à au moins **2 étapes** de votre projet, et le **documenter** dans le README :
- Étape : *(ex. EDA — proposer des features à créer)*
- Prompt utilisé : *(intégral)*
- Output reçu : *(résumé)*
- Décision prise : *(retenu, modifié, rejeté ; pourquoi)*

⚠️ **Le LLM est un assistant, pas l'auteur.** Vous devez pouvoir reproduire à la main toute ligne de code produite par le LLM. En soutenance, vous serez interrogé sur n'importe quelle ligne.

---

# 📊 BARÈME GLOBAL /20

| Critère | Pts | Ce qu'on regarde |
|---|---|---|
| **Préparation données + EDA** | 4 | Nettoyage justifié, visualisations parlantes, intuition métier |
| **Pipeline sklearn anti-fuite** | **4** | ColumnTransformer + Pipeline ; fit sur train uniquement. **Pas de Pipeline → max 2/4** |
| **Modélisation + métrique** | 4 | ≥2 modèles comparés, métrique justifiée par le coût métier, CV reportée |
| **Présentation + vulgarisation** | 4 | Problématique métier claire, slides propres, métriques vulgarisées |
| **Démo + transparence + perspectives** | 4 | Démo qui tourne, limites assumées, pistes d'amélioration concrètes |
| **Bonus** | +1 | Déploiement OU augment LLM documenté |

**Pénalisations fortes** :
- Pas de Pipeline anti-fuite → max **2/4** sur le critère Pipeline
- 1 seul modèle testé → max **2/4** sur Modélisation
- MAE/RMSE non vulgarisés *(« j'ai un MAE de 24 »* sans expliquer ce que c'est en €)* → max **2/4** sur Présentation
- Pas de démo qui tourne → max **2/4** sur Démo

**Grille de soutenance détaillée** : distribuée jeudi matin.

---

# 📅 PLANNING SERRÉ

| Quand | Quoi |
|---|---|
| **Jeudi 10h00-10h15** | Pitch sujets libres (2 min × équipe) + validation par moi |
| **Jeudi 10h15-10h30** | Formation groupes confirmée, démarrage env uv |
| **Jeudi journée** | EDA + premier Pipeline, comparaison modèles, tuning, threshold |
| **Jeudi soir** | Démo qui tourne, slides en cours, README rédigé |
| **Vendredi matin** | Finalisation, répétition soutenance |
| **Vendredi 12h00** | **Deadline rendu Teams** *(zip ou dépôt git)* |
| **Vendredi après-midi** | **Soutenances** (30 min par équipe : 15 min démo + 15 min Q&R) |

---

# 💡 Conseils

**Conseil 1** : commencez par votre **baseline la plus simple** (LinearRegression + Pipeline minimal) **avant de complexifier**. Si votre baseline ne tourne pas, complexifier n'aidera pas.

**Conseil 2** : **VERSIONNEZ avec git** dès la première heure. Vous éviterez les nuits blanches à chercher quelle version marchait. `git init` puis `git commit -m "baseline qui tourne"` toutes les 2-3 heures.

**Conseil 3** : **gardez un fichier `decisions.md`** où vous notez vos choix techniques au fil de l'eau. *« On a choisi RandomForest plutôt que GradientBoosting car X. »* Ce sera votre matière première pour le README et la soutenance.

**Conseil 4** : **pensez à votre soutenance dès le démarrage**. Quelles sont les 3 phrases qui résument votre projet ? Si vous ne pouvez pas répondre maintenant, vous ne pourrez pas y répondre vendredi non plus.

**Conseil 5** : **dormez la nuit de jeudi à vendredi**. Le projet final récompense la clarté, pas l'épuisement.

---

# 🆘 Aide pendant le projet

- **Direct/Teams** : Me solliciter directement ou par message direct sur Teams et/ou sur un canal de groupe dans lequel vous m'aurez ajouté au prélable
- **Vous pouvez utiliser un LLM** pour vous débloquer, mais documentez-le si vous prenez le bonus B

---

> **Bonne chance — c'est votre premier vrai projet de Data Scientist. Faites-le bien.** 🚀

— Mor NDIAYE · NEEKOCODE · IPSSI M1

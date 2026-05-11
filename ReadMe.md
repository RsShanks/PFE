# 🌍 GeoPredict - Global Stock API

Bienvenue sur le repository de **GeoPredict**, une application web d'analyse boursière pilotée par l'Intelligence Artificielle. Le système croise le renseignement géopolitique mondial (via la base de données GDELT) et l'historique des marchés financiers (Yahoo Finance) pour anticiper les tendances des actifs majeurs (S&P 500, Pétrole Brut, Or).

---

## 🚀 Fonctionnalités Principales

1. **🔮 Terminal IA :** Prédiction en temps réel de la direction du marché (Hausse/Baisse) basée sur le climat géopolitique actuel.
2. **📡 Radar GDELT :** Tableau de bord répertoriant les alertes sécuritaires et diplomatiques majeures des 7 derniers jours.
3. **⚠️ Simulateur de Crise :** Outil d'analyse "What-If" permettant de modifier manuellement les tensions mondiales pour observer la réaction théorique des algorithmes.

---

## 🛠️ Installation & Configuration (Guide Windows)

### 1. Prérequis
* **Python 3.9** ou supérieur installé sur votre machine.
* **Git** pour cloner le projet.

### 2. Cloner le projet
Ouvrez votre terminal (PowerShell ou Invite de commandes) et exécutez :
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_DOSSIER>
```
### 3. Créer et activer l'environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet :

```bash
python -m venv env
.\env\Scripts\activate
```

### 4. Installer les dépendances
``` Bash
pip install -r requirements.txt
```
### 5.Initialisation du Système
Avant de lancer le serveur web, la base de données locale et les modèles de Machine Learning doivent être construits.

Nous avons mis en place un script automatisé qui va s'occuper de créer les dossiers manquants, nettoyer les données GDELT, entraîner l'IA (Random Forest / CatBoost) et générer le cache en temps réel.

Dans votre terminal (avec l'environnement virtuel activé), lancez :

```Bash
python init_project.py
Patientez pendant l'exécution (le téléchargement des données boursières et de l'historique GDELT peut prendre 1 à 2 minutes).
```
### 6.Lancer l'Application Web
Une fois l'initialisation terminée avec succès, démarrez le backend FastAPI :

```Bash
uvicorn api.main:app --reload
```
L'application est maintenant en ligne ! Ouvrez votre navigateur web :

🏠 Interface Utilisateur (Dashboard) : http://127.0.0.1:8000/

📚 Documentation API (Swagger) : http://127.0.0.1:8000/docs

### Architecture du Projet
```Plaintext
📁 GeoPredict/
│
├── 📁 api/                   # Backend FastAPI
│   ├── main.py               # Point d'entrée de l'application
│   ├── 📁 routers/           # Routes de l'API (predict, events, simulate)
│   ├── 📁 models/            # Schémas Pydantic de validation des données
│   └── 📁 data/              # Fichiers générés (Cache JSON et Modèles .pkl)
│
├── 📁 frontend/              # Interface Utilisateur (UI)
│   ├── 📁 static/            # Fichiers statiques (style.css, logo.png)
│   └── 📁 templates/         # Vues HTML propulsées par Jinja2
│
├── 📁 scripts/               # Pipeline Data Science & Machine Learning
│   ├── data_pipeline.py      # Nettoyage et fusion GDELT x Yahoo Finance
│   ├── train_model.py        # Entraînement des modèles prédictifs
│   └── update_cache.py       # Génération du cache des 3 dernières semaines
│
├── init_project.py           # Script chef d'orchestre d'installation
├── requirements.txt          # Liste des librairies Python (Pandas, FastAPI, Catboost...)
└── README.md                 # Documentation
```
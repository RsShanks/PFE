from fastapi import APIRouter, HTTPException
from api.models.schemas import MarketPredictionInput, MarketPredictionOutput, SimulationInput, SimulationOutput
import pandas as pd
import os
import joblib  
import yfinance as yf
import json
from catboost import CatBoostClassifier
from api.models.schemas import CrisisPredictionOutput
import numpy as np
router = APIRouter(
    prefix="/predict",
    tags=["Prédictions Boursières"]
)



@router.get("/{ticker}")
def predict_live_market(ticker: str):
    """
    Prédit la tendance pour le ticker demandé (ex: ^GSPC pour le S&P500)
    en combinant le cache GDELT et Yahoo Finance en direct.
    """

    if ticker ==  "^GSPC":
        model = joblib.load('api/data/models/catboost_GSPC.pkl')
    elif ticker == "CL=F":  # Pétrole Brut (WTI)
        model = joblib.load('api/data/models/random_forest_petrole.pkl')
    elif ticker == "GC=F":  # Or
        model = joblib.load('api/data/models/random_forest_or.pkl')
    elif ticker == "BTC-USD":  # Bitcoin
        model = joblib.load('api/data/models/catboost_BTC-USD.pkl')
    elif ticker == "GLD":  # Or en USD
        model = joblib.load('api/data/models/catboost_GLD.pkl')
    # elif ticker == "^GSPC":
    #     model = joblib.load('api/data/models/random_forest_sp500.pkl')
    elif ticker == "NVDA":  # NVIDIA
        model = joblib.load('api/data/models/catboost_NVDA.pkl')
    elif ticker == "^VIX":  # Volatility Index
        model = joblib.load('api/data/models/catboost_VIX.pkl')
    else:
        raise HTTPException(status_code=400, detail="Ticker non supporté")

    if model is None:
        raise HTTPException(status_code=500, detail="Modèle ML non chargé.")
    
    try:
        # 1. Lecture du cache GDELT (Les 10 premières variables)
        cache_path = "api/data/cache_features.json"
        if not os.path.exists(cache_path):
            raise HTTPException(status_code=500, detail="Cache GDELT introuvable. Lancez le script de maj.")
            
        with open(cache_path, "r") as f:
            gdelt_features = json.load(f)
            
        # 2. Récupération du momentum boursier en direct (La 11ème variable)
        # On télécharge les 10 derniers jours pour être sûr d'avoir la semaine
        market_data = yf.download(ticker, period="10d", progress=False)
        
        if 'Adj Close' in market_data.columns:
            prices = market_data['Adj Close']
        else:
            prices = market_data.iloc[:, 0]
            
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        # Calcul du rendement des 5 derniers jours ouvrés
        current_return = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        
        # 3. Assemblage des features DANS LE BON ORDRE (Ordre de ton X_train)
        feature_vector = pd.DataFrame([{
            'nb_events': gdelt_features['nb_events'],
            'avg_tone': gdelt_features['avg_tone'],
            'tension_score': gdelt_features['tension_score'],
            'mat_conf_mentions': gdelt_features['mat_conf_mentions'],
            'nb_events_lag1': gdelt_features['nb_events_lag1'],
            'avg_tone_lag1': gdelt_features['avg_tone_lag1'],
            'tension_score_lag1': gdelt_features['tension_score_lag1'],
            'nb_events_lag2': gdelt_features['nb_events_lag2'],
            'avg_tone_lag2': gdelt_features['avg_tone_lag2'],
            'tension_score_lag2': gdelt_features['tension_score_lag2'],
            'weekly_return_current': float(current_return)
        }])
        
        # 4. Prédiction
        pred = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0][pred]
        
        tendance = "HAUSSIÈRE" if pred == 1 else "BAISSIÈRE"
        
        return {
            "actif": ticker,
            "tendance": tendance,
            "confiance": round(proba * 100, 2),
            "derniere_maj_gdelt": gdelt_features["last_updated"],
            "rendement_actuel_pris_en_compte": f"{current_return * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
    

@router.post("/", response_model=MarketPredictionOutput)
def predict_market_direction(data: MarketPredictionInput):
    try:
        # 1. Convertir les données reçues en DataFrame pour scikit-learn
        input_data = pd.DataFrame([data.dict()])
        
        # 2. Faire la prédiction (Logique factice ici pour tester l'API)
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][pred]
        
        # --- SIMULATION EN ATTENDANT LE VRAI MODÈLE ---
        # pred_factice = 1 
        # proba_factice = 0.58
        # ----------------------------------------------
        
        # tendance_str = "HAUSSIÈRE" if pred_factice == 1 else "BAISSIÈRE"
        confiance = proba * 100
        tendance_str = "HAUSSIÈRE" if pred == 1 else "BAISSIÈRE"
        # 3. Renvoyer la réponse formatée
        return MarketPredictionOutput(
            prediction=pred,
            tendance=tendance_str,
            confiance=confiance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
    
@router.post("/simulate", response_model=SimulationOutput)
def simulate_market_reaction(data: SimulationInput):
    try:
        # 1. Création d'une "Baseline" (Situation normale/neutre pour les variables cachées)
        # On remplit les lags avec des valeurs historiques moyennes pour que le modèle fonctionne
        baseline_features = {
            'nb_events': 500, 
            'avg_tone': data.simulated_tone, # La valeur du slider utilisateur !
            'tension_score': data.simulated_tension, # La valeur du slider utilisateur !
            'mat_conf_mentions': data.simulated_material_conflicts, # La valeur du slider utilisateur !
            
            # Variables de "mémoire" neutres
            'nb_events_lag1': 500,
            'avg_tone_lag1': 0.0, 
            'tension_score_lag1': 100.0,
            'nb_events_lag2': 500,
            'avg_tone_lag2': 0.0,
            'tension_score_lag2': 100.0,
            
            # Momentum neutre (le marché a fait 0% la semaine dernière)
            'weekly_return_current': 0.0 
        }
        
        # 2. Préparation pour Scikit-Learn
        df_simul = pd.DataFrame([baseline_features])
        
        # 3. Choix du modèle et prédiction (SIMULATION ICI)
        # En production : 
        # if data.actif == "Pétrole Brut (WTI)":
        #     pred = model_oil.predict(df_simul)[0]
        #     proba = model_oil.predict_proba(df_simul)[0][pred]
        # else: ...

        if data.actif ==  "^GSPC":
            model = joblib.load('api/data/models/catboost_GSPC.pkl')
        elif data.actif == "CL=F":  # Pétrole Brut (WTI)
            model = joblib.load('api/data/models/random_forest_petrole.pkl')
        elif data.actif == "GC=F":  # Or
            model = joblib.load('api/data/models/random_forest_or.pkl')
        elif data.actif == "BTC-USD":  # Bitcoin
            model = joblib.load('api/data/models/catboost_BTC-USD.pkl')
        elif data.actif == "GLD":  # Or en USD
            model = joblib.load('api/data/models/catboost_GLD.pkl')
        # elif data.actif == "^GSPC":
        #     model = joblib.load('api/data/models/random_forest_sp500.pkl')
        elif data.actif == "NVDA":  # NVIDIA
            model = joblib.load('api/data/models/catboost_NVDA.pkl')
        elif data.actif == "^VIX":  # Volatility Index
            model = joblib.load('api/data/models/catboost_VIX.pkl')
        else:
            raise HTTPException(status_code=400, detail="Ticker non supporté")
        
        pred = model.predict(df_simul)[0]
        proba = model.predict_proba(df_simul)[0][pred]

        
        tendance_str = "HAUSSIÈRE" if pred == 1 else "BAISSIÈRE"
        
        return SimulationOutput(
            actif=data.actif,
            prediction=pred,
            tendance=tendance_str,
            confiance=round(proba * 100, 2),
            message_analyse=tendance_str + " anticipée pour " + data.actif + " avec une confiance de " + str(round(proba * 100, 2)) + "%."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de simulation : {str(e)}")

@router.get("/crisis/{ticker}", response_model=CrisisPredictionOutput)
def predict_crisis(ticker: str):
    model_paths = {
        "^GSPC": "api/data/models/stress_crisis_model.pkl",
        "CL=F": "api/data/models/stress_crisis_model_petrole.pkl",
        "BTC-USD": "api/data/models/stress_crisis_model_bitcoin.pkl",
    }

    if ticker not in model_paths:
        raise HTTPException(
            status_code=400,
            detail="Ticker cas crise non supporté. Tickers disponibles : ^GSPC, CL=F, BTC-USD"
        )

    try:
        bundle = joblib.load(model_paths[ticker])

        if isinstance(bundle, dict):
            model = bundle["model"]
            expected_features = bundle["features"]
        else:
            model = bundle
            expected_features = None

        cache_paths = {
            "^GSPC": "api/data/cache_features_crise_gspc.json",
            "CL=F": "api/data/cache_features_crise_petrole.json",
            "BTC-USD": "api/data/cache_features_crise_bitcoin.json",
        }

        cache_path = cache_paths.get(ticker)
        if cache_path is None:
            raise HTTPException(
                status_code=400,
                detail="Cache de crise introuvable pour ce ticker."
            )

        if not os.path.exists(cache_path):
            raise HTTPException(
                status_code=500,
                detail="Cache GDELT pour crise introuvable. Lancez le script de maj."
            )

        with open(cache_path, "r", encoding="utf-8") as f:
            gdelt_cache = json.load(f)

        features_live = gdelt_cache["features"]

        feature_vector = pd.DataFrame([features_live])

        if expected_features is not None:
            feature_vector = feature_vector.reindex(
                columns=expected_features,
                fill_value=0
            )

        pred = int(np.ravel(model.predict(feature_vector))[0])
        probas = model.predict_proba(feature_vector)[0]
        pred_proba = float(probas[pred])

        crisis_mapping = {
            0: "NORMAL",
            1: "STRESS",
        }

        probabilites = {
            crisis_mapping.get(i, f"CLASSE_{i}").lower(): round(float(p) * 100, 2)
            for i, p in enumerate(probas)
        }

        return {
            "actif": ticker,
            "niveau_crise": crisis_mapping.get(pred, "INCONNU"),
            "classe_predite": pred,
            "probabilites": probabilites,
            "confiance_prediction": round(pred_proba * 100, 2),
            "derniere_maj_gdelt": gdelt_cache["last_updated"],
            "semaine_gdelt": gdelt_cache["week"],
            "variables_importantes": {
                "tension_score": features_live.get("tension_score"),
                "conflict_ratio": features_live.get("conflict_ratio"),
                "material_conflict_ratio": features_live.get("material_conflict_ratio"),
                "usa_china_tension": features_live.get("usa_china_tension"),
                "usa_russia_tension": features_live.get("usa_russia_tension"),
                "avg_tone": features_live.get("avg_tone"),

                "weekly_return_current": features_live.get("weekly_return_current"),
                "vol_4w": features_live.get("vol_4w"),
                "vol_8w": features_live.get("vol_8w"),
                "momentum_4w": features_live.get("momentum_4w"),
                "momentum_12w": features_live.get("momentum_12w"),
                "drawdown_26w": features_live.get("drawdown_26w"),

                "vix": features_live.get("vix"),
                "vix_delta_1w": features_live.get("vix_delta_1w"),
                "vix_zscore_52w": features_live.get("vix_zscore_52w"),

                "tension_x_vol_4w": features_live.get("tension_x_vol_4w"),
                "conflict_x_drawdown": features_live.get("conflict_x_drawdown"),
                "vix_x_tension": features_live.get("vix_x_tension")
            },
            "top_events": gdelt_cache.get("top_events", {}),
            "top_conflict_countries": gdelt_cache.get("top_conflict_countries", {}),
            "evolution": gdelt_cache.get("evolution", {})
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction crise : {str(e)}"
        )
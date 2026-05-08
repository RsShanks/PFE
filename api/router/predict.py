from fastapi import APIRouter, HTTPException
from api.models.schemas import MarketPredictionInput, MarketPredictionOutput, SimulationInput, SimulationOutput
import pandas as pd
import joblib  

router = APIRouter(
    prefix="/predict",
    tags=["Prédictions Boursières"]
)

# Simulation du chargement du modèle (A déplacer dans ml_service.py plus tard)
model = joblib.load('api/data/models/random_forest_model.pkl')

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
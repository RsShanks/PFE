from pydantic import BaseModel, Field

# Les données d'entrée attendues par le modèle (X)
class MarketPredictionInput(BaseModel):
    nb_events: int = Field(..., description="Nombre total d'événements cette semaine")
    avg_tone: float = Field(..., description="Sentiment moyen de la semaine")
    tension_score: float = Field(..., description="Score de tension GDELT")
    mat_conf_mentions: int = Field(..., description="Mentions de conflits matériels")
    
    # Lags semaine - 1
    nb_events_lag1: int
    avg_tone_lag1: float
    tension_score_lag1: float
    
    # Lags semaine - 2
    nb_events_lag2: int
    avg_tone_lag2: float
    tension_score_lag2: float
    
    # Marché
    weekly_return_current: float = Field(..., description="Rendement boursier de la semaine en cours")

# La réponse renvoyée par l'API (y)
class MarketPredictionOutput(BaseModel):
    prediction: int = Field(..., description="1 pour Hausse, 0 pour Baisse")
    tendance: str = Field(..., description="'HAUSSIÈRE' ou 'BAISSIÈRE'")
    confiance: float = Field(..., description="Pourcentage de confiance du modèle")



class SimulationInput(BaseModel):
    actif: str = Field(..., description="L'actif ciblé (ex: S&P 500, Pétrole)")
    simulated_tone: float = Field(..., description="Sentiment moyen (ex: de -10 à +10)")
    simulated_tension: float = Field(..., description="Score de tension (ex: de 0 à 2000)")
    simulated_material_conflicts: int = Field(..., description="Mentions de conflits matériels")

class SimulationOutput(BaseModel):
    actif: str
    prediction: int
    tendance: str
    confiance: float
    message_analyse: str
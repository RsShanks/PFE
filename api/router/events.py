from fastapi import APIRouter, HTTPException
import pandas as pd
import gdelt
import numpy as np
import os
import json

router = APIRouter(
    prefix="/events",
    tags=["Événements GDELT Temps Réel"]
)

# On initialise le client GDELT en dehors de la route pour ne le faire qu'une fois
gd2 = gdelt.gdelt(version=2)

@router.get("/daily-focus")
def get_daily_events():
    """
    Récupère les événements GDELT du jour en cours.
    """
    try:
        # 1. Formatage de la date du jour (Format GDELT : YYYYMMDD)
        today = pd.Timestamp.today().strftime('%Y%m%d')
        # Get yesterday's date
        yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y%m%d')

        # 2. Requête GDELT (Laissé par défaut en DataFrame Pandas pour faciliter le tri)
        results_df = gd2.Search([yesterday], table='events')
        
        # Sécurité : Si GDELT ne renvoie rien
        if results_df is None or results_df.empty:
            return {"date": today, "message": "Aucun événement pour le moment.", "events": []}
            
        # 3. Nettoyage et Filtrage pour le Front-end
        # On ne garde que l'essentiel pour ne pas surcharger l'API
        colonnes_utiles = [
            'GlobalEventID', 'Actor1Name', 'Actor1CountryCode', 
            'EventRootCode', 'GoldsteinScale', 'NumMentions', 'SOURCEURL'
        ]
        
        # On s'assure que les colonnes existent
        df_clean = results_df[[col for col in colonnes_utiles if col in results_df.columns]].copy()
        
        # On enlève les lignes sans URL source (inutile pour lire l'article)
        if 'SOURCEURL' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['SOURCEURL'])
            
        # On remplace les NaN restants par des None (compatibles avec le format JSON)
        df_clean = df_clean.replace({np.nan: None})
        
        # Optionnel : On trie par nombre de mentions pour avoir les plus importants d'abord
        if 'NumMentions' in df_clean.columns:
            df_clean = df_clean.sort_values(by='NumMentions', ascending=False)
            
        # On limite aux 50 événements les plus importants pour la rapidité d'affichage
        df_final = df_clean.head(50)
        
        # 4. Conversion en dictionnaire pour que FastAPI génère le JSON
        events_list = df_final.to_dict(orient="records")
        
        return {
            "date": today,
            "total_events_found": len(results_df),
            "events_returned": len(events_list),
            "events": events_list
        }

    except Exception as e:
        # Si le site GDELT est down ou qu'il y a une erreur
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération GDELT : {str(e)}")
    
@router.get("/weekly-focus")
def get_weekly_focus_events():
    """
    Récupère instantanément les événements géopolitiques majeurs 
    pré-calculés de la semaine passée (Zéro surcharge RAM).
    """
    try:
        cache_path = "api/data/weekly_radar.json"
        
        # Sécurité : si le fichier n'existe pas encore
        if not os.path.exists(cache_path):
            raise HTTPException(
                status_code=503, 
                detail="Le radar hebdomadaire est en cours d'initialisation. Veuillez réessayer dans quelques instants."
            )
            
        # Lecture ultra-rapide du fichier JSON
        with open(cache_path, "r", encoding="utf-8") as f:
            radar_data = json.load(f)
            
        return radar_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du radar : {str(e)}")
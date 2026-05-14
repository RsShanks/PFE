from fastapi import APIRouter, HTTPException
import pandas as pd
import gdelt
import numpy as np
import gc

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
    Récupère les événements géopolitiques majeurs des 7 derniers jours.
    Version optimisée pour basse mémoire (RAM < 512MB).
    """
    try:
        last_7_days = pd.date_range(end=pd.Timestamp.today(), periods=7).strftime('%Y%m%d').tolist()
        
        cibles_geo = ["USA", "CHN", "EUR", "FRA", "DEU", "GBR", "ITA", "ESP"]
        colonnes_utiles = [
            'GlobalEventID', 'SQLDATE', 'Actor1Name', 'Actor1CountryCode', 
            'ActionGeo_CountryCode', 'EventRootCode', 'GoldsteinScale', 
            'NumMentions', 'SOURCEURL'
        ]
        
        all_filtered_events = []
        total_raw_events_count = 0
        
        # 1. Traitement itératif : Un jour à la fois pour économiser la RAM
        for day in last_7_days:
            try:
                # On télécharge un seul jour
                df_day = gd2.Search([day], table='events')
                
                if df_day is None or df_day.empty:
                    continue
                    
                total_raw_events_count += len(df_day)
                
                # On filtre tout de suite pour réduire la taille par 100
                mask = (
                    df_day['Actor1CountryCode'].isin(cibles_geo) |
                    df_day['Actor2CountryCode'].isin(cibles_geo)
                )
                df_filtered_day = df_day[mask]
                
                # On ne garde que les colonnes utiles
                cols_to_keep = [col for col in colonnes_utiles if col in df_filtered_day.columns]
                df_filtered_day = df_filtered_day[cols_to_keep]
                
                # On sauvegarde ce petit bout
                all_filtered_events.append(df_filtered_day)
                
                # NETTOYAGE MANUEL DE LA RAM (Crucial pour Render)
                del df_day
                gc.collect() 
                
            except Exception as e:
                print(f"Erreur lors de la récupération du jour {day} : {e}")
                continue
                
        # Si on n'a rien trouvé du tout sur la semaine
        if not all_filtered_events:
            return {"message": "Aucun événement majeur trouvé.", "events": []}
            
        # 2. On assemble nos petits bouts pré-filtrés
        df_clean = pd.concat(all_filtered_events, ignore_index=True)
        
        # 3. Calculs et Tris
        df_clean['GoldsteinScale'] = pd.to_numeric(df_clean['GoldsteinScale'], errors='coerce').fillna(0)
        df_clean['NumMentions'] = pd.to_numeric(df_clean['NumMentions'], errors='coerce').fillna(0)

        df_clean['RiskScore'] = np.where(
            df_clean['GoldsteinScale'] < 0,
            abs(df_clean['GoldsteinScale']) * df_clean['NumMentions'],
            0
        )

        if 'SOURCEURL' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['SOURCEURL'])
            
        # On trie par RiskScore
        df_clean = df_clean.sort_values(by='RiskScore', ascending=False)
        
        # LIMITATION STRICTE POUR JSON (Remis à 50 ou 100 max)
        df_final = df_clean.head(50)
        
        # 4. Conversion propre
        df_final = df_final.fillna("")
        df_string = df_final.astype(str)
        events_list = df_string.to_dict(orient="records")
        
        return {
            "period": f"{last_7_days[0]} to {last_7_days[-1]}",
            "regions_tracked": cibles_geo,
            "total_events_filtered": str(len(df_clean)),
            "events_returned": str(len(events_list)),
            "events": events_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur GDELT Hebdo : {str(e)}")
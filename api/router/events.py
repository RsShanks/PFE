from fastapi import APIRouter, HTTPException
import pandas as pd
import gdelt
import numpy as np

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
    Récupère les événements géopolitiques majeurs des 7 derniers jours 
    impliquant les USA, la Chine ou l'Europe.
    """
    try:
        # 1. Générer les dates des 7 derniers jours
        # On crée une liste de chaînes 'YYYYMMDD' pour les 7 jours passés
        last_7_days = pd.date_range(end=pd.Timestamp.today(), periods=7).strftime('%Y%m%d').tolist()
        
        # 2. Requête GDELT (Attention, ça peut prendre 10-20 secondes pour 7 jours !)
        results_df = gd2.Search(last_7_days, table='events')
        
        if results_df is None or len(results_df) == 0:
            return {"message": "Aucun événement trouvé.", "events": []}
            
        # 3. Filtrage Géopolitique
        # Liste des codes CAMEO cibles
        cibles_geo = [
            "USA", # États-Unis
            "CHN", # Chine
            "EUR", # Union Européenne (en tant qu'organisation)
            "FRA", "DEU", "GBR", "ITA", "ESP" # Pays européens majeurs
        ]
        
        # On filtre : On veut que l'acteur 1, l'acteur 2 OU le lieu de l'action soit dans notre liste
        mask = (
            results_df['Actor1CountryCode'].isin(cibles_geo) |
            results_df['Actor2CountryCode'].isin(cibles_geo) |
            results_df['ActionGeo_CountryCode'].isin(cibles_geo)
        )
        
        df_filtered = results_df[mask].copy()
        
        if df_filtered.empty:
            return {"message": "Aucun événement majeur pour ces régions.", "events": []}

        # 4. Nettoyage et préparation pour le front
        colonnes_utiles = [
            'GlobalEventID', 'SQLDATE', 'Actor1Name', 'Actor1CountryCode', 
            'ActionGeo_CountryCode', 'EventRootCode', 'GoldsteinScale', 
            'NumMentions', 'SOURCEURL'
        ]
        
        df_clean = df_filtered[[col for col in colonnes_utiles if col in df_filtered.columns]].copy()

        # On force les types pour le calcul
        df_clean['GoldsteinScale'] = pd.to_numeric(df_clean['GoldsteinScale'], errors='coerce').fillna(0)
        df_clean['NumMentions'] = pd.to_numeric(df_clean['NumMentions'], errors='coerce').fillna(0)

        # Calcul du "Risk Score" : on pénalise les événements négatifs
        # On prend la valeur absolue de Goldstein (uniquement s'il est négatif) multipliée par les mentions
        df_clean['RiskScore'] = np.where(
            df_clean['GoldsteinScale'] < 0,
            abs(df_clean['GoldsteinScale']) * df_clean['NumMentions'],
            0
        )

        # On trie par ce nouveau scored
        df_clean = df_clean.sort_values(by='RiskScore', ascending=False)

        
        if 'SOURCEURL' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['SOURCEURL'])
            
        # On trie par impact médiatique (Mentions)
        if 'NumMentions' in df_clean.columns:
            df_clean = df_clean.sort_values(by='RiskScore', ascending=False)            
        # On prend le Top 50 de la semaine pour ne pas saturer l'interface
        df_final = df_clean.head(50)
        
        # 5. L'Option Anti-NumPy
        df_final = df_final.fillna("")
        df_string = df_final.astype(str)
        events_list = df_string.to_dict(orient="records")
        
        return {
            "period": f"{last_7_days[0]} to {last_7_days[-1]}",
            "regions_tracked": cibles_geo,
            "total_events_filtered": str(len(df_filtered)),
            "events_returned": str(len(events_list)),
            "events": events_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur GDELT Hebdo : {str(e)}")
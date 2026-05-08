# scripts/update_cache.py
import pandas as pd
import json
from datetime import datetime, timedelta
import gdelt # Assure-toi que la librairie est installée
import os

def update_live_cache():
    print("Mise à jour du cache GDELT (3 dernières semaines)...")
    gd2 = gdelt.gdelt(version=2)
    
    # 1. Générer les dates des 21 derniers jours
    dates = pd.date_range(end=datetime.today(), periods=21).strftime('%Y%m%d').tolist()
    
    # 2. Récupérer les données brutes
    # Attention: Sur 21 jours, GDELT peut renvoyer beaucoup de données, on filtre direct
    results = []
    for d in dates:
        try:
            df_day = gd2.Search([d], table='events')
            if df_day is not None and not df_day.empty:
                results.append(df_day)
        except Exception as e:
            print(f"Erreur le {d}: {e}")
            
    df_raw = pd.concat(results, ignore_index=True)
    
    # 3. Application de TES filtres (adaptation de ta fonction)
    df_raw["SQLDATE"] = pd.to_datetime(df_raw["SQLDATE"].astype(str), format="%Y%m%d")
    df_raw["week"] = df_raw["SQLDATE"].dt.to_period("W").dt.to_timestamp()
    
    # Nettoyage
    df_raw["EventDescriptionFinal"] = df_raw.get("EventDescriptionFinal", df_raw.get("EventRootCode", ""))
    df_raw = df_raw.dropna(subset=["GoldsteinScale", "NumMentions", "AvgTone"])
    df_raw["GoldsteinScale"] = pd.to_numeric(df_raw["GoldsteinScale"], errors='coerce').fillna(0)
    df_raw["NumMentions"] = pd.to_numeric(df_raw["NumMentions"], errors='coerce').fillna(0)
    df_raw["AvgTone"] = pd.to_numeric(df_raw["AvgTone"], errors='coerce').fillna(0)
    
    df_raw["negative_intensity"] = df_raw["GoldsteinScale"].clip(upper=0).abs() * df_raw["NumMentions"]
    df_raw["IsMaterialConflict"] = df_raw["QuadClass"].astype(str) == "3" # 3 = Material Conflict dans CAMEO
    
    # 4. Agrégation Hebdomadaire
    features = df_raw.groupby("week").agg(
        nb_events=("EventCode", "count"),
        avg_tone=("AvgTone", "mean"),
        tension_score=("negative_intensity", "sum"),
        mat_conf_mentions=("NumMentions", lambda x: x[df_raw.loc[x.index, "IsMaterialConflict"]].sum())
    ).reset_index()
    
    features = features.sort_values("week", ascending=False).head(3) # On garde les 3 dernières semaines
    
    if len(features) < 3:
        print("Attention : Pas assez de données pour générer les lags.")
        return
        
    # 5. Création du vecteur final (Semaine 0 = current, Semaine 1 = lag1, Semaine 2 = lag2)
    # L'index 0 est la semaine en cours, index 1 est la semaine dernière, etc.
    final_cache = {
        "last_updated": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "nb_events": int(features.iloc[0]['nb_events']),
        "avg_tone": float(features.iloc[0]['avg_tone']),
        "tension_score": float(features.iloc[0]['tension_score']),
        "mat_conf_mentions": int(features.iloc[0]['mat_conf_mentions']),
        
        "nb_events_lag1": int(features.iloc[1]['nb_events']),
        "avg_tone_lag1": float(features.iloc[1]['avg_tone']),
        "tension_score_lag1": float(features.iloc[1]['tension_score']),
        
        "nb_events_lag2": int(features.iloc[2]['nb_events']),
        "avg_tone_lag2": float(features.iloc[2]['avg_tone']),
        "tension_score_lag2": float(features.iloc[2]['tension_score'])
    }
    
    # 6. Sauvegarde
    os.makedirs("api/data", exist_ok=True)
    with open("api/data/cache_features.json", "w") as f:
        json.dump(final_cache, f, indent=4)
        
    print("Cache généré avec succès dans api/data/cache_features.json")

if __name__ == "__main__":
    update_live_cache()
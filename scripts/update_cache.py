import pandas as pd
import numpy as np
import json
from datetime import datetime
import gdelt
import os
import gc

def update_live_cache():
    print("=== 🛠️ DÉBUT DE LA MISE À JOUR DU CACHE GEOPREDICT ===")
    gd2 = gdelt.gdelt(version=2)
    os.makedirs("api/data", exist_ok=True)
    
    # 1. Configuration des Filtres
    cibles_geo = ["USA", "CHN", "EUR", "FRA", "DEU", "GBR", "ITA", "ESP"]
    
    dates = pd.date_range(end=datetime.today(), periods=21).strftime('%Y%m%d').tolist()
    last_7_days = dates[-7:]
    
    all_raw_days = []
    all_radar_events = []
    
    for day in dates:
        try:
            print(f"📥 Téléchargement du jour GDELT : {day}...")
            # Le flux en direct renvoie 58 colonnes
            df_day = gd2.Search([day], table='events')
            
            if df_day is None or df_day.empty:
                continue
                
            # --- PARTIE A : ML (Utilisation stricte de tes colonnes) ---
            df_day["SQLDATE"] = pd.to_datetime(df_day["SQLDATE"].astype(str), format="%Y%m%d")
            df_day["week"] = df_day["SQLDATE"].dt.to_period("W").dt.to_timestamp()
            
            # Forcer le format numérique sur tes colonnes cibles
            df_day["GoldsteinScale"] = pd.to_numeric(df_day["GoldsteinScale"], errors='coerce').fillna(0)
            df_day["NumMentions"] = pd.to_numeric(df_day["NumMentions"], errors='coerce').fillna(0)
            df_day["AvgTone"] = pd.to_numeric(df_day["AvgTone"], errors='coerce').fillna(0)
            
            # Calculs exacts comme dans ton notebook
            df_day["negative_intensity"] = df_day["GoldsteinScale"].clip(upper=0).abs() * df_day["NumMentions"]
            # Dans GDELT, QuadClass = 3 correspond aux conflits matériels
            df_day["IsMaterialConflict"] = df_day["QuadClass"].astype(str).str.strip() == "3"
            
            # Agrégation ML : on compte les lignes (EventCode) comme dans ton script d'entraînement
            df_day_agg = df_day.groupby("week").agg(
                nb_events=("EventCode", "count"), 
                avg_tone=("AvgTone", "mean"),
                tension_score=("negative_intensity", "sum"),
                mat_conf_mentions=("NumMentions", lambda x: x[df_day.loc[x.index, "IsMaterialConflict"]].sum())
            ).reset_index()
            all_raw_days.append(df_day_agg)
            
            # --- PARTIE B : RADAR FRONT-END (On garde les URL et Noms) ---
            if day in last_7_days:
                mask = (df_day['Actor1CountryCode'].isin(cibles_geo) | df_day['Actor2CountryCode'].isin(cibles_geo))
                df_filtered_radar = df_day[mask].copy()
                all_radar_events.append(df_filtered_radar)
            
            del df_day
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ Erreur ignorée pour le jour {day} : {e}")
            continue

    # =========================================================
    # ÉCRITURE CACHE 1 : ML (cache_features.json)
    # =========================================================
    print("\n💾 Structuration des variables de prédiction...")
    if all_raw_days:
        df_features_raw = pd.concat(all_raw_days, ignore_index=True)
        features_agg = df_features_raw.groupby("week").agg(
            nb_events=("nb_events", "sum"),
            avg_tone=("avg_tone", "mean"),
            tension_score=("tension_score", "sum"),
            mat_conf_mentions=("mat_conf_mentions", "sum")
        ).reset_index().sort_values("week", ascending=False).head(3)
        
        if len(features_agg) >= 3:
            cache_ml = {
                "last_updated": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "nb_events": int(features_agg.iloc[0]['nb_events']),
                "avg_tone": float(features_agg.iloc[0]['avg_tone']),
                "tension_score": float(features_agg.iloc[0]['tension_score']),
                "mat_conf_mentions": int(features_agg.iloc[0]['mat_conf_mentions']),
                "nb_events_lag1": int(features_agg.iloc[1]['nb_events']),
                "avg_tone_lag1": float(features_agg.iloc[1]['avg_tone']),
                "tension_score_lag1": float(features_agg.iloc[1]['tension_score']),
                "nb_events_lag2": int(features_agg.iloc[2]['nb_events']),
                "avg_tone_lag2": float(features_agg.iloc[2]['avg_tone']),
                "tension_score_lag2": float(features_agg.iloc[2]['tension_score'])
            }
            with open("api/data/cache_features.json", "w") as f:
                json.dump(cache_ml, f, indent=4)
            print("✅ cache_features.json généré.")

    # =========================================================
    # ÉCRITURE CACHE 2 : RADAR (weekly_radar.json)
    # =========================================================
    if all_radar_events:
        print("💾 Structuration du Radar Géopolitique...")
        df_radar = pd.concat(all_radar_events, ignore_index=True)
        
        # RiskScore pour trier les pires événements de la semaine
        df_radar['RiskScore'] = np.where(df_radar['GoldsteinScale'] < 0, abs(df_radar['GoldsteinScale']) * df_radar['NumMentions'], 0)
        
        # On ne garde que ceux qui ont une URL (pour pouvoir cliquer dessus sur le site)
        if 'SOURCEURL' in df_radar.columns:
            df_radar = df_radar.dropna(subset=['SOURCEURL'])
            
        df_radar = df_radar.sort_values(by='RiskScore', ascending=False)
        
        # Colonnes finales envoyées au Front-End
        colonnes_radar_front = ['SQLDATE', 'Actor1Name', 'ActionGeo_CountryCode', 'GoldsteinScale', 'NumMentions', 'SOURCEURL', 'RiskScore']
        cols_existantes = [col for col in colonnes_radar_front if col in df_radar.columns]
        
        df_top_radar = df_radar[cols_existantes].head(50).fillna("")
        events_list = df_top_radar.astype(str).to_dict(orient="records")
        
        cache_radar = {
            "period": f"{last_7_days[0]} to {last_7_days[-1]}",
            "regions_tracked": cibles_geo,
            "total_events_filtered": str(len(df_radar)),
            "events_returned": str(len(events_list)),
            "events": events_list
        }
        with open("api/data/weekly_radar.json", "w") as f:
            json.dump(cache_radar, f, indent=4)
        print("✅ weekly_radar.json généré.")
        
    print("\n=== ✨ FIN DE LA MISE À JOUR DU CACHE ===")

if __name__ == "__main__":
    update_live_cache()
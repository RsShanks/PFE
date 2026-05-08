import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# 1. Préparation des données GDELT
# ==========================================
def load_and_preprocess_gdelt(filepath):
    print(f"Chargement de {filepath}...")
    df = pd.read_csv(filepath)
    
    # Dates
    df["SQLDATE"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d")
    df["year"] = df["SQLDATE"].dt.year
    df["week"] = df["SQLDATE"].dt.to_period("W").dt.to_timestamp()

    # Nettoyage et sécurité
    df["EventDescriptionFinal"] = df["EventDescriptionFinal"].fillna(df["EventRootDescription"])
    df = df.dropna(subset=["EventDescriptionFinal", "GoldsteinScale", "NumMentions", "AvgTone"])
    
    # Calcul de l'intensité négative (Score de tension)
    df["negative_intensity"] = df["GoldsteinScale"].clip(upper=0).abs() * df["NumMentions"]
    
    return df

# ==========================================
# 2. Agrégation hebdomadaire (Features)
# ==========================================
def generate_finance_features(df_gdelt):
    print("Génération des indicateurs hebdomadaires...")
    
    # Agrégation par semaine
    features = df_gdelt.groupby("week").agg(
        nb_events=("EventCode", "count"),
        total_mentions=("NumMentions", "sum"),
        avg_goldstein=("GoldsteinScale", "mean"),
        avg_tone=("AvgTone", "mean"),
        tension_score=("negative_intensity", "sum"),
        # Somme des mentions pour les conflits matériels
        mat_conf_mentions=("NumMentions", lambda x: x[df_gdelt.loc[x.index, "QuadClassLabel"] == "Material conflict"].sum()),
        # Somme des mentions pour les conflits verbaux
        verb_conf_mentions=("NumMentions", lambda x: x[df_gdelt.loc[x.index, "QuadClassLabel"] == "Verbal conflict"].sum())
    ).reset_index()
    
    return features

# ==========================================
# 3. Ajout des données boursières (Target)
# ==========================================
def add_market_data(df_features, ticker="^GSPC", start="2006-01-01", end="2022-12-31"):
    print(f"Téléchargement des données Yahoo Finance pour {ticker}...")
    
    sp500 = yf.download(ticker, start=start, end=end, auto_adjust=False)
    
    # Gestion robuste des colonnes Yahoo Finance (MultiIndex ou non)
    if 'Adj Close' in sp500.columns:
        close_prices = sp500['Adj Close']
    else:
        close_prices = sp500.iloc[:, 0]
        
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # Calcul du rendement hebdo et décalage pour la prédiction (Target)
    data_weekly = close_prices.resample('W-MON').last().pct_change().to_frame(name='weekly_return')
    data_weekly['target_return'] = data_weekly['weekly_return'].shift(-1)
    data_weekly = data_weekly.reset_index()
    
    # Fusion avec les features GDELT
    data_weekly['Date'] = pd.to_datetime(data_weekly['Date'])
    df_features['week'] = pd.to_datetime(df_features['week'])

    final_df = df_features.merge(
        data_weekly[['Date', 'target_return']], 
        left_on="week", 
        right_on="Date", 
        how="left"
    ).drop(columns=["Date"])
    
    return final_df.dropna(subset=['target_return'])

# ==========================================
# EXÉCUTION DU PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Charger
    df_raw = load_and_preprocess_gdelt("scripts/events/gdelt_clean_mapped.csv")
    
    # 2. Transformer en Hebdo
    df_weekly = generate_finance_features(df_raw)
    # df_weekly = pd.read_csv("scripts/events/gdelt_weekly_finance_features.csv")  # Chargement direct pour accélérer les tests
    # 3. Fusionner avec la bourse (tu peux changer le ticker ici !)
    # Exemple : "^FCHI" pour le CAC40, "BTC-USD" pour le Bitcoin
    df_final = add_market_data(df_weekly, ticker="^GSPC")
    
    # 4. Sauvegarder
    df_final.to_csv("scripts/events/gdelt_weekly_finance_features_with_returns.csv", index=False)
    print("\nPipeline terminé ! Fichier prêt.")
    print(df_final.head())
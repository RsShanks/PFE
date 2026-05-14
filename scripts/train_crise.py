import os
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "scripts", "events", "gdelt_clean_mapped.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "api", "data", "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TICKER = "BTC-USD"
START_DATE = "2006-01-01"
END_DATE = "2022-12-31"

SPLIT_DATE = "2020-08-01"

MODEL_PATH = os.path.join(MODEL_DIR, "stress_crisis_model_bitcoin.pkl")


# ============================================================
# 1. CHARGEMENT ET NETTOYAGE GDELT
# ============================================================

def load_gdelt(filepath: str) -> pd.DataFrame:
    print(f"Chargement du fichier GDELT : {filepath}")

    df = pd.read_csv(filepath)

    required_cols = [
        "SQLDATE",
        "Actor1CountryCode",
        "Actor2CountryCode",
        "ActionGeo_CountryCode",
        "EventCode",
        "EventRootCode",
        "GoldsteinScale",
        "QuadClass",
        "NumMentions",
        "NumArticles",
        "AvgTone",
        "is_major_country",
        "QuadClassLabel",
        "EventRootDescription",
        "EventDescriptionFinal"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {missing}")

    df["SQLDATE"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["SQLDATE"])

    df["week"] = df["SQLDATE"].dt.to_period("W").dt.to_timestamp()

    numeric_cols = [
        "GoldsteinScale",
        "QuadClass",
        "NumMentions",
        "NumArticles",
        "AvgTone",
        "is_major_country"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["GoldsteinScale", "QuadClass", "NumMentions", "AvgTone"])

    df["Actor1CountryCode"] = df["Actor1CountryCode"].fillna("UNK").astype(str)
    df["Actor2CountryCode"] = df["Actor2CountryCode"].fillna("UNK").astype(str)
    df["ActionGeo_CountryCode"] = df["ActionGeo_CountryCode"].fillna("UNK").astype(str)

    df["QuadClassLabel"] = df["QuadClassLabel"].fillna("Unknown")
    df["EventRootDescription"] = df["EventRootDescription"].fillna("Unknown")
    df["EventDescriptionFinal"] = df["EventDescriptionFinal"].fillna(df["EventRootDescription"])
#     df["future_min_return_4w"] = (
#     df["target_return"]
#     .rolling(window=4, min_periods=1)
#     .min()
#     .shift(-3)
# )
    return df


# ============================================================
# 2. FEATURE ENGINEERING GÉOPOLITIQUE
# ============================================================

def create_gdelt_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Création des features hebdomadaires GDELT...")

    df["negative_intensity"] = df["GoldsteinScale"].clip(upper=0).abs() * df["NumMentions"]
    df["positive_intensity"] = df["GoldsteinScale"].clip(lower=0) * df["NumMentions"]

    df["is_verbal_coop"] = df["QuadClass"] == 1
    df["is_material_coop"] = df["QuadClass"] == 2
    df["is_verbal_conflict"] = df["QuadClass"] == 3
    df["is_material_conflict"] = df["QuadClass"] == 4

    df["is_major_country"] = df["is_major_country"].fillna(0).astype(int)

    major_codes = ["USA", "CHN", "RUS", "GBR", "FRA", "DEU", "JPN", "IND", "IRN", "ISR", "UKR"]

    df["major_actor_event"] = (
        df["Actor1CountryCode"].isin(major_codes)
        | df["Actor2CountryCode"].isin(major_codes)
        | df["ActionGeo_CountryCode"].isin(major_codes)
        | (df["is_major_country"] == 1)
    )

    df["usa_event"] = (
        (df["Actor1CountryCode"] == "USA")
        | (df["Actor2CountryCode"] == "USA")
        | (df["ActionGeo_CountryCode"] == "US")
    )

    df["china_event"] = (
        (df["Actor1CountryCode"] == "CHN")
        | (df["Actor2CountryCode"] == "CHN")
        | (df["ActionGeo_CountryCode"] == "CH")
    )

    df["russia_event"] = (
        (df["Actor1CountryCode"] == "RUS")
        | (df["Actor2CountryCode"] == "RUS")
        | (df["ActionGeo_CountryCode"] == "RS")
    )

    df["middle_east_event"] = df["ActionGeo_CountryCode"].isin(
        ["IR", "IZ", "IS", "SY", "LE", "SA", "AE", "YM", "JO", "TU"]
    )

    df["usa_china_relation"] = (
        ((df["Actor1CountryCode"] == "USA") & (df["Actor2CountryCode"] == "CHN"))
        | ((df["Actor1CountryCode"] == "CHN") & (df["Actor2CountryCode"] == "USA"))
    )

    df["usa_russia_relation"] = (
        ((df["Actor1CountryCode"] == "USA") & (df["Actor2CountryCode"] == "RUS"))
        | ((df["Actor1CountryCode"] == "RUS") & (df["Actor2CountryCode"] == "USA"))
    )

    weekly = df.groupby("week").agg(
        nb_events=("EventCode", "count"),
        total_mentions=("NumMentions", "sum"),
        total_articles=("NumArticles", "sum"),

        avg_goldstein=("GoldsteinScale", "mean"),
        min_goldstein=("GoldsteinScale", "min"),
        avg_tone=("AvgTone", "mean"),
        min_tone=("AvgTone", "min"),

        tension_score=("negative_intensity", "sum"),
        cooperation_score=("positive_intensity", "sum"),

        verbal_conflict_mentions=("NumMentions", lambda x: x[df.loc[x.index, "is_verbal_conflict"]].sum()),
        material_conflict_mentions=("NumMentions", lambda x: x[df.loc[x.index, "is_material_conflict"]].sum()),
        verbal_coop_mentions=("NumMentions", lambda x: x[df.loc[x.index, "is_verbal_coop"]].sum()),
        material_coop_mentions=("NumMentions", lambda x: x[df.loc[x.index, "is_material_coop"]].sum()),

        major_country_events=("major_actor_event", "sum"),
        major_country_conflict_mentions=("NumMentions", lambda x: x[
            df.loc[x.index, "major_actor_event"] & df.loc[x.index, "is_material_conflict"]
        ].sum()),

        usa_conflict_mentions=("NumMentions", lambda x: x[
            df.loc[x.index, "usa_event"] & df.loc[x.index, "is_material_conflict"]
        ].sum()),

        china_conflict_mentions=("NumMentions", lambda x: x[
            df.loc[x.index, "china_event"] & df.loc[x.index, "is_material_conflict"]
        ].sum()),

        russia_conflict_mentions=("NumMentions", lambda x: x[
            df.loc[x.index, "russia_event"] & df.loc[x.index, "is_material_conflict"]
        ].sum()),

        middle_east_conflict_mentions=("NumMentions", lambda x: x[
            df.loc[x.index, "middle_east_event"] & df.loc[x.index, "is_material_conflict"]
        ].sum()),

        usa_china_tension=("negative_intensity", lambda x: x[
            df.loc[x.index, "usa_china_relation"]
        ].sum()),

        usa_russia_tension=("negative_intensity", lambda x: x[
            df.loc[x.index, "usa_russia_relation"]
        ].sum())
    ).reset_index()

    weekly["conflict_mentions"] = (
        weekly["verbal_conflict_mentions"] + weekly["material_conflict_mentions"]
    )

    weekly["coop_mentions"] = (
        weekly["verbal_coop_mentions"] + weekly["material_coop_mentions"]
    )

    weekly["conflict_ratio"] = weekly["conflict_mentions"] / (weekly["total_mentions"] + 1)
    weekly["material_conflict_ratio"] = weekly["material_conflict_mentions"] / (weekly["total_mentions"] + 1)
    weekly["major_conflict_ratio"] = weekly["major_country_conflict_mentions"] / (weekly["total_mentions"] + 1)

    weekly = weekly.sort_values("week").reset_index(drop=True)

    return weekly


# ============================================================
# 3. AJOUT DES DONNÉES MARCHÉ
# ============================================================

def add_market_target(df_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    print(f"Téléchargement des données marché : {ticker}")

    market = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

    if market.empty:
        raise ValueError("Yahoo Finance n'a retourné aucune donnée.")

    prices = market["Adj Close"] if "Adj Close" in market.columns else market["Close"]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    weekly_price = prices.resample("W-MON").last()
    weekly_return = weekly_price.pct_change()

    market_df = pd.DataFrame(index=weekly_price.index)
    market_df["weekly_price"] = weekly_price
    market_df["weekly_return_current"] = weekly_return

    # Volatilité
    market_df["vol_4w"] = market_df["weekly_return_current"].rolling(4).std()
    market_df["vol_8w"] = market_df["weekly_return_current"].rolling(8).std()
    market_df["vol_12w"] = market_df["weekly_return_current"].rolling(12).std()

    # Momentum
    market_df["momentum_4w"] = weekly_price.pct_change(4)
    market_df["momentum_12w"] = weekly_price.pct_change(12)
    market_df["momentum_26w"] = weekly_price.pct_change(26)

    # Drawdown
    rolling_max_26w = weekly_price.rolling(26).max()
    market_df["drawdown_26w"] = weekly_price / rolling_max_26w - 1

    # Target return future
    market_df["future_return_4w"] = weekly_price.shift(-4) / weekly_price - 1

    # VIX
    vix = yf.download(
        "^VIX",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

    if not vix.empty:
        vix_price = vix["Adj Close"] if "Adj Close" in vix.columns else vix["Close"]

        if isinstance(vix_price, pd.DataFrame):
            vix_price = vix_price.iloc[:, 0]

        weekly_vix = vix_price.resample("W-MON").last()

        market_df["vix"] = weekly_vix
        market_df["vix_delta_1w"] = market_df["vix"].diff()
        market_df["vix_zscore_52w"] = (
            market_df["vix"] - market_df["vix"].rolling(52, min_periods=20).mean()
        ) / (
            market_df["vix"].rolling(52, min_periods=20).std() + 1e-9
        )

    market_df = market_df.reset_index().rename(columns={"Date": "week"})
    market_df["week"] = pd.to_datetime(market_df["week"])
    df_features["week"] = pd.to_datetime(df_features["week"])

    final_df = df_features.merge(market_df, on="week", how="left")

    final_df = final_df.dropna().reset_index(drop=True)

    return final_df


# ============================================================
# 4. CIBLE STRESS / CRISE
# ============================================================

def create_crisis_target(df: pd.DataFrame) -> pd.DataFrame:
    print("Création de la cible stress/crise avec rendement futur 4 semaines...")

    df = df.sort_values("week").reset_index(drop=True)

    def label_crisis(future_return):
        if future_return <= -0.15:
            return 1
        else:
            return 0

    df["crisis_level"] = df["future_return_4w"].apply(label_crisis)

    df = df.dropna(subset=["future_return_4w", "crisis_level"]).reset_index(drop=True)

    return df

# ============================================================
# 5. LAGS, DELTAS, Z-SCORES
# ============================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Ajout des lags, variations et z-scores...")

    df = df.sort_values("week").reset_index(drop=True)

    base_cols = [
        "nb_events",
        "total_mentions",
        "avg_goldstein",
        "avg_tone",
        "tension_score",
        "cooperation_score",
        "material_conflict_mentions",
        "verbal_conflict_mentions",
        "conflict_ratio",
        "material_conflict_ratio",
        "major_conflict_ratio",
        "major_country_conflict_mentions",
        "usa_conflict_mentions",
        "china_conflict_mentions",
        "russia_conflict_mentions",
        "middle_east_conflict_mentions",
        "usa_china_tension",
        "usa_russia_tension"
    ]

    for col in base_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_delta1"] = df[col] - df[f"{col}_lag1"]

        rolling_mean = df[col].rolling(window=52, min_periods=20).mean()
        rolling_std = df[col].rolling(window=52, min_periods=20).std()

        df[f"{col}_zscore_52w"] = (df[col] - rolling_mean) / (rolling_std + 1e-9)
        # Interactions GDELT x marché
    df["tension_x_vol_4w"] = df["tension_score_zscore_52w"] * df["vol_4w"]
    df["conflict_x_drawdown"] = df["material_conflict_ratio_zscore_52w"] * abs(df["drawdown_26w"])
    df["vix_x_tension"] = df["vix_zscore_52w"] * df["tension_score_zscore_52w"]
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# 6. ENTRAÎNEMENT
# ============================================================

def train_model(df: pd.DataFrame):
    print("Préparation train/test temporel...")

    features = [
        "nb_events",
        "total_mentions",
        "avg_goldstein",
        "avg_tone",
        "tension_score",
        "cooperation_score",
        "material_conflict_mentions",
        "verbal_conflict_mentions",
        "conflict_ratio",
        "material_conflict_ratio",
        "major_conflict_ratio",
        "major_country_conflict_mentions",
        "usa_conflict_mentions",
        "china_conflict_mentions",
        "russia_conflict_mentions",
        "middle_east_conflict_mentions",
        "usa_china_tension",
        "usa_russia_tension",
        "weekly_return_current",
    ]

    temporal_suffixes = ["_lag1", "_lag2", "_delta1", "_zscore_52w"]

    extra_features = []
    for col in features:
        if col != "weekly_return_current":
            for suffix in temporal_suffixes:
                candidate = f"{col}{suffix}"
                if candidate in df.columns:
                    extra_features.append(candidate)

    features = features + extra_features

    train = df[df["week"] < SPLIT_DATE].copy()
    test = df[df["week"] >= SPLIT_DATE].copy()

    X_train = train[features]
    y_train = train["crisis_level"]

    X_test = test[features]
    y_test = test["crisis_level"]

    print("\nDistribution des classes - train :")
    print(y_train.value_counts().sort_index())

    print("\nDistribution des classes - test :")
    print(y_test.value_counts().sort_index())
    print(f"\nNombre de lignes train : {len(train)}")
    print(f"Nombre de lignes test  : {len(test)}")
    model = CatBoostClassifier(
        iterations=100,
        depth=2,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        l2_leaf_reg=10,
        random_strength=2,
        bagging_temperature=1,
        random_state=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    y_proba = model.predict_proba(X_test)
    stress_proba = y_proba[:, 1]

    THRESHOLD = 0.55
    y_pred = (stress_proba >= THRESHOLD).astype(int)

    print("\n==============================")
    print("RÉSULTATS MODÈLE STRESS / CRISE")
    print("==============================")

    print(f"Accuracy simple : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nRapport de classification :")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Normal",
            "Stress"
        ],
        zero_division=0
    ))

    results = test[["week", "weekly_return_current", "future_return_4w", "crisis_level"]].copy()
    results["predicted_crisis_level"] = y_pred
    results["proba_normal"] = y_proba[:, 0]
    results["proba_stress"] = y_proba[:, 1]

    results_path = os.path.join(OUTPUT_DIR, "stress_crisis_predictions.csv")
    results.to_csv(results_path, index=False)

    print(f"\nPrédictions sauvegardées dans : {results_path}")

    return model, features, X_test, y_test, y_pred, results


# ============================================================
# 7. VISUALISATIONS
# ============================================================

def plot_results(model, features, y_test, y_pred):
    print("Création des graphiques...")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Normal", "Stress", "Crise sévère"],
        yticklabels=["Normal", "Stress", "Crise sévère"]
    )
    plt.title("Matrice de confusion - Détection stress/crise")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_stress_crisis.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()

    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False).head(25)

    plt.figure(figsize=(10, 8))
    importances.sort_values().plot(kind="barh")
    plt.title("Top 25 variables les plus importantes")
    plt.tight_layout()

    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_stress_crisis.png")
    plt.savefig(imp_path, dpi=300)
    plt.show()

    print(f"Graphiques sauvegardés dans : {OUTPUT_DIR}")


# ============================================================
# 8. SAUVEGARDE MODÈLE + MÉTADONNÉES
# ============================================================

def save_model(model, features):
    bundle = {
        "model": model,
        "features": features,
        "ticker": TICKER,
        "target": "crisis_level",
        "classes": {
            0: "Normal",
            1: "Stress"
        },
        "description": "Modèle de détection de stress/crise basé sur GDELT + rendement marché courant"
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")


# ============================================================
# 9. PIPELINE COMPLET
# ============================================================

if __name__ == "__main__":
    df_raw = load_gdelt(INPUT_FILE)

    df_weekly = create_gdelt_weekly_features(df_raw)

    weekly_path = os.path.join(OUTPUT_DIR, "gdelt_weekly_stress_features.csv")
    df_weekly.to_csv(weekly_path, index=False)
    print(f"Features GDELT hebdo sauvegardées dans : {weekly_path}")

    df_model = add_market_target(df_weekly, TICKER)

    df_model = create_crisis_target(df_model)

    df_model = add_temporal_features(df_model)

    dataset_path = os.path.join(OUTPUT_DIR, "stress_crisis_training_dataset.csv")
    df_model.to_csv(dataset_path, index=False)
    print(f"Dataset final sauvegardé dans : {dataset_path}")

    model, features, X_test, y_test, y_pred, results = train_model(df_model)

    plot_results(model, features, y_test, y_pred)

    save_model(model, features)
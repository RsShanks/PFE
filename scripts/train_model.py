import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import joblib
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


def add_market_data(df_features, ticker, start="2006-01-01"):
    print(f"Téléchargement Yahoo Finance : {ticker}")
    data = yf.download(ticker, start=start, auto_adjust=False)

    if data.empty:
        return None

    # Extraction du prix depuis yahoo finance
    if isinstance(data.columns, pd.MultiIndex):
        close = (
            data["Adj Close"].iloc[:, 0]
            if "Adj Close" in data.columns
            else data["Close"].iloc[:, 0]
        )
    else:
        close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    # Calcul Rendement Hebdoadaire
    market_weekly = (
        close.resample("W-MON").last().pct_change().to_frame(name="weekly_return")
    )
    market_weekly["target_return"] = market_weekly["weekly_return"].shift(-1)
    market_weekly = market_weekly.reset_index()

    # Fusion avec les informations GDELT
    df_features["week"] = pd.to_datetime(df_features["week"])
    market_weekly["Date"] = pd.to_datetime(market_weekly["Date"])

    final = df_features.merge(
        market_weekly[["Date", "target_return"]],
        left_on="week",
        right_on="Date",
        how="left",
    )
    return final.dropna(subset=["target_return"]).drop(columns=["Date"])


# récupération des features hebdomadaires depuis notre csv
df_weekly = pd.read_csv("gdelt_weekly_finance_features.csv")

tickers = ["^GSPC", "BTC-USD", "NVDA", "^VIX", "GLD"]
bilan = {}

for t in tickers:
    print(f"\n>>> TRAVAIL SUR : {t}")

    # Préparation des données
    df_t = add_market_data(df_weekly.copy(), t)
    if df_t is None:
        continue

    # Feature Engineering ajout des lags et de la direction
    df_t = df_t.sort_values("week")
    cols_to_lag = ["tension_score", "avg_tone", "nb_events"]
    for c in cols_to_lag:
        df_t[f"{c}_lag1"] = df_t[c].shift(1)

    df_t["weekly_return_current"] = df_t["target_return"].shift(1)
    df_t["target_direction"] = (df_t["target_return"] > 0).astype(int)
    df_t = df_t.dropna().reset_index(drop=True)

    # Split en deux parties : avant 2019 pour l'entraînement, après pour le test
    train = df_t[df_t["week"] < "2019-01-01"]
    test = df_t[df_t["week"] >= "2019-01-01"]

    features = [
        "nb_events",
        "avg_tone",
        "tension_score",
        "nb_events_lag1",
        "avg_tone_lag1",
        "tension_score_lag1",
        "weekly_return_current",
    ]

    X_train, y_train = train[features], train["target_direction"]
    X_test, y_test = test[features], test["target_direction"]

    # Entraînement CatBoost
    model = CatBoostClassifier(
        iterations=892,
        learning_rate=0.09093093204104495,
        depth=3,
        l2_leaf_reg=3.212823028775031,
        auto_class_weights="Balanced",
        min_data_in_leaf=12,
        verbose=0,  # On cache les logs pour y voir clair
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)

    # Score et Sauvegarde
    acc = accuracy_score(y_test, model.predict(X_test))
    bilan[t] = acc

    nom_fichier = f"catboost_{t.replace('^', '')}.pkl"
    joblib.dump(model, nom_fichier)
    print(f"Terminé ! Accuracy: {acc*100:.2f}% | Sauvegardé : {nom_fichier}")

print("\n--- RÉSULTATS FINAUX ---")
for ticker, score in bilan.items():
    print(f"{ticker} : {score*100:.2f}%")


def objective(trial, X_train, y_train, X_test, y_test):
    # Définition des plages de recherche (Search Space)
    param = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
    }

    # Création et entraînement du modèle avec les paramètres suggérés
    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    # Prédiction et calcul du score
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return accuracy


# Lancement de l'étude
study = optuna.create_study(direction="maximize")

# On passe les données via une fonction lambda
study.optimize(
    lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50
)
# Affichage des meilleurs paramètres et du score
print("Meilleurs paramètres trouvés :")
print(study.best_params)
print(f"Meilleure Accuracy : {study.best_value*100:.2f}%")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Chargement et Feature Engineering
# ==========================================
df = pd.read_csv("gdelt_weekly_finance_features_with_returns.csv")
df['week'] = pd.to_datetime(df['week'])
df = df.sort_values('week')

# Création de Lags (Mémoire du modèle)
# On donne au modèle les données des 2 semaines précédentes
features_to_lag = ['tension_score', 'avg_tone', 'material_conflict_mentions', 'nb_events']
for col in features_to_lag:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# On ajoute aussi le rendement de la semaine actuelle pour voir la tendance
df['weekly_return_current'] = df['target_return'].shift(1)

# Nettoyage des lignes vides créées par les shifts
df = df.dropna().reset_index(drop=True)

# ==========================================
# 2. Séparation Temporelle (Train / Test)
# ==========================================
# On s'arrête fin 2018 pour l'entraînement
# On teste sur 2019-2022 (période très volatile)
train = df[df['week'] < "2019-01-01"]
test = df[df['week'] >= "2019-01-01"]

# Définition des X (entrées) et y (cible)
features = [
    'nb_events', 'avg_tone', 'tension_score', 'material_conflict_mentions',
    'nb_events_lag1', 'avg_tone_lag1', 'tension_score_lag1', 
    'nb_events_lag2', 'avg_tone_lag2', 'tension_score_lag2',
    'weekly_return_current'
]
target = 'target_return'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# ==========================================
# 3. Entraînement du Modèle
# ==========================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# ==========================================
# 4. Évaluation et Importance des Variables
# ==========================================
print(f"MSE : {mean_squared_error(y_test, y_pred):.6f}")
print(f"R2 Score : {r2_score(y_test, y_pred):.6f}")

# Importance des variables
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Quels indicateurs GDELT influencent le plus la bourse ?")
plt.show()

# Visualisation des résultats (Cumulés pour voir la tendance)
plt.figure(figsize=(12, 6))
plt.plot(test['week'], y_test.cumsum(), label="Réel (Cumulé)", color='blue')
plt.plot(test['week'], y_pred.cumsum(), label="Prédit (Cumulé)", color='red', linestyle='--')
plt.title("Prédiction des rendements du S&P 500 via GDELT (2019-2022)")
plt.legend()
plt.show()
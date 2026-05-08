import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Chargement et Feature Engineering
# ==========================================
df = pd.read_csv("scripts/events/gdelt_weekly_finance_features_with_returns.csv")
df['week'] = pd.to_datetime(df['week'])
df = df.sort_values('week')

# Création de Lags (Mémoire du modèle)
features_to_lag = ['tension_score', 'avg_tone', 'mat_conf_mentions', 'nb_events']
for col in features_to_lag:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# Le momentum (rendement de la semaine actuelle)
df['weekly_return_current'] = df['target_return'].shift(1)

# Nettoyage des NaN créés par les shifts
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# LA NOUVEAUTÉ : Création de la cible de classification
# 1 = Hausse (Positif), 0 = Baisse (Négatif ou Nul)
# ---------------------------------------------------------
df['target_direction'] = (df['target_return'] > 0).astype(int)

# ==========================================
# 2. Séparation Temporelle (Train / Test)
# ==========================================
train = df[df['week'] < "2019-01-01"]
test = df[df['week'] >= "2019-01-01"]

# Définition des entrées (X) et de la NOUVELLE cible (y)
features = [
    'nb_events', 'avg_tone', 'tension_score', 'mat_conf_mentions',
    'nb_events_lag1', 'avg_tone_lag1', 'tension_score_lag1', 
    'nb_events_lag2', 'avg_tone_lag2', 'tension_score_lag2',
    'weekly_return_current'
]
target = 'target_direction' # On utilise la colonne binaire

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# ==========================================
# 3. Entraînement du Modèle (Classifier)
# ==========================================
# max_depth=5 aide à éviter que le modèle n'apprenne par coeur (overfitting) le bruit
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# ==========================================
# 4. Évaluation du Modèle
# ==========================================
print("=== RÉSULTATS DE LA CLASSIFICATION ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Précision globale) : {accuracy * 100:.2f}%\n")

print("Rapport Détaillé :")
print(classification_report(y_test, y_pred, target_names=["Baisse (0)", "Hausse (1)"]))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Prédit Baisse", "Prédit Hausse"], 
            yticklabels=["Réel Baisse", "Réel Hausse"])
plt.title("Matrice de Confusion (2019-2022)")
plt.ylabel('Vraie Tendance')
plt.xlabel('Tendance Prédite')
plt.show()

# Importance des variables
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Importance des indicateurs GDELT (Classification Hausse/Baisse)")
plt.tight_layout()
plt.show()

joblib.dump(model, 'api/data/models/random_forest_model.pkl')
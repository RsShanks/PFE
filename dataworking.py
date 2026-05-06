import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("gdelt_clean_mapped.csv")

df["SQLDATE"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d")
df["year"] = df["SQLDATE"].dt.year
df["month"] = df["SQLDATE"].dt.to_period("M").dt.to_timestamp()

# sécurité
df["EventDescriptionFinal"] = df["EventDescriptionFinal"].fillna(df["EventRootDescription"])
df = df.dropna(subset=["EventDescriptionFinal", "GoldsteinScale", "NumMentions", "AvgTone"])

# =========================
# 1. Indice mensuel de tension géopolitique
#    Score négatif pondéré par NumMentions
# =========================

df["negative_intensity"] = df["GoldsteinScale"].clip(upper=0).abs() * df["NumMentions"]

monthly_tension = (
    df.groupby("month")["negative_intensity"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 5))
plt.plot(monthly_tension["month"], monthly_tension["negative_intensity"])
plt.title("Indice mensuel de tension géopolitique pondéré par les mentions")
plt.xlabel("Date")
plt.ylabel("Somme |Goldstein négatif| × NumMentions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 2. Part des conflits matériels dans le temps
#    Plus pertinent que compter toutes les lignes
# =========================

monthly_quad = (
    df.groupby(["month", "QuadClassLabel"])["NumMentions"]
    .sum()
    .reset_index()
)

monthly_quad_pivot = monthly_quad.pivot(
    index="month",
    columns="QuadClassLabel",
    values="NumMentions"
).fillna(0)

monthly_quad_share = monthly_quad_pivot.div(monthly_quad_pivot.sum(axis=1), axis=0)

plt.figure(figsize=(12, 5))
for col in monthly_quad_share.columns:
    plt.plot(monthly_quad_share.index, monthly_quad_share[col], label=col)

plt.title("Évolution de la structure des événements GDELT")
plt.xlabel("Date")
plt.ylabel("Part des mentions")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 3. Top pays les plus liés à des événements négatifs
# =========================

country_risk = (
    df.groupby("Actor1CountryName")
    .agg(
        nb_events=("EventCode", "count"),
        total_mentions=("NumMentions", "sum"),
        avg_goldstein=("GoldsteinScale", "mean"),
        risk_score=("negative_intensity", "sum")
    )
    .sort_values("risk_score", ascending=False)
    .head(15)
)

print("\nTop pays par score de tension :")
print(country_risk)

plt.figure(figsize=(10, 5))
country_risk["risk_score"].sort_values().plot(kind="barh")
plt.title("Pays les plus associés à des événements négatifs pondérés")
plt.xlabel("Score de tension")
plt.tight_layout()
plt.show()


# =========================
# 4. Les motifs les plus négatifs ET fréquents
#    Pas juste les plus fréquents
# =========================

event_risk = (
    df.groupby("EventDescriptionFinal")
    .agg(
        nb_events=("EventCode", "count"),
        total_mentions=("NumMentions", "sum"),
        avg_goldstein=("GoldsteinScale", "mean"),
        avg_tone=("AvgTone", "mean"),
        risk_score=("negative_intensity", "sum")
    )
)

# On enlève les motifs trop rares
event_risk = event_risk[event_risk["nb_events"] >= 100]

top_event_risk = event_risk.sort_values("risk_score", ascending=False).head(15)

print("\nMotifs les plus risqués :")
print(top_event_risk)

plt.figure(figsize=(10, 6))
top_event_risk["risk_score"].sort_values().plot(kind="barh")
plt.title("Motifs d'événements les plus négatifs et médiatiquement importants")
plt.xlabel("Score de tension")
plt.tight_layout()
plt.show()


# =========================
# 5. Matrice Pays x Type d'événement
#    Pour voir le profil géopolitique des pays
# =========================

top_countries = country_risk.index.tolist()

matrix = (
    df[df["Actor1CountryName"].isin(top_countries)]
    .pivot_table(
        index="Actor1CountryName",
        columns="QuadClassLabel",
        values="NumMentions",
        aggfunc="sum",
        fill_value=0
    )
)

matrix_share = matrix.div(matrix.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
matrix_share.plot(kind="bar", stacked=True, figsize=(11, 6))
plt.title("Profil des événements par pays")
plt.xlabel("Pays")
plt.ylabel("Part des mentions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# =========================
# 6. Détection des mois de crise
#    Mois où l'indice de tension est anormalement élevé
# =========================

monthly_tension["zscore"] = (
    monthly_tension["negative_intensity"] - monthly_tension["negative_intensity"].mean()
) / monthly_tension["negative_intensity"].std()

crisis_months = monthly_tension[monthly_tension["zscore"] >= 2].copy()

print("\nMois de tension anormale :")
print(crisis_months.sort_values("zscore", ascending=False).head(20))

plt.figure(figsize=(12, 5))
plt.plot(monthly_tension["month"], monthly_tension["zscore"])
plt.axhline(2, linestyle="--")
plt.title("Détection des mois de tension anormale")
plt.xlabel("Date")
plt.ylabel("Z-score de tension")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 7. Pour chaque année : motif dominant en tension
# =========================

year_event = (
    df.groupby(["year", "EventDescriptionFinal"])["negative_intensity"]
    .sum()
    .reset_index()
)

idx = year_event.groupby("year")["negative_intensity"].idxmax()
top_event_by_year = year_event.loc[idx].sort_values("year")

print("\nMotif le plus tendu par année :")
print(top_event_by_year)

plt.figure(figsize=(12, 5))
plt.bar(top_event_by_year["year"].astype(str), top_event_by_year["negative_intensity"])
plt.title("Intensité du motif le plus tendu par année")
plt.xlabel("Année")
plt.ylabel("Score de tension du motif dominant")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 8. Préparation d'un dataset mensuel exportable pour finance
# =========================

finance_features = df.groupby("month").agg(
    nb_events=("EventCode", "count"),
    total_mentions=("NumMentions", "sum"),
    avg_goldstein=("GoldsteinScale", "mean"),
    avg_tone=("AvgTone", "mean"),
    tension_score=("negative_intensity", "sum"),
    material_conflict_mentions=("NumMentions", lambda x: x[df.loc[x.index, "QuadClassLabel"] == "Material conflict"].sum()),
    verbal_conflict_mentions=("NumMentions", lambda x: x[df.loc[x.index, "QuadClassLabel"] == "Verbal conflict"].sum()),
).reset_index()

finance_features.to_csv("gdelt_monthly_finance_features.csv", index=False)

print("\nDataset finance exporté : gdelt_monthly_finance_features.csv")
print(finance_features.head())
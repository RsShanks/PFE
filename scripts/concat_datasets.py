import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# =========================
# 1. Charger les fichiers
# =========================

debut = pd.read_csv("events/data_06_09.csv")
milieu = pd.read_csv("events/data_10_18.csv")
fin = pd.read_csv("events/data_19_22.csv")
# Empiler les lignes
df = pd.concat([debut, milieu, fin], ignore_index=True)

print("Taille après concat :", df.shape)
print(df.head())

# =========================
# 2. Nettoyage des codes
# =========================

df["EventCode"] = df["EventCode"].astype(str).str.zfill(3)
df["EventRootCode"] = df["EventRootCode"].astype(str).str.zfill(2)
print(df["EventCode"].head(20))
for col in ["Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# =========================
# 3. Mapping pays
# =========================

country_map = pd.read_csv(
    "https://www.gdeltproject.org/data/lookups/CAMEO.country.txt",
    sep="\t",
    header=None,
    names=["CountryCode", "CountryName"]
)

country_map["CountryCode"] = country_map["CountryCode"].astype(str).str.strip()

df = df.merge(
    country_map.rename(columns={
        "CountryCode": "Actor1CountryCode",
        "CountryName": "Actor1CountryName"
    }),
    on="Actor1CountryCode",
    how="left"
)

df = df.merge(
    country_map.rename(columns={
        "CountryCode": "Actor2CountryCode",
        "CountryName": "Actor2CountryName"
    }),
    on="Actor2CountryCode",
    how="left"
)

df = df.merge(
    country_map.rename(columns={
        "CountryCode": "ActionGeo_CountryCode",
        "CountryName": "ActionGeo_CountryName"
    }),
    on="ActionGeo_CountryCode",
    how="left"
)

# =========================
# 4. Mapping EventCode
# =========================

event_map = pd.read_csv(
    "https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt",
    sep="\t",
    header=None,
    names=["EventCode", "EventDescription"]
)

event_map = event_map[event_map["EventCode"] != "CAMEOEVENTCODE"]

event_map["EventCode"] = event_map["EventCode"].astype(str).str.zfill(3)

df = df.merge(event_map, on="EventCode", how="left")

# =========================
# 5. Mapping QuadClass
# =========================

quad_map = {
    1: "Verbal cooperation",
    2: "Material cooperation",
    3: "Verbal conflict",
    4: "Material conflict"
}

df["QuadClassLabel"] = df["QuadClass"].map(quad_map)

# =========================
# 6. Vérification
# =========================

print("\nColonnes finales :")
print(df.columns)

print("\nAperçu complet :")
print(df.head())

print("\nNaN sur les mappings pays :")
print(df[[
    "Actor1CountryName",
    "Actor2CountryName",
    "ActionGeo_CountryName",
    "EventDescription"
]].isna().sum())
# fallback description à partir de EventRootCode si EventDescription est NaN
root_map = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use unconventional mass violence"
}

df["EventRootDescription"] = df["EventRootCode"].map(root_map)

df["EventDescriptionFinal"] = df["EventDescription"].fillna(
    df["EventRootDescription"]
)
# =========================
# 7. Sauvegarde
# =========================
print(df["EventDescription"].isna().sum())
print(df["Actor1CountryName"].isna().sum())

print(df[df["Actor1CountryName"].isna()]["Actor1CountryCode"].unique())
# =========================
# Vérification cohérence EventCode / EventRootCode
# =========================

# Le root théorique = 2 premiers chiffres de EventCode
df["EventRootFromEventCode"] = df["EventCode"].astype(str).str[:2]

# Vérifie que EventRootCode correspond bien au début de EventCode
incoherent_root = df[
    df["EventRootFromEventCode"] != df["EventRootCode"]
]

print("\nNombre d'incohérences EventCode / EventRootCode :", len(incoherent_root))

if len(incoherent_root) > 0:
    print(incoherent_root[
        ["EventCode", "EventRootCode", "EventRootFromEventCode", "EventDescription", "EventRootDescription"]
    ].drop_duplicates().head(20))
else:
    print("EventCode et EventRootCode sont cohérents.")

# Vérifie les descriptions manquantes après fallback
missing_final_desc = df["EventDescriptionFinal"].isna().sum()
print("\nDescriptions finales encore manquantes :", missing_final_desc)

# Affiche les EventCode sans description détaillée mais avec fallback root
fallback_used = df[
    df["EventDescription"].isna() & df["EventDescriptionFinal"].notna()
]

print("\nNombre de lignes où le fallback EventRootDescription est utilisé :", len(fallback_used))

print("\nExemples de fallback utilisé :")
print(
    fallback_used[
        ["EventCode", "EventRootCode", "EventDescription", "EventRootDescription", "EventDescriptionFinal"]
    ].drop_duplicates().head(20)
)
print(df[["EventCode", "EventRootCode", "EventDescription", "EventRootDescription", "EventDescriptionFinal"]
    ].drop_duplicates().head(20))
df.to_csv("events/gdelt_clean_mapped.csv", index=False, encoding="utf-8")
print("\nFichier sauvegardé : scripts/events/gdelt_clean_mapped.csv")

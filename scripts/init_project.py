import subprocess
import sys
import os

def run_script(script_path):
    """Lance un script Python et attend la fin de son exécution."""
    print(f"\n--- 🚀 Lancement de : {script_path} ---")
    try:
        # On utilise sys.executable pour être sûr d'utiliser le Python de l'env virtuel
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ {script_path} terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de {script_path}.")
        sys.exit(1)

def main():
    print("=== 🌍 Initialisation du Système GeoPredict ===")

    # 1. Création des dossiers de base si manquants
    directories = [
        "api/data/models",
        "scripts/events",
        "frontend/static",
        "frontend/templates"
    ]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📁 Dossier créé : {d}")

    # 2. Pipeline de Données (Nettoyage GDELT + Yahoo Finance)
    # Ce script génère : gdelt_weekly_finance_features_with_returns.csv
    run_script("scripts/dataworking.py") 

    # 3. Entraînement du Modèle ML
    # Ce script génère : random_forest_model.pkl (ou catboost)
    # run_script("scripts/train_model.py")

    # 4. Mise à jour du Cache Live
    # Ce script génère : cache_features.json pour l'inférence immédiate
    run_script("scripts/update_cache.py")

    print("\n" + "="*40)
    print("  TOUT EST PRÊT !")
    print("Vous pouvez maintenant lancer l'API avec :")
    print("uvicorn api.main:app --reload")
    print("="*40)

if __name__ == "__main__":
    main()
from fastapi import APIRouter, HTTPException
import pandas as pd

# import gdelt
import numpy as np
import os
import json

router = APIRouter(prefix="/events", tags=["Événements GDELT Temps Réel"])


@router.get("/daily-focus")
def get_daily_events():
    """
    Récupère instantanément les trois derniers événements géopolitiques majeurs
    """
    try:
        cache_path = "api/data/weekly_radar.json"

        # Sécurité : si le fichier n'existe pas encore
        if not os.path.exists(cache_path):
            raise HTTPException(
                status_code=503,
                detail="Le radar hebdomadaire est en cours d'initialisation. Veuillez réessayer dans quelques instants.",
            )

        with open(cache_path, "r", encoding="utf-8") as f:
            radar_data = json.load(f)
            radar_data["events"] = sorted(
                radar_data["events"], key=lambda x: x.get("RiskScore", 0), reverse=True
            )[:3]
        return radar_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la lecture du radar : {str(e)}"
        )


@router.get("/weekly-focus")
def get_weekly_focus_events():
    """
    Récupère instantanément les événements géopolitiques majeurs
    pré-calculés de la semaine passée (Zéro surcharge RAM).
    """
    try:
        cache_path = "api/data/weekly_radar.json"

        # Sécurité : si le fichier n'existe pas encore
        if not os.path.exists(cache_path):
            raise HTTPException(
                status_code=503,
                detail="Le radar hebdomadaire est en cours d'initialisation. Veuillez réessayer dans quelques instants.",
            )

        with open(cache_path, "r", encoding="utf-8") as f:
            radar_data = json.load(f)

        return radar_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la lecture du radar : {str(e)}"
        )

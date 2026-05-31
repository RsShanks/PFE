from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api.router import predict, events

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Bourse & Géopolitique (GDELT)",
    description="Prédiction des mouvements boursiers basés sur l'actualité mondiale.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes API (Le backend)
app.include_router(predict.router, prefix="/api/v1")
app.include_router(events.router, prefix="/api/v1")

# ==========================================
# CONFIGURATION FRONT-END (JINJA2)
# ==========================================

# 1. Monter le dossier statique
app.mount("/static", StaticFiles(directory="api/frontend/static"), name="static")

# 2. Déclarer le dossier des templates HTML
templates = Jinja2Templates(directory="api/frontend/templates")

# --- ROUTES FRONT-END (UI) ---


@app.get("/")
def render_home(request: Request):
    """Page d'accueil"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/radar")
def render_radar(request: Request):
    """Page des actualités GDELT"""
    return templates.TemplateResponse(request=request, name="radar.html")


@app.get("/simulator")
def render_simulator(request: Request):
    """Page du Simulateur de Crise"""
    return templates.TemplateResponse(request=request, name="simulator.html")


@app.get("/risque-crise")
def render_risque_crise(request: Request):
    return templates.TemplateResponse(request=request, name="risque_crise.html")

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI,Request, Form
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pathlib import Path
import pandas as pd
import joblib
import time

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()

templates= Jinja2Templates(directory="templates")

model = joblib.load(BASE_DIR / "models"/ "diabete_model_svc.pkl")
std = joblib.load(BASE_DIR / "models"/ "scaler.pkl")


PREDICTION_COUNTER = Counter(
    "diabetes_prediction_total",
    "Nombre total de prédictions"
)

POSITIVE_PREDICTION_COUNTER = Counter(
    "diabetes_positive_prediction_total",
    "Nombre de prédictions positives"
)

PREDICTION_DURATION = Histogram(
    "diabetes_prediction_duration_seconds",
    "Temps d'exécution des prédictions"
)

ERROR_COUNTER = Counter(
    "diabetes_prediction_errors_total",
    "Nombre d'erreurs de prédiction"
)


class Patient(BaseModel):
    pregnancies: int
    glucose: float
    diastolic: float
    triceps: float
    insulin: float
    bmi: float
    dpf: float
    age: int


@app.get("/")
def accueil(request:Request):
    return templates.TemplateResponse(
        request=request,
        name="accueil.html"
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: Patient):
    try:
        start_time = time.time()

        dt = data.model_dump()

        df = pd.DataFrame([dt])

        data_std = std.transform(df)

        prediction = model.predict(data_std)

        prediction_value = int(prediction[0])

        # Compteur total
        PREDICTION_COUNTER.inc()

        # Compteur diabète détecté
        if prediction_value == 1:
            POSITIVE_PREDICTION_COUNTER.inc()

        # Temps d'inférence
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        return {
            "resultat": prediction_value,
            "temps_prediction": round(duration, 5),
            "patient": dt
        }
    except Exception as e:
        ERROR_COUNTER.inc()
        return {"error": str(e)}


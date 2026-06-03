from fastapi import FastAPI,Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

templates= Jinja2Templates(directory="templates")

model = joblib.load("../models/diabete_model_svc.pkl")
std = joblib.load("../models/scaler.pkl")

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

@app.post("/predict")
def predict(data: Patient):

    dt = data.model_dump()

    df = pd.DataFrame([dt])

    data_std = std.transform(df)

    predictions = model.predict(data_std)

    return {
        "resultat": int(predictions[0]),
        "patient": dt
    }


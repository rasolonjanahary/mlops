import joblib
import numpy as p
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_json = {
  "pregnancies": 6,
  "glucose": 149,
  "diastolic": 74,
  "triceps": 36,
  "insulin": 82,
  "bmi": 38.1,
  "dpf": 0.650,
  "age": 52
}
data = pd.DataFrame([data_json])

model = joblib.load(BASE_DIR/"models"/"diabete_model_svc.pkl")
std = joblib.load(BASE_DIR/"models"/"scaler.pkl")

data_std = std.transform(data)

predictions = model.predict(data_std)

print(predictions)


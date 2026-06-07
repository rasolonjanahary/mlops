import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

data = pd.read_csv(BASE_DIR/"data"/"data_clean.csv")

X = data[["pregnancies","glucose","diastolic","triceps","insulin","bmi","dpf","age"]]
y = data["diabetes"]

model = joblib.load(BASE_DIR/"models"/"diabete_model_svc.pkl")
std = joblib.load(BASE_DIR/"models"/"scaler.pkl")
X_std = std.transform(X)

predictions = model.predict(X_std)

metrics = {
    "accuracy":accuracy_score(y, predictions),
    "recall":recall_score(y, predictions),
    "precision":precision_score(y, predictions),
    "f1_score":f1_score(y, predictions)
}

with open(BASE_DIR/"artifacts"/"metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


if metrics["recall"] < 0.75:
    raise Exception(
        f"Model n'est pas accepté, recall={metrics['recall']}"
    )

print(metrics)
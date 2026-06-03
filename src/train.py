import mlflow
import joblib
import matplotlib
import pandas as pd
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

matplotlib.use("Agg")

data = pd.read_csv("../data/data_clean.csv")

X = data[["pregnancies","glucose","diastolic","triceps","insulin","bmi","dpf","age"]]
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight":'balanced',
    "random_state": 42
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("classification_des_diabetes")
mlflow.sklearn.autolog()


with mlflow.start_run():
    mlflow.log_params(params)

    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_std, y_train)

    lr = LogisticRegression(**params)
    lr.fit(X_train_sm, y_train_sm)

    svc_model = SVC(kernel="rbf")
    svc_model.fit(X_train_sm, y_train_sm)

    rdf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rdf.fit(X_train, y_train)

    model_info = mlflow.sklearn.log_model(sk_model=lr, name="diabete_model")
    mlflow.sklearn.log_model(sk_model=std, name="scaler_model")
    mlflow.sklearn.log_model(sk_model=svc_model, name="svc_model")

    y_pred = lr.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)

    y_pred_svc = svc_model.predict(X_test_std)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    recall_svc = recall_score(y_test, y_pred_svc)
    mlflow.log_metric("accuracy_svc", accuracy_svc)
    mlflow.log_metric("recall_svc", recall_svc)

    y_pred_rdf = rdf.predict(X_test)
    accuracy_rdf = accuracy_score(y_test, y_pred_rdf)
    recall_rdf = recall_score(y_test, y_pred_rdf)
    mlflow.log_metric("accuracy_rdf", accuracy_rdf)
    mlflow.log_metric("recall_rdf", recall_rdf)

    if recall_svc >= 0.74:
        joblib.dump(svc_model, "../models/diabete_model_svc.pkl")
        joblib.dump(std, "../models/scaler.pkl")
        print("Modèle svc sauvegardé")
    elif recall >= 0.65:
        joblib.dump(lr, "../models/diabete_model_lr.pkl")
        joblib.dump(std, "../models/scaler.pkl")
        print("Modèle lr sauvegardé")
    elif recall_rdf >= 0.65:
        joblib.dump(lr, "../models/diabete_model_rdf.pkl")
        joblib.dump(std, "../data/scaler.pkl")
        print("Modèle rdf sauvegardé")
    else:
        print("Recall insuffisant")

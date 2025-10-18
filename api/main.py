from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import joblib

app = FastAPI()

# --- Cargar modelo desde MLflow ---
model = joblib.load('/app/models/fraud_model.pkl')

# --- Cargar scaler e Isolation Forest ---
scaler = joblib.load("/app/models/scaler.pkl")
iso_forest = joblib.load("/app/models/iso.pkl")


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    # Agregar feature de anomalía
    df["is_anomaly"] = iso_forest.predict(df)  # o predict_proba si quieres
    
    # Escalar features
    X_scaled = scaler.transform(df)
    
    # Predecir probabilidad de fraude
    pred = model.predict_proba(X_scaled)[:, 1]
    return {"fraud_probability": float(pred[0])}
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()

# --- Cargar modelo desde MLflow ---
model = joblib.load('/app/models/fraud_model.pkl')
scaler = joblib.load("/app/models/scaler.pkl")
iso_forest = joblib.load("/app/models/iso.pkl")

# --- Definir el schema de entrada ---
class TransactionData(BaseModel):
    """
    Datos de transacción con tarjeta de crédito.
    V1-V28 son componentes principales obtenidos con PCA.
    """
    Time: float = Field(description="Segundos transcurridos desde la primera transacción")
    V1: float = Field(description="Componente principal 1")
    V2: float = Field(description="Componente principal 2")
    V3: float = Field(description="Componente principal 3")
    V4: float = Field(description="Componente principal 4")
    V5: float = Field(description="Componente principal 5")
    V6: float = Field(description="Componente principal 6")
    V7: float = Field(description="Componente principal 7")
    V8: float = Field(description="Componente principal 8")
    V9: float = Field(description="Componente principal 9")
    V10: float = Field(description="Componente principal 10")
    V11: float = Field(description="Componente principal 11")
    V12: float = Field(description="Componente principal 12")
    V13: float = Field(description="Componente principal 13")
    V14: float = Field(description="Componente principal 14")
    V15: float = Field(description="Componente principal 15")
    V16: float = Field(description="Componente principal 16")
    V17: float = Field(description="Componente principal 17")
    V18: float = Field(description="Componente principal 18")
    V19: float = Field(description="Componente principal 19")
    V20: float = Field(description="Componente principal 20")
    V21: float = Field(description="Componente principal 21")
    V22: float = Field(description="Componente principal 22")
    V23: float = Field(description="Componente principal 23")
    V24: float = Field(description="Componente principal 24")
    V25: float = Field(description="Componente principal 25")
    V26: float = Field(description="Componente principal 26")
    V27: float = Field(description="Componente principal 27")
    V28: float = Field(description="Componente principal 28")
    Amount: float = Field(description="Monto de la transacción")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "Time": 0,
                    "V1": -1.359807,
                    "V2": -0.072781,
                    "V3": 2.536347,
                    "V4": 1.378155,
                    "V5": -0.338321,
                    "V6": 0.462388,
                    "V7": 0.239599,
                    "V8": 0.098698,
                    "V9": 0.363787,
                    "V10": 0.090794,
                    "V11": -0.551600,
                    "V12": -0.617801,
                    "V13": -0.991390,
                    "V14": -0.311169,
                    "V15": 1.468177,
                    "V16": -0.470401,
                    "V17": 0.207971,
                    "V18": 0.025791,
                    "V19": 0.403993,
                    "V20": 0.251412,
                    "V21": -0.018307,
                    "V22": 0.277838,
                    "V23": -0.110474,
                    "V24": 0.066928,
                    "V25": 0.128539,
                    "V26": -0.189115,
                    "V27": 0.133558,
                    "V28": -0.021053,
                    "Amount": 149.62
                }
            ]
        }

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

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
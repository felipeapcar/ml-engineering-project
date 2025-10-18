import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_and_preprocess(data):
    # Cargar dataset
    df = data
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Guardar scaler
    joblib.dump(scaler, BASE_DIR / "models" / 'scaler.pkl')
    
    return X_scaled, y, scaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE

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

def apply_smote(X_train, y_train, sampling_strategy='auto', random_state=42):    
    # Aplicar SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
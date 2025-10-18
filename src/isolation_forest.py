from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def train_isolation_forest(train_data, contamination=0.01, random_state=42):
    df = pd.read_csv(train_data)
    y = df['Class']

    df.drop(columns=['Class'], inplace=True)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df['is_anomaly'] = (iso_forest.fit_predict(df) == -1).astype(int)
    df['Class'] = y

    joblib.dump(iso_forest, BASE_DIR / "models" / "iso.pkl")

    return df

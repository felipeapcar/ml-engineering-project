import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

def save_model(model, path='fraud_model.pkl'):
    joblib.dump(model, BASE_DIR / "models" / path)
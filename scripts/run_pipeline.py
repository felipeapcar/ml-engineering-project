from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.export_model import save_model
from src.isolation_forest import train_isolation_forest
from pathlib import Path
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel desde scripts/
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"

# --- Generar feature 'is_anomaly' ---
fraud_data =  train_isolation_forest(TRAIN_DATA_PATH)

# --- Cargar y preprocesar ---
X_scaled, y, scaler = load_and_preprocess(fraud_data)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# --- Hiperparámetros encontrados ---
best_params = {
    'colsample_bytree': 0.8731,
    'learning_rate': 0.1161,
    'max_depth': 10,
    'min_child_weight': 1,
    'n_estimators': 370,
    'scale_pos_weight': 578.5465,
    'subsample': 0.6,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

with mlflow.start_run():
    # --- Entrenar ---
    model = train_model(X_train, y_train, best_params)

    # --- Evaluar ---
    metrics = evaluate_model(model, X_test, y_test)
    print("=== Final Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        mlflow.log_metric(k, v)

    # --- Exportar ---
    save_model(model, 'fraud_model.pkl')
    print("\n✅ Pipeline completo: Modelo y scaler exportados.")

     # --- Log scaler ---
    mlflow.log_artifact(BASE_DIR / "models" / "scaler.pkl")
    mlflow.log_artifact(BASE_DIR / "models" / "iso.pkl")
    mlflow.sklearn.log_model(model, BASE_DIR / "models" / "fraud_model.pkl")

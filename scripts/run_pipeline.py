from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_preprocess, apply_smote
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.export_model import save_model
from src.isolation_forest import train_isolation_forest
from pathlib import Path
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"

# Config
USE_SMOTE = True
SMOTE_STRATEGY = 0.5

# Pipeline
fraud_data = train_isolation_forest(TRAIN_DATA_PATH)
X_scaled, y, scaler = load_and_preprocess(fraud_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Aplicar SMOTE
if USE_SMOTE:
    X_train, y_train = apply_smote(X_train, y_train, sampling_strategy=SMOTE_STRATEGY)
    scale_pos_weight = 1
else:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

best_params = {
    'colsample_bytree': 0.8731,
    'learning_rate': 0.1161,
    'max_depth': 10,
    'min_child_weight': 1,
    'n_estimators': 370,
    'scale_pos_weight': scale_pos_weight,
    'subsample': 0.6,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

mlflow.set_experiment("Fraud Detection")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run():
    mlflow.log_param("use_smote", USE_SMOTE)
    mlflow.log_params(best_params)
    
    model = train_model(X_train, y_train, best_params)
    metrics = evaluate_model(model, X_test, y_test)
    
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    
    save_model(model, 'fraud_model.pkl')
    
    mlflow.log_artifact(BASE_DIR / "models" / "scaler.pkl")
    mlflow.log_artifact(BASE_DIR / "models" / "iso.pkl")
    mlflow.sklearn.log_model(model, artifact_path="fraud_model")

print("✅ Pipeline completo")
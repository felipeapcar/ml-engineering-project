from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_preprocess, apply_smote
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.export_model import save_model
from src.isolation_forest import train_isolation_forest
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Fraud Detection"

USE_SMOTE = True
SMOTE_STRATEGY = 0.5

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Pipeline
fraud_data = train_isolation_forest(TRAIN_DATA_PATH)
X_scaled, y, scaler = load_and_preprocess(fraud_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

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

# Train
run_name = f"xgboost_smote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param("use_smote", USE_SMOTE)
    mlflow.log_param("smote_strategy", SMOTE_STRATEGY)
    mlflow.log_params(best_params)
    
    model = train_model(X_train, y_train, best_params)
    metrics = evaluate_model(model, X_test, y_test)
    
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    
    save_model(model, 'fraud_model.pkl')
    
    mlflow.log_artifact(str(BASE_DIR / "models" / "fraud_model.pkl"))
    mlflow.log_artifact(str(BASE_DIR / "models" / "scaler.pkl"))
    mlflow.log_artifact(str(BASE_DIR / "models" / "iso.pkl"))
    
    signature = mlflow.models.infer_signature(X_test, model.predict_proba(X_test))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name="fraud_detection_xgboost"
    )

print(f"✅ Done | ROC-AUC: {metrics.get('roc_auc_score', 0):.4f} | {MLFLOW_TRACKING_URI}")
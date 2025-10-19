# End-to-End Fraud Detection System

Sistema completo de detección de fraude en transacciones de tarjetas de crédito con pipeline automatizado de ML, experiment tracking, validaciones automáticas y API de predicciones en tiempo real.

---

## Tabla de Contenidos

- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Componentes Principales](#componentes-principales)
- [Flujo del Sistema](#flujo-del-sistema)
- [Inicio Rápido](#inicio-rápido)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Cómo Funciona](#cómo-funciona)
- [Validaciones y Calidad](#validaciones-y-calidad)
- [Configuración](#configuración)
- [Stack Tecnológico](#stack-tecnológico)
- [Troubleshooting](#troubleshooting)

---

## Arquitectura del Sistema

El sistema está compuesto por cuatro componentes principales que trabajan juntos para proporcionar un pipeline completo de MLOps:

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection System                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    ┌───▼────┐          ┌─────▼─────┐        ┌─────▼─────┐
    │ MLflow │          │  Airflow  │        │  FastAPI  │
    │ Server │          │ Scheduler │        │    API    │
    └───┬────┘          └─────┬─────┘        └─────┬─────┘
        │                     │                     │
    Tracks           Orchestrates            Serves
    Experiments      Training                Predictions
```

---

## Componentes Principales

### 1. ML Pipeline (Manual/Ad-hoc Training)
- **Propósito**: Entrenamiento manual y experimentación
- **Ubicación**: `scripts/run_pipeline.py` + `src/`
- **Funcionalidad**:
  - Carga y preprocesamiento de datos (Time, V1-V28, Amount, Class)
  - Feature engineering con IsolationForest
  - Balanceo de clases con SMOTE (50% sampling strategy)
  - Entrenamiento de XGBoost con hiperparámetros optimizados
  - Evaluación con métricas: Precision, Recall, F1-Score, ROC-AUC
  - Logging completo a MLflow
  - Exportación de modelos (fraud_model.pkl, scaler.pkl, iso.pkl)

### 2. MLflow Server (Experiment Tracking)
- **Propósito**: Tracking de experimentos y gestión de modelos
- **Puerto**: 5000
- **Funcionalidad**:
  - Almacena métricas, parámetros y artifacts
  - Backend: PostgreSQL para metadata
  - Storage: Volúmenes Docker para artifacts
  - UI para visualización y comparación de runs

### 3. Airflow (Automated Orchestration)
- **Propósito**: Orquestación y reentrenamiento automático
- **Puerto**: 8080
- **Funcionalidad**:
  - **DAG Principal**: `weekly_model_retraining`
    - Schedule: Domingos 2:00 AM
    - Ejecuta pipeline completo en Docker container
    - Validaciones automáticas de datos y modelos
    - Deployment condicional basado en métricas
  - **DAG de Prueba**: `test_dag`
    - Verifica configuración de Airflow
    - Chequea acceso a volúmenes

### 4. FastAPI (Prediction Service)
- **Propósito**: API REST para predicciones en tiempo real
- **Puerto**: 8000
- **Funcionalidad**:
  - Endpoint `/predict`: Probabilidad de fraude
  - Validación de schema con Pydantic
  - Carga modelos desde volumen compartido
  - Documentación automática en `/docs`

---

## Flujo del Sistema

### Flujo de Entrenamiento Semanal Automatizado

```
┌──────────────────────────────────────────────────────────────┐
│ DOMINGO 2:00 AM - Airflow Scheduler Trigger                 │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Task 1: start                  │
        │  (EmptyOperator)                │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │  Task 2: run_training_pipeline  │
        │  (DockerOperator)               │
        └────────────┬────────────────────┘
                     │
        Creates Docker Container ───────────────┐
                     │                          │
                     ▼                          │
        ┌────────────────────────────┐          │
        │  fraud-detection:latest    │          │
        │  Container Running         │          │
        │                            │          │
        │  1. Load Data              │◄─────────┤ Mounts:
        │     from /app/data         │          │ - data/
        │                            │          │ - models/
        │  2. Preprocess             │          │ - mlruns/
        │     - IsolationForest      │          │
        │     - StandardScaler       │          │ Network:
        │     - SMOTE                │          │ fase1_fraud_
        │                            │          │ detection_network
        │  3. Train XGBoost          │          │
        │                            │          │ Env Vars:
        │  4. Evaluate Model         │          │ MLFLOW_TRACKING_URI=
        │     - ROC-AUC              │◄─────────┤ http://mlflow-server:5000
        │     - Precision            │          │
        │     - Recall               │          │
        │     - F1-Score             │          │
        │                            │          │
        │  5. Log to MLflow ─────────┼──────────┤─► MLflow Server
        │                            │          │   (Port 5000)
        │  6. Save Models            │          │
        │     to /app/models/        │◄─────────┘
        └────────────┬───────────────┘
                     │
                     │ Container completes
                     │ Models saved to host
                     ▼
        ┌─────────────────────────────────┐
        │  Task 3: end                    │
        │  (EmptyOperator)                │
        └─────────────────────────────────┘
                     │
                     ▼
        Models now in: models/fraud_model.pkl
                      models/scaler.pkl
                      models/iso.pkl
```

### Flujo de Predicción (API)

```
Client Request
    │
    ▼
POST /predict
    │
    ▼
┌────────────────────────────┐
│ Pydantic Schema Validation │
│ - Time: float              │
│ - V1-V28: float            │
│ - Amount: float            │
└───────────┬────────────────┘
            │ ✓ Valid
            ▼
┌────────────────────────────┐
│ Load Models                │
│ - fraud_model.pkl          │
│ - scaler.pkl               │
│ - iso.pkl                  │
└───────────┬────────────────┘
            │
            ▼
┌────────────────────────────┐
│ Feature Engineering        │
│ 1. IsolationForest predict │
│ 2. StandardScaler          │
└───────────┬────────────────┘
            │
            ▼
┌────────────────────────────┐
│ XGBoost predict_proba      │
└───────────┬────────────────┘
            │
            ▼
    {"fraud_probability": 0.023}
```

---

## Inicio Rápido

### Prerrequisitos

- Docker & Docker Desktop instalado y corriendo
- 8 GB RAM mínimo
- Puertos 5000, 5432, 5433, 8000, 8080 disponibles

### 1. Levantar Todo el Sistema

```bash
# Clonar repositorio y navegar a la carpeta
cd "ML Engineer Roadmap/Fase1"

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Construir imagen de entrenamiento
docker build -t fraud-detection:latest .

# Levantar todos los servicios
docker-compose up -d

# Verificar que todo esté corriendo
docker-compose ps
```

### 2. Acceder a los Servicios

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **Airflow UI** | http://localhost:8080 | User: `airflow` (configurado en .env)<br>Pass: `airflow` (configurado en .env) |
| **MLflow UI** | http://localhost:5000 | Sin autenticación |
| **API Docs** | http://localhost:8000/docs | Sin autenticación |
| **API Health** | http://localhost:8000/ | Sin autenticación |

### 3. Ejecutar el Pipeline

**Opción A: Trigger manual desde Airflow UI**
1. Ve a http://localhost:8080
2. Login con credenciales
3. Busca el DAG `weekly_model_retraining`
4. Click en el toggle para activarlo
5. Click en "Play" (▶) para ejecutar manualmente

**Opción B: Desde línea de comandos**
```bash
docker exec airflow_scheduler airflow dags trigger weekly_model_retraining
```

**Opción C: Entrenamiento manual (sin Airflow)**
```bash
python scripts/run_pipeline.py
```

### 4. Hacer una Predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Respuesta esperada:
```json
{
  "fraud_probability": 0.0234
}
```

---

## Estructura del Proyecto

```
Fase1/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv              # Dataset original
│   └── processed/
│       ├── train_data.csv              # Split de entrenamiento
│       └── test_data.csv               # Split de prueba
│
├── src/                                # Código fuente ML
│   ├── data_preprocessing.py           # Carga y preprocesamiento
│   ├── train_model.py                  # Entrenamiento XGBoost
│   ├── evaluate_model.py               # Cálculo de métricas
│   ├── export_model.py                 # Serialización de modelos
│   └── isolation_forest.py             # Detección de anomalías
│
├── scripts/
│   └── run_pipeline.py                 # Pipeline completo
│
├── models/
│   ├── fraud_model.pkl                 # XGBoost entrenado
│   ├── scaler.pkl                      # StandardScaler fitted
│   ├── iso.pkl                         # IsolationForest fitted
│   ├── staging/                        # Modelos en validación (futuro)
│   ├── production/                     # Modelos en prod (futuro)
│   └── backups/                        # Backups automáticos (futuro)
│
├── mlruns/                             # Experimentos MLflow (local)
│
├── mlflow_server/
│   ├── docker-compose.yml              # MLflow + PostgreSQL
│   └── Dockerfile.mlflow               # Imagen custom
│
├── airflow/
│   ├── dags/
│   │   ├── test_dag.py                 # DAG de prueba
│   │   └── weekly_retrain_dag.py       # DAG de reentrenamiento
│   ├── validators/
│   │   ├── data_validator.py           # Validaciones de datos
│   │   └── model_validator.py          # Validaciones de modelo
│   ├── notifiers/
│   │   ├── slack_notifier.py           # Notificaciones Slack
│   │   └── email_notifier.py           # Notificaciones Email
│   ├── config/
│   │   └── validation_thresholds.yaml  # Configuración de validaciones
│   ├── logs/                           # Logs de Airflow
│   ├── plugins/                        # Plugins custom
│   └── docker-compose.yml
│
├── api/
│   ├── main.py                         # FastAPI application
│   ├── Dockerfile                      # Container de API
│   └── requirements.txt                # Dependencias API
│
├── notebooks/
│   └── data_exploration.ipynb          # EDA
│
├── tests/                              # Tests (futuro)
├── docs/                               # Documentación (futuro)
│
├── docker-compose.yml                  # Orquestación completa
├── Dockerfile                          # Imagen de entrenamiento
├── .dockerignore                       # Exclusiones de build
├── requirements.txt                    # Dependencias Python
├── setup.py                            # Package setup
├── .env                                # Variables de entorno
└── README.md                           # Este archivo
```

---

## Cómo Funciona

### 1. ML Pipeline (scripts/run_pipeline.py)

**Paso a paso**:

```python
# 1. Configuración
BASE_DIR = Path(__file__).parent.parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# 2. Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Fraud Detection")

# 3. Feature Engineering: IsolationForest
fraud_data = train_isolation_forest(TRAIN_DATA_PATH)
# Detecta anomalías y agrega feature 'is_anomaly'

# 4. Preprocessing
X_scaled, y, scaler = load_and_preprocess(fraud_data)
# StandardScaler para normalización

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# 6. SMOTE para balanceo
if USE_SMOTE:
    X_train, y_train = apply_smote(X_train, y_train, strategy=0.5)

# 7. Entrenar XGBoost
with mlflow.start_run():
    model = train_model(X_train, y_train)

    # 8. Evaluar
    metrics = evaluate_model(model, X_test, y_test)

    # 9. Log a MLflow
    mlflow.log_params({...})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

    # 10. Guardar modelos
    save_model(model, scaler, iso_forest)
```

**Hiperparámetros XGBoost** (optimizados con Optuna):
```python
{
    'n_estimators': 370,
    'max_depth': 10,
    'learning_rate': 0.1161,
    'colsample_bytree': 0.8731,
    'subsample': 0.8,
    'gamma': 0,
    'min_child_weight': 1
}
```

### 2. Airflow Orchestration

**DAG: weekly_model_retraining**

```python
# Schedule
schedule='0 2 * * 0'  # Domingos 2:00 AM

# Tareas
start >> run_training_pipeline >> end

# DockerOperator ejecuta:
DockerOperator(
    image='fraud-detection:latest',
    docker_url='unix://var/run/docker.sock',
    network_mode='fase1_fraud_detection_network',
    mounts=[
        Mount(source='...Fase1/data', target='/app/data'),
        Mount(source='...Fase1/models', target='/app/models'),
    ],
    environment={
        'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000'
    }
)
```

**Por qué Docker-in-Docker**:
- Aislamiento: Cada entrenamiento en su propio container
- Reproducibilidad: Mismo environment siempre
- Escalabilidad: Fácil mover a Kubernetes
- Limpieza: Container se elimina automáticamente (auto_remove=True)

### 3. API de Predicciones

**Proceso de Inferencia**:

```python
# 1. Validación con Pydantic
class TransactionData(BaseModel):
    Time: float
    V1: float
    ...
    V28: float
    Amount: float

# 2. Carga de modelos (al startup)
model = joblib.load('/app/models/fraud_model.pkl')
scaler = joblib.load('/app/models/scaler.pkl')
iso_forest = joblib.load('/app/models/iso.pkl')

# 3. Predicción
@app.post("/predict")
def predict(data: TransactionData):
    df = pd.DataFrame([data.model_dump()])

    # Feature engineering
    df["is_anomaly"] = iso_forest.predict(df)

    # Scaling
    X_scaled = scaler.transform(df)

    # Predicción
    prob = model.predict_proba(X_scaled)[:, 1]

    return {"fraud_probability": float(prob[0])}
```

---

## Validaciones y Calidad

### Data Validation (airflow/validators/data_validator.py)

Validaciones ejecutadas ANTES del entrenamiento:

| Validación | Qué chequea | Threshold |
|-----------|-------------|-----------|
| `check_file_exists()` | Archivo CSV existe | - |
| `check_schema()` | 31 columnas presentes | Time, V1-V28, Amount, Class |
| `check_data_quality()` | Missing values | < 5% |
| `check_class_distribution()` | Proporción de fraude | 0.01% - 10% |
| `check_feature_ranges()` | Valores válidos | Amount >= 0, Time >= 0 |

### Model Validation (airflow/validators/model_validator.py)

Validaciones ejecutadas DESPUÉS del entrenamiento:

| Validación | Qué chequea | Threshold |
|-----------|-------------|-----------|
| `check_model_exists()` | fraud_model.pkl existe | - |
| `check_model_loadable()` | Modelo se puede cargar | Tiene predict() y predict_proba() |
| `check_performance_thresholds()` | Métricas sobre mínimos | ROC-AUC >= 0.85<br>Precision >= 0.70<br>Recall >= 0.60 |
| `check_metrics_sanity()` | Métricas razonables | Valores 0-1, no extremos |
| `compare_with_production()` | Mejor que producción | ROC-AUC mejora >= 1%<br>Recall drop <= 2% |

### Notification System

**Slack** (airflow/notifiers/slack_notifier.py):
```python
slack = SlackNotifier()
slack.send_training_success(metrics, run_id)
slack.send_deployment_success(metrics, improvement=0.02)
slack.send_validation_failure(results, stage="training")
```

**Email** (airflow/notifiers/email_notifier.py):
```python
email = EmailNotifier()
email.send_training_report(to_emails, metrics, run_id)
email.send_deployment_notification(to_emails, metrics, deployed=True)
```

---

## Configuración

### Variables de Entorno (.env)

```bash
# Airflow PostgreSQL
AIRFLOW_POSTGRES_USER=airflow
AIRFLOW_POSTGRES_PASSWORD=your_secure_password
AIRFLOW_POSTGRES_DB=airflow

# Airflow Web UI
AIRFLOW_WWW_USER=airflow
AIRFLOW_WWW_PASSWORD=your_secure_password

# Slack (opcional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email (opcional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@yourcompany.com
```

### Thresholds de Validación (airflow/config/validation_thresholds.yaml)

```yaml
model_performance:
  min_roc_auc: 0.85
  min_precision: 0.70
  min_recall: 0.60
  min_f1_score: 0.65

model_comparison:
  min_improvement: 0.01
  max_recall_drop: 0.02

data_quality:
  max_missing_pct: 5.0
  min_fraud_rate_pct: 0.01
  max_fraud_rate_pct: 10.0
```

---

## Stack Tecnológico

### Machine Learning
- **XGBoost** 2.1.2 - Gradient boosting classifier
- **scikit-learn** 1.7.2 - Preprocessing, metrics
- **imbalanced-learn** 0.14.0 - SMOTE
- **pandas** 2.2.3 - Data manipulation
- **joblib** 1.4.2 - Model serialization

### MLOps
- **MLflow** 2.16.2 - Experiment tracking, model registry
- **Apache Airflow** 2.10.4 - Workflow orchestration
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

### API & Web
- **FastAPI** 0.115.4 - REST API framework
- **Pydantic** 2.10.3 - Data validation
- **Uvicorn** 0.32.1 - ASGI server

### Infrastructure
- **PostgreSQL** 15 - Metadata storage (2 instances)
  - MLflow backend (port 5432)
  - Airflow backend (port 5433)
- **Docker Networks** - Container communication

---

## Comandos Útiles

### Docker

```bash
# Ver estado de servicios
docker-compose ps

# Logs en tiempo real
docker-compose logs -f [servicio]

# Reiniciar servicio específico
docker-compose restart airflow-scheduler

# Detener todo
docker-compose down

# Limpiar todo (incluyendo volúmenes)
docker-compose down -v

# Reconstruir imagen de entrenamiento
docker build -t fraud-detection:latest .

# Ver networks
docker network ls

# Ver volúmenes
docker volume ls
```

### Airflow

```bash
# Trigger DAG manualmente
docker exec airflow_scheduler airflow dags trigger weekly_model_retraining

# Listar DAGs
docker exec airflow_scheduler airflow dags list

# Ver estado de DAG
docker exec airflow_scheduler airflow dags state weekly_model_retraining

# Ver logs de tarea
docker exec airflow_scheduler airflow tasks logs weekly_model_retraining run_training_pipeline <execution_date>

# Pausar DAG
docker exec airflow_scheduler airflow dags pause weekly_model_retraining

# Despausar DAG
docker exec airflow_scheduler airflow dags unpause weekly_model_retraining
```

### MLflow

```bash
# Ver runs en CLI
docker exec mlflow_server mlflow runs list --experiment-name "Fraud Detection"

# Servir modelo desde MLflow
mlflow models serve -m runs:/<run_id>/model -p 5001
```

---

## Troubleshooting

### El scheduler de Airflow no aparece como corriendo

**Problema**: "The scheduler does not appear to be running"

**Solución**:
```bash
# Ver logs del scheduler
docker logs airflow_scheduler

# Reiniciar scheduler
docker-compose restart airflow-scheduler

# Si persiste, recrear servicios
docker-compose down
docker-compose up -d
```

### El DAG no aparece en Airflow UI

**Problema**: DAG no visible después de 30 segundos

**Soluciones**:
```bash
# 1. Verificar errores en DAG
docker exec airflow_scheduler airflow dags list-import-errors

# 2. Ver logs del scheduler
docker logs airflow_scheduler --tail 100

# 3. Verificar que el archivo esté montado
docker exec airflow_scheduler ls /opt/airflow/dags
```

### DockerOperator falla con "network not found"

**Problema**: `network fase1_fraud_detection_network not found`

**Solución**:
```bash
# Ver nombre real de la red
docker network ls | grep fraud

# Actualizar network_mode en el DAG con el nombre correcto
# Ejemplo: fase1_fraud_detection_network
```

### API no encuentra los modelos

**Problema**: `FileNotFoundError: fraud_model.pkl`

**Soluciones**:
```bash
# 1. Verificar que los modelos existen
ls models/

# 2. Verificar mount en docker-compose.yml
# Debe tener: - ./models:/app/models

# 3. Reconstruir y reiniciar API
docker-compose restart fraud-api
```

### MLflow no se conecta desde el container

**Problema**: `Connection refused: localhost:5000`

**Solución**:
- Verificar que `MLFLOW_TRACKING_URI` esté configurado correctamente
- Dentro del container debe ser: `http://mlflow-server:5000`
- NO usar `localhost:5000`

### Imagen fraud-detection:latest no existe

**Problema**: `pull access denied for fraud-detection`

**Solución**:
```bash
# Construir la imagen primero
docker build -t fraud-detection:latest .

# Verificar que existe
docker images | grep fraud-detection
```

### Puertos ocupados

**Problema**: `port is already allocated`

**Soluciones**:
```bash
# Opción 1: Detener servicio que usa el puerto
# Opción 2: Cambiar puerto en docker-compose.yml

# Ejemplo para Airflow:
# Cambiar: "8080:8080" → "8081:8080"
# Acceder: http://localhost:8081
```

---

## Próximas Mejoras

**Phase 2 - MLOps Avanzado**:
- [ ] Tests unitarios e integración (pytest)
- [ ] CI/CD con GitHub Actions
- [ ] Monitoreo con Prometheus + Grafana
- [ ] Data drift detection automático

**Phase 3 - Deployment en Cloud**:
- [ ] Deployment en GCP (Cloud Run + Vertex AI)
- [ ] Kubernetes para orquestación
- [ ] BigQuery para almacenamiento de logs

**Phase 4 - Real-Time**:
- [ ] Kafka para streaming
- [ ] Dashboard operacional
- [ ] Alertas avanzadas

---

## Licencia

MIT

---

## Autor

Desarrollado como proyecto de MLOps end-to-end

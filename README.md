# Fraud Detection System - ML Engineer Roadmap Phase 1

Sistema completo de detección de fraude en transacciones de tarjetas de crédito con pipeline de entrenamiento automatizado, tracking de experimentos y API de predicciones.

## Arquitectura del Sistema

### Componentes Principales

1. **ML Pipeline** - Pipeline de entrenamiento manual y experimentación
2. **MLflow Server** - Tracking de experimentos y gestión de modelos
3. **Airflow** - Orquestación y reentrenamiento automático semanal
4. **FastAPI** - API REST para predicciones en tiempo real

### Servicios y Puertos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| MLflow UI | 5000 | Interfaz de tracking de experimentos |
| Airflow UI | 8080 | Interfaz de orquestación |
| FastAPI | 8000 | API de predicciones |
| MLflow PostgreSQL | 5432 | Base de datos de MLflow |
| Airflow PostgreSQL | 5433 | Base de datos de Airflow |

## Inicio Rápido

### Opción 1: Levantar Todo el Sistema

```bash
# Desde la raíz del proyecto
docker-compose up -d
```

Esto levanta todos los servicios:
- MLflow Server + PostgreSQL
- Airflow Webserver + Scheduler + PostgreSQL
- Fraud Detection API

### Opción 2: Levantar Servicios Individuales

```bash
# Solo MLflow
cd mlflow_server
docker-compose up -d

# Solo Airflow
cd airflow
docker-compose up -d

# Solo API (requiere build primero)
docker build -t fraud-api -f api/Dockerfile .
docker run -p 8000:8080 -v ./models:/app/models fraud-api
```

### Acceso a los Servicios

- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080
  - Usuario: `airflow`
  - Contraseña: `airflow`
- **API**: http://localhost:8000
  - Docs: http://localhost:8000/docs
  - Health: http://localhost:8000/

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                        # Dataset original
│   └── processed/                  # Datos procesados (train/test)
│
├── src/                            # Código fuente del pipeline ML
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── export_model.py
│   └── isolation_forest.py
│
├── scripts/
│   └── run_pipeline.py             # Pipeline de entrenamiento manual
│
├── models/
│   ├── staging/                    # Modelos nuevos en validación
│   ├── production/                 # Modelos en producción
│   └── backups/                    # Backups de modelos anteriores
│
├── notebooks/
│   └── data_exploration.ipynb      # Análisis exploratorio
│
├── mlflow_server/                  # MLflow + PostgreSQL
│   ├── docker-compose.yml
│   └── Dockerfile.mlflow
│
├── airflow/                        # Orquestación con Airflow
│   ├── dags/                       # Definiciones de DAGs
│   ├── tasks/                      # Tareas reutilizables
│   ├── plugins/                    # Operadores y sensores custom
│   ├── config/                     # Configuraciones
│   ├── logs/                       # Logs de Airflow
│   └── docker-compose.yml
│
├── api/                            # API de predicciones
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/                          # Tests unitarios e integración
├── docs/                           # Documentación
│
├── docker-compose.yml              # Orquestación completa
├── requirements.txt
└── README.md
```

## Uso del Sistema

### 1. Entrenamiento Manual

```bash
python scripts/run_pipeline.py
```

Esto ejecuta:
1. Carga y preprocesamiento de datos
2. Detección de anomalías con IsolationForest
3. Balanceo con SMOTE
4. Entrenamiento de XGBoost
5. Evaluación y logging a MLflow
6. Exportación de modelos

### 2. Predicciones con la API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.359807,
    "V2": -0.072781,
    ... (V3-V28),
    "Amount": 149.62
  }'
```

Respuesta:
```json
{
  "fraud_probability": 0.0234
}
```

### 3. Reentrenamiento Automático

El sistema incluye un DAG de Airflow para reentrenamiento semanal automático:

1. Validación de datos nuevos
2. Entrenamiento del modelo
3. Evaluación y comparación con producción
4. Deployment condicional (solo si mejora métricas)
5. Notificaciones por Slack/Email

El DAG se ejecuta automáticamente cada domingo a las 2am.

## Comandos Útiles

### Docker

```bash
# Ver servicios corriendo
docker-compose ps

# Ver logs
docker-compose logs -f [servicio]

# Detener todo
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Reiniciar un servicio específico
docker-compose restart [servicio]
```

### MLflow

```bash
# Ver experimentos
mlflow ui --host 0.0.0.0 --port 5000

# Registrar modelo
mlflow models serve -m runs:/<run_id>/model
```

### Airflow

```bash
# Trigger manual de DAG
docker exec airflow_webserver airflow dags trigger weekly_retrain

# Listar DAGs
docker exec airflow_webserver airflow dags list

# Ver logs de task
docker exec airflow_webserver airflow tasks logs <dag_id> <task_id> <execution_date>
```

## Stack Tecnológico

### Machine Learning
- **XGBoost** 2.1.2 - Clasificador principal
- **scikit-learn** 1.7.2 - Preprocessing y evaluación
- **imbalanced-learn** 0.14.0 - SMOTE para balanceo
- **pandas** 2.2.3 - Manipulación de datos

### MLOps
- **MLflow** 2.16.2 - Experiment tracking
- **Apache Airflow** 2.10.4 - Orquestación
- **Docker & Docker Compose** - Containerización

### API & Deployment
- **FastAPI** 0.115.4 - Framework web
- **Pydantic** 2.10.3 - Validación de datos
- **Uvicorn** 0.32.1 - ASGI server

### Databases
- **PostgreSQL** 15 - Metadata storage (MLflow & Airflow)

## Próximos Pasos

- [ ] Implementar DAGs de reentrenamiento en Airflow
- [ ] Configurar notificaciones Slack/Email
- [ ] Agregar tests unitarios e integración
- [ ] Implementar monitoreo con Prometheus/Grafana
- [ ] Agregar detección de data drift
- [ ] CI/CD pipeline con GitHub Actions
- [ ] Optimización de hiperparámetros con Optuna

## Troubleshooting

### Conflicto de puertos
Si los puertos están ocupados, modifica en `docker-compose.yml`:
- MLflow: cambiar `5000:5000` a `5001:5000`
- Airflow: cambiar `8080:8080` a `8081:8080`
- API: cambiar `8000:8080` a `8001:8080`

### Permisos en Linux/Mac
```bash
echo -e "AIRFLOW_UID=$(id -u)" > airflow/.env
```

### Logs de errores
```bash
docker-compose logs -f [servicio]
```

## Licencia

MIT

## Autor

Desarrollado como parte del ML Engineer Roadmap - Phase 1

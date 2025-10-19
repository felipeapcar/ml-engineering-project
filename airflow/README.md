# Airflow Setup for Automated Model Retraining

## Overview
Apache Airflow orchestration platform for weekly automated model retraining, validation, and deployment.

## Architecture
- **Airflow Webserver**: UI on http://localhost:8080
- **Airflow Scheduler**: Executes scheduled DAGs
- **PostgreSQL**: Metadata database (port 5433)
- **LocalExecutor**: Runs tasks in separate processes

## Quick Start

### 1. Start Airflow
```bash
cd airflow
docker-compose up -d
```

### 2. Check Services Status
```bash
docker-compose ps
```

You should see:
- `airflow-webserver` (port 8080)
- `airflow-scheduler`
- `postgres` (port 5433)

### 3. Access Airflow UI
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 4. View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
```

### 5. Stop Airflow
```bash
docker-compose down
```

### 6. Clean Up (Remove volumes)
```bash
docker-compose down -v
```

## Directory Structure
```
airflow/
├── dags/              # DAG definitions (Python files)
├── logs/              # Airflow logs
├── plugins/           # Custom plugins/operators
├── config/            # Configuration files
├── docker-compose.yml # Infrastructure definition
├── .env               # Environment variables
└── README.md          # This file
```

## Next Steps
1. Create DAGs in `dags/` folder
2. Configure connections (MLflow, Slack, SMTP)
3. Set up variables for thresholds and paths
4. Implement retraining pipeline

## Troubleshooting

### Port Conflicts
- Airflow webserver uses port 8080
- Airflow postgres uses port 5433 (to avoid conflict with MLflow's postgres on 5432)

### Permission Issues (Linux/Mac)
If you get permission errors:
```bash
echo -e "AIRFLOW_UID=$(id -u)" >> .env
```

### View Container Status
```bash
docker-compose ps
docker-compose logs airflow-webserver
```

## Important Notes
- First startup takes 2-3 minutes (database initialization)
- Default credentials: airflow/airflow (change in `.env` file)
- DAGs are auto-detected from `dags/` folder
- Changes to DAGs are reflected automatically (may take ~30 seconds)

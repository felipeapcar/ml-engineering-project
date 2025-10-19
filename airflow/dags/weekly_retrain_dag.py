"""
Weekly Model Retraining DAG

This DAG automatically retrains the fraud detection model every week
by running the training pipeline in a Docker container.

Schedule: Every Sunday at 2:00 AM
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from docker.types import Mount


# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['team@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


# Define the DAG
with DAG(
    dag_id='weekly_model_retraining',
    default_args=default_args,
    description='Weekly automated model retraining using Docker container',
    schedule='0 2 * * 0',  # Every Sunday at 2:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['production', 'ml', 'retraining'],
) as dag:

    # Task 1: Start
    start = EmptyOperator(
        task_id='start',
    )

    # Task 2: Run the training pipeline in Docker container
    run_pipeline = DockerOperator(
        task_id='run_training_pipeline',
        image='fraud-detection:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='fase1_fraud_detection_network',
        mount_tmp_dir=False,
        force_pull=False,
        mounts=[
            # Mount data directory
            Mount(
                source='/c/Users/pipea/Desktop/Python projects/ML Engineer Roadmap/Fase1/data',
                target='/app/data',
                type='bind'
            ),
            # Mount models directory
            Mount(
                source='/c/Users/pipea/Desktop/Python projects/ML Engineer Roadmap/Fase1/models',
                target='/app/models',
                type='bind'
            ),
        ],
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
        },
        doc_md="""
        ### Run Training Pipeline in Docker

        Executes the complete ML pipeline inside a Docker container:
        1. Builds/uses the fraud-detection:latest image
        2. Mounts data and models directories
        3. Connects to MLflow server
        4. Runs the training script
        5. Saves models to mounted volume

        The container runs on the same network as MLflow for tracking.
        """,
    )

    # Task 3: End
    end = EmptyOperator(
        task_id='end',
    )

    # Define task dependencies
    start >> run_pipeline >> end

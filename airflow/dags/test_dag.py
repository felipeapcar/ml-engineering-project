"""
Simple test DAG to verify Airflow setup.
This DAG runs basic tasks to ensure the environment is working correctly.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


def print_hello():
    """Simple Python function to test PythonOperator"""
    print("Hello from Airflow!")
    print(f"Current time: {datetime.now()}")
    return "Success"


def check_environment():
    """Check Python environment and available packages"""
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")

    # Check for key packages
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("Pandas not available")

    return "Environment check complete"


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


# Define the DAG
with DAG(
    dag_id='test_dag',
    default_args=default_args,
    description='Simple test DAG to verify Airflow setup',
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['test', 'setup'],
) as dag:

    # Task 1: Print hello
    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )

    # Task 2: Check environment
    task_check_env = PythonOperator(
        task_id='check_environment',
        python_callable=check_environment,
    )

    # Task 3: Run bash command
    task_bash = BashOperator(
        task_id='run_bash_command',
        bash_command='echo "Bash operator working!" && date',
    )

    # Task 4: Check data directory
    task_check_data = BashOperator(
        task_id='check_data_directory',
        bash_command='ls -la /opt/airflow/data/ || echo "Data directory not found"',
    )

    # Task 5: Check models directory
    task_check_models = BashOperator(
        task_id='check_models_directory',
        bash_command='ls -la /opt/airflow/models/ || echo "Models directory not found"',
    )

    # Define task dependencies
    task_hello >> task_check_env >> task_bash >> [task_check_data, task_check_models]

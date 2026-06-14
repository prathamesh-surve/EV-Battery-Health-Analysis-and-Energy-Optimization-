from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ev_battery_health_pipeline',
    default_args=default_args,
    description='Automated orchestration of EV Battery cleaning and machine learning models',
    schedule_interval=None, # Trigger manually via UI
    catchup=False,
) as dag:

    task_data_processing = BashOperator(
        task_id='clean_and_engineer_features',
        bash_command='python3 /opt/airflow/scripts/data_processing.py',
    )

    task_model_training = BashOperator(
        task_id='train_and_evaluate_models',
        bash_command='python3 /opt/airflow/scripts/train_models.py',
    )

    # NEW: Automated dashboard execution task block
    task_generate_dashboard = BashOperator(
        task_id='generate_analytical_dashboard',
        bash_command='python3 /opt/airflow/scripts/generate_dashboard.py',
    )

    # Establish updated three-node serial processing pipeline dependency flow
    task_data_processing >> task_model_training >> task_generate_dashboard
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


# 各タスクで実行する処理
def extract() -> None:
    print("Step 1: Extract data")


def transform() -> None:
    print("Step 2: Transform data")


def load() -> None:
    print("Step 3: Load data")


# DAG定義
with DAG(
    dag_id="simple_etl_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    load_task = PythonOperator(task_id="load", python_callable=load)

    extract_task >> transform_task >> load_task

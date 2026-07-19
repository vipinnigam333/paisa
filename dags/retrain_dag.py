"""
retrain_dag.py
---------------
Airflow DAG that automates the retraining pipeline weekly.

To use this:
1. `pip install apache-airflow`
2. Copy this file into your Airflow `dags/` folder.
3. Make sure the `stock-mlops` project (src/, models/, data/) is on the
   Airflow worker's PYTHONPATH (e.g. mount it as a volume if using
   Docker-based Airflow).
4. Turn the DAG on in the Airflow UI.

This is intentionally simple (PythonOperator calling our existing
retrain_pipeline.run function) rather than reimplementing logic in
Airflow-specific code -- the DAG is just the scheduler/orchestrator,
the actual ML logic stays in src/.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _retrain(**context):
    import sys
    sys.path.insert(0, "/opt/stock-mlops")  # adjust to your mounted project path
    from src.retrain_pipeline import run
    promoted = run(ticker="AAPL", lookback_years=8)
    context["ti"].xcom_push(key="promoted", value=promoted)


def _notify(**context):
    promoted = context["ti"].xcom_pull(key="promoted", task_ids="retrain_model")
    msg = "New model PROMOTED to production." if promoted else "Retraining ran; existing model kept (no improvement)."
    print(f"[notify] {msg}")
    # Extend this to post to Slack/email using Airflow's notification operators.


default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="stock_model_weekly_retrain",
    description="Weekly retrain of the stock direction prediction model with champion/challenger promotion.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * 1",  # every Monday at 06:00
    catchup=False,
    tags=["mlops", "stocks"],
) as dag:

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=_retrain,
    )

    notify_task = PythonOperator(
        task_id="notify",
        python_callable=_notify,
    )

    retrain_task >> notify_task

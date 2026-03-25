import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from docker.types import Mount

# Environment configuration
HOST_PATH = os.getenv("MLOPS_HOST_PATH", "/home/yura/PolProjects/mlops_labs")
DOCKER_IMAGE = os.getenv("MLOPS_DOCKER_IMAGE", "ml-pipeline:latest")
DOCKER_URL = "unix://var/run/docker.sock"
METRICS_FILE_PATH = "/opt/airflow/project/metrics.json"
RAW_DATA_PATH = "/opt/airflow/project/data/raw/House_Rent_10M_balanced_40cities.csv"

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def evaluate_model_metrics(**kwargs) -> str:
    """
    Reads the metrics JSON file and determines the next pipeline step
    based on the evaluation threshold.
    """
    if not os.path.exists(METRICS_FILE_PATH):
        raise FileNotFoundError(f"Metrics file not found at {METRICS_FILE_PATH}")

    with open(METRICS_FILE_PATH, "r") as f:
        metrics = json.load(f)

    print(f"Full metrics.json content: {metrics}")

    # Fetch the exact r2 score
    r2_score = metrics.get("r2", 0.0)

    # Set realistic threshold for Quality Gate
    r2_threshold = 0.25

    print(f"Evaluated R2 Score: {r2_score}. Target Threshold: {r2_threshold}")

    if r2_score > r2_threshold:
        print("Model passed Quality Gate! Proceeding to registration.")
        return "register_model"

    print("Model failed Quality Gate.")
    return "skip_registration"


with DAG(
    dag_id="ml_training_and_registry_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline with DVC execution, metric evaluation, and MLflow registration",
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "dvc", "mlflow"],
) as dag:

    def validate_data_exists(**kwargs):
        """
        Immediate check to ensure the raw data file is present before starting the pipeline.
        """
        raw_dir = os.path.dirname(RAW_DATA_PATH)

        if not os.path.exists(RAW_DATA_PATH):
            if os.path.exists(raw_dir):
                found_files = os.listdir(raw_dir)
                error_msg = f"File not found! Expected: {RAW_DATA_PATH}. Directory contents: {found_files}"
            else:
                error_msg = f"Directory {raw_dir} does not exist inside Airflow container!"
            raise FileNotFoundError(error_msg)

        print(f"Data found at {RAW_DATA_PATH}. Proceeding.")

    check_data_availability = PythonOperator(
        task_id="check_data_availability",
        python_callable=validate_data_exists,
    )

    prepare_data = DockerOperator(
        task_id="prepare_data",
        image=DOCKER_IMAGE,
        command="dvc repro prepare",
        docker_url=DOCKER_URL,
        network_mode="bridge",
        auto_remove="force",
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_PATH, target="/app", type="bind")],
    )

    train_model = DockerOperator(
        task_id="train_model",
        image=DOCKER_IMAGE,
        # Execute training and explicitly grant read permissions to the generated metrics file
        command='sh -c "dvc repro train && chmod 777 metrics.json"',
        docker_url=DOCKER_URL,
        network_mode="bridge",
        auto_remove="force",
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_PATH, target="/app", type="bind")],
    )

    evaluate_and_branch = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=evaluate_model_metrics,
    )

    register_model = DockerOperator(
        task_id="register_model",
        image=DOCKER_IMAGE,
        command="python src/register.py",
        docker_url=DOCKER_URL,
        network_mode="bridge",
        auto_remove="force",
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_PATH, target="/app", type="bind")],
    )

    skip_registration = EmptyOperator(task_id="skip_registration")

    # Pipeline definition
    check_data_availability >> prepare_data >> train_model >> evaluate_and_branch
    evaluate_and_branch >> [register_model, skip_registration]

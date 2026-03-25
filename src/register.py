import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def register_latest_model(model_name: str, tracking_uri: str, target_stage: str = "Staging"):
    """
    Finds the latest MLflow run across all experiments, registers the model,
    and transitions it to the target stage.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Retrieve all experiment IDs to ensure no runs are missed
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

        logger.info(f"Searching for runs across experiment IDs: {experiment_ids}")

        # Fetch the most recent run from ANY experiment
        runs = mlflow.search_runs(experiment_ids=experiment_ids, order_by=["start_time DESC"], max_results=1)

        if runs.empty:
            raise ValueError("No MLflow runs found. Cannot register the model.")

        run_id = runs.iloc[0].run_id

        # Note: Ensure 'xgboost_model' matches the artifact_path used in train.py
        model_uri = f"runs:/{run_id}/xgboost_model"

        logger.info(f"Registering model '{model_name}' from Run ID: {run_id}")
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(f"Transitioning model version {model_details.version} to {target_stage}")
        client.transition_model_version_stage(
            name=model_name, version=model_details.version, stage=target_stage, archive_existing_versions=True
        )

        logger.info(f"Model version {model_details.version} successfully transitioned to {target_stage}.")

    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise


if __name__ == "__main__":
    MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "HouseRent_XGBoost")
    # Use absolute path inside the Docker container to ensure stable connection
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////app/mlflow.db")

    register_latest_model(model_name=MODEL_NAME, tracking_uri=TRACKING_URI)

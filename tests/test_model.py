import os
import json
import glob
import pandas as pd
import pytest


@pytest.fixture
def metrics():
    """Fixture to load metrics.json for quality gate tests."""
    metrics_path = "metrics.json"
    assert os.path.exists(metrics_path), f"Metrics file not found: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_data_schema_basic():
    """Pre-train: Validate data schema and quality."""
    data_path = os.getenv("DATA_PATH", "data/prepared/train.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)
    required_cols = {"Rent"}
    missing_cols = required_cols - set(df.columns)

    assert not missing_cols, f"Missing required columns: {sorted(missing_cols)}"
    assert df["Rent"].notna().all(), "Target column 'Rent' contains NaN values."
    assert df.shape[0] >= 50, f"Insufficient data points for training: {df.shape[0]} rows."


def test_artifacts_exist():
    """Post-train: Verify all required artifacts are generated."""
    assert os.path.exists("model.pkl"), "Artifact 'model.pkl' is missing."
    assert os.path.exists("metrics.json"), "Artifact 'metrics.json' is missing."

    plot_files = glob.glob("feature_importance_*.png")
    assert plot_files, "Feature importance plot artifact is missing."


def test_quality_gate_rmse(metrics):
    """Post-train: Evaluate RMSE against the defined threshold."""
    threshold = float(os.getenv("RMSE_THRESHOLD", "5000.0"))
    rmse = float(metrics.get("rmse", float("inf")))

    assert rmse <= threshold, f"RMSE Quality Gate failed: {rmse:.2f} > {threshold:.2f}"


def test_quality_gate_r2(metrics):
    """Post-train: Evaluate R2 score against the defined threshold."""
    threshold = float(os.getenv("R2_THRESHOLD", "0.25"))
    r2 = float(metrics.get("r2", -float("inf")))

    assert r2 >= threshold, f"R2 Quality Gate failed: {r2:.4f} < {threshold:.2f}"

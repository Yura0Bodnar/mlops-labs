import os
from airflow.models import DagBag


def test_dag_integrity():
    """
    Loads all DAGs from the 'dags/' directory and checks for import errors.
    """
    dag_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dags")

    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

    assert not dag_bag.import_errors, f"DAG import failures: {dag_bag.import_errors}"

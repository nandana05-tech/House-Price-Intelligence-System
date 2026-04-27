"""
MLflow Model Registry helper — register, transition, and load models.
Used during retraining to promote new model versions to Production.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(_TRACKING_URI)
client = MlflowClient()


def register_catboost_model(
    model_path: str | Path,
    registered_name: str,
    run_id: str,
    description: str = "",
) -> str:
    """
    Register a CatBoost model file to the MLflow Model Registry.
    Returns the new model version string.
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=registered_name)
    version = result.version

    client.update_model_version(
        name=registered_name,
        version=version,
        description=description,
    )
    print(f"[Registry] Registered '{registered_name}' version {version}")
    return version


def promote_to_production(registered_name: str, version: str) -> None:
    """Transition a model version to Production, archiving the previous one."""
    client.transition_model_version_stage(
        name=registered_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[Registry] '{registered_name}' v{version} → Production")


def get_production_version(registered_name: str) -> str | None:
    """Return the current Production version number, or None if not found."""
    try:
        versions = client.get_latest_versions(registered_name, stages=["Production"])
        return versions[0].version if versions else None
    except Exception:  # noqa: BLE001
        return None

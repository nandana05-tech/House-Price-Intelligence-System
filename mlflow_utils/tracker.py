"""
MLflow tracking helper — wraps run creation with sensible defaults.
Each prediction event is logged as a single MLflow run in the configured experiment.
"""
from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


class PredictionTracker:
    """Thin wrapper around mlflow to log a single prediction as an MLflow run."""

    def __init__(self, experiment_name: str) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log(
        self,
        params: dict,
        metrics: dict,
        tags: dict | None = None,
        run_name: str | None = None,
    ) -> str:
        """
        Start a new MLflow run, log params + metrics + tags, then end the run.
        Returns the run_id.
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Params: flatten & stringify nested values
            flat_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(flat_params)

            # Metrics must be numeric
            numeric_metrics = {
                k: float(v) for k, v in metrics.items()
                if v is not None and isinstance(v, (int, float))
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)

            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})

            return run.info.run_id

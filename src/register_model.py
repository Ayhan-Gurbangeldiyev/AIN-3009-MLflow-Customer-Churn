"""Register the best MLflow run as a versioned churn model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from train_models import EXPERIMENT_NAME, TRACKING_URI


def find_best_run(metric_name: str = "roc_auc") -> object:
    """Find the best completed baseline/tuning run by metric."""

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("No MLflow experiment found. Run train_models.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=100,
    )
    model_runs = [
        run
        for run in runs
        if run.data.tags.get("run_type") in {"baseline", "tuning"}
        and metric_name in run.data.metrics
    ]
    if not model_runs:
        raise RuntimeError("No eligible training runs found. Run train_models.py or tune_model.py first.")

    return model_runs[0]


def wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 60) -> None:
    """Wait for a newly created model version to be ready."""

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        if model_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Model version registration failed: {model_version.status_message}")
        time.sleep(1)
    raise TimeoutError(f"Model version {model_name} v{version} was not ready after {timeout_seconds}s.")


def transition_stage(client: MlflowClient, model_name: str, version: str, stage: str) -> None:
    """Use stages when available and aliases as a modern MLflow fallback."""

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False,
        )
    except Exception as exc:
        print(f"Stage transition to {stage} was skipped: {exc}")

    alias = stage.lower()
    try:
        client.set_registered_model_alias(model_name, alias, version)
    except Exception as exc:
        print(f"Alias assignment '{alias}' was skipped: {exc}")


def register_best_model(model_name: str, metric_name: str = "roc_auc") -> dict[str, object]:
    """Register the best run and move it through Staging and Production."""

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    best_run = find_best_run(metric_name=metric_name)
    model_uri = f"runs:/{best_run.info.run_id}/model"

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(client, model_name, model_version.version)
    transition_stage(client, model_name, model_version.version, "Staging")
    transition_stage(client, model_name, model_version.version, "Production")

    summary = {
        "model_name": model_name,
        "version": model_version.version,
        "source_run_id": best_run.info.run_id,
        "source_model_uri": model_uri,
        "metric_name": metric_name,
        "metric_value": best_run.data.metrics[metric_name],
        "run_type": best_run.data.tags.get("run_type"),
        "registered_at": int(time.time()),
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/registered_model.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register the best MLflow churn model.")
    parser.add_argument("--model-name", default="ChurnPredictionModel", help="Registered model name.")
    parser.add_argument("--metric", default="roc_auc", help="Metric used to select the best run.")
    args = parser.parse_args()
    register_best_model(args.model_name, metric_name=args.metric)

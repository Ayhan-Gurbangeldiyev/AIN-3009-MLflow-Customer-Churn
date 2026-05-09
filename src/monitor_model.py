"""Simulate production monitoring for the registered churn model."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from data_preprocessing import RANDOM_STATE, prepare_data
from train_models import EXPERIMENT_NAME, TRACKING_URI, configure_mlflow


def add_simulated_outcome_noise(y_true: pd.Series, noise_rate: float, seed: int) -> pd.Series:
    """Flip a percentage of labels to mimic delayed/noisy production outcomes."""

    if noise_rate <= 0:
        return y_true.copy()

    rng = np.random.default_rng(seed)
    noisy = y_true.copy().reset_index(drop=True)
    flip_count = int(round(len(noisy) * noise_rate))
    if flip_count == 0:
        return noisy

    flip_indices = rng.choice(noisy.index.to_numpy(), size=flip_count, replace=False)
    noisy.loc[flip_indices] = 1 - noisy.loc[flip_indices]
    return noisy


def evaluate_batch(model: object, x_batch: pd.DataFrame, y_batch: pd.Series) -> dict[str, float]:
    """Calculate monitoring metrics for a batch."""

    y_pred = model.predict(x_batch)
    y_proba = model.predict_proba(x_batch)[:, 1]
    return {
        "accuracy": accuracy_score(y_batch, y_pred),
        "precision": precision_score(y_batch, y_pred, zero_division=0),
        "recall": recall_score(y_batch, y_pred, zero_division=0),
        "f1_score": f1_score(y_batch, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_batch, y_proba),
        "actual_churn_rate": float(np.mean(y_batch)),
        "predicted_churn_rate": float(np.mean(y_pred)),
        "average_tenure": float(x_batch["tenure"].mean()),
        "average_monthly_charges": float(x_batch["MonthlyCharges"].mean()),
    }


def simulate_monitoring(model_uri: str, data_path: str, batches: int = 4) -> pd.DataFrame:
    """Log batch-level monitoring metrics to MLflow."""

    configure_mlflow()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    prepared = prepare_data(data_path)
    model = mlflow.sklearn.load_model(model_uri)
    x_splits = np.array_split(prepared.x_test.reset_index(drop=True), batches)
    y_splits = np.array_split(prepared.y_test.reset_index(drop=True), batches)
    noise_schedule = np.linspace(0.0, 0.15, num=batches)
    records: list[dict[str, float | int]] = []

    with mlflow.start_run(run_name="monitoring_simulated_batches") as run:
        mlflow.set_tags(
            {
                "project": "Customer Churn Prediction",
                "run_type": "monitoring",
                "model_uri": model_uri,
            }
        )
        mlflow.log_param("number_of_batches", batches)
        mlflow.log_param("monitoring_data", data_path)

        for batch_number, (x_batch, y_batch, noise_rate) in enumerate(
            zip(x_splits, y_splits, noise_schedule),
            start=1,
        ):
            noisy_y = add_simulated_outcome_noise(
                pd.Series(y_batch),
                noise_rate=float(noise_rate),
                seed=RANDOM_STATE + batch_number,
            )
            metrics = evaluate_batch(model, x_batch, noisy_y)
            record = {
                "batch_number": batch_number,
                "rows": len(x_batch),
                "simulated_outcome_noise_rate": float(noise_rate),
                **metrics,
            }
            records.append(record)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"monitoring_{metric_name}", metric_value, step=batch_number)
            mlflow.log_metric("simulated_outcome_noise_rate", float(noise_rate), step=batch_number)

        monitoring_df = pd.DataFrame(records)
        Path("reports").mkdir(exist_ok=True)
        monitoring_df.to_csv("reports/monitoring_metrics.csv", index=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "monitoring_metrics.csv"
            plot_path = tmp_path / "monitoring_f1_by_batch.png"
            monitoring_df.to_csv(csv_path, index=False)

            plt.figure(figsize=(8, 5))
            plt.plot(monitoring_df["batch_number"], monitoring_df["f1_score"], marker="o", label="F1-score")
            plt.plot(monitoring_df["batch_number"], monitoring_df["roc_auc"], marker="o", label="ROC-AUC")
            plt.xlabel("Batch")
            plt.ylabel("Metric value")
            plt.title("Simulated Production Monitoring")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=160)
            plt.close()

            mlflow.log_artifact(str(csv_path), artifact_path="monitoring")
            mlflow.log_artifact(str(plot_path), artifact_path="monitoring")

        print(f"Monitoring run id: {run.info.run_id}")

    print(monitoring_df.to_string(index=False))
    return monitoring_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate production monitoring and log batch metrics.")
    parser.add_argument("--model-uri", default="models:/ChurnPredictionModel/Production", help="MLflow model URI.")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to the churn CSV file.")
    parser.add_argument("--batches", type=int, default=4, help="Number of simulated production batches.")
    args = parser.parse_args()
    simulate_monitoring(args.model_uri, args.data, batches=args.batches)

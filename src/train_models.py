"""Train baseline churn models and log experiments with MLflow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, prepare_data


EXPERIMENT_NAME = "Telco Customer Churn"
TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow() -> str:
    """Configure MLflow with a local SQLite backend and local artifact folder."""

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_location = Path("mlruns").resolve().as_uri()
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def get_models() -> dict[str, object]:
    """Return baseline models for comparison."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate classification metrics used throughout the project."""

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_reports(model_name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Log report files and a confusion matrix artifact to MLflow."""

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        report_path = tmp_path / "classification_report.json"
        matrix_path = tmp_path / "confusion_matrix.png"

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No Churn", "Churn"])
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=160)
        plt.close()

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(matrix_path), artifact_path="figures")


def train_and_log(data_path: str) -> pd.DataFrame:
    """Train all baseline models and log each run."""

    configure_mlflow()
    prepared = prepare_data(data_path)
    results: list[dict[str, object]] = []

    for model_name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", prepared.preprocessor),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=f"baseline_{model_name.replace(' ', '_').lower()}") as run:
            pipeline.fit(prepared.x_train, prepared.y_train)
            metrics = evaluate_model(pipeline, prepared.x_test, prepared.y_test)

            mlflow.set_tags(
                {
                    "project": "Customer Churn Prediction",
                    "run_type": "baseline",
                    "model_name": model_name,
                }
            )
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_rows", len(prepared.x_train))
            mlflow.log_param("test_rows", len(prepared.x_test))
            mlflow.log_param("numeric_features", ",".join(prepared.numeric_features))
            mlflow.log_param("categorical_features", ",".join(prepared.categorical_features))

            classifier_params = {
                f"classifier__{key}": value
                for key, value in estimator.get_params().items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            mlflow.log_params(classifier_params)
            mlflow.log_metrics(metrics)
            log_reports(model_name, pipeline, prepared.x_test, prepared.y_test)

            signature = infer_signature(prepared.x_train.head(5), pipeline.predict(prepared.x_train.head(5)))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=prepared.x_train.head(2),
            )

            result = {
                "run_id": run.info.run_id,
                "model_name": model_name,
                **metrics,
            }
            results.append(result)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/baseline_results.csv", index=False)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline churn models and log MLflow runs.")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to the churn CSV file.")
    args = parser.parse_args()
    train_and_log(args.data)

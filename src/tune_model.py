"""Tune Random Forest with RandomizedSearchCV and track trials in MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, prepare_data
from train_models import configure_mlflow, evaluate_model, log_reports


N_ITER = 20
CV_FOLDS = 3
SCORING = "roc_auc"


def parameter_distributions() -> dict[str, list[object]]:
    """Return the randomized search space for the Random Forest pipeline."""

    return {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [4, 6, 8, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }


def build_search_pipeline(preprocessor: object) -> Pipeline:
    """Build the sklearn pipeline used by RandomizedSearchCV."""

    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def cv_results_to_frame(search: RandomizedSearchCV) -> pd.DataFrame:
    """Convert RandomizedSearchCV results into a compact report table."""

    results_df = pd.DataFrame(search.cv_results_)
    param_columns = [column for column in results_df.columns if column.startswith("param_")]
    selected_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "mean_score_time",
        *param_columns,
    ]
    results_df = results_df[selected_columns].sort_values("rank_test_score")
    results_df = results_df.rename(
        columns={
            "rank_test_score": "cv_rank",
            "mean_test_score": "mean_cv_roc_auc",
            "std_test_score": "std_cv_roc_auc",
        }
    )
    return results_df


def log_cv_trials(results_df: pd.DataFrame) -> None:
    """Log each randomized CV candidate as its own MLflow run."""

    for trial_number, (_, row) in enumerate(results_df.sort_values("cv_rank").iterrows(), start=1):
        with mlflow.start_run(run_name=f"randomized_search_cv_trial_{trial_number:02d}"):
            mlflow.set_tags(
                {
                    "project": "Customer Churn Prediction",
                    "run_type": "tuning_randomized_search",
                    "model_name": "Random Forest",
                    "tuning_method": "RandomizedSearchCV",
                    "trial_number": str(trial_number),
                }
            )
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("scoring", SCORING)
            mlflow.log_param("n_iter", N_ITER)

            for column, value in row.items():
                if column.startswith("param_"):
                    mlflow.log_param(column.replace("param_", ""), value)

            mlflow.log_metric("mean_cv_roc_auc", float(row["mean_cv_roc_auc"]))
            mlflow.log_metric("std_cv_roc_auc", float(row["std_cv_roc_auc"]))
            mlflow.log_metric("cv_rank", int(row["cv_rank"]))
            mlflow.log_metric("mean_fit_time", float(row["mean_fit_time"]))
            mlflow.log_metric("mean_score_time", float(row["mean_score_time"]))


def log_best_model(
    search: RandomizedSearchCV,
    prepared: object,
) -> dict[str, object]:
    """Evaluate, log, and return the best tuned model summary."""

    best_model = search.best_estimator_
    test_metrics = evaluate_model(best_model, prepared.x_test, prepared.y_test)

    with mlflow.start_run(run_name="best_randomized_search_model") as run:
        mlflow.set_tags(
            {
                "project": "Customer Churn Prediction",
                "run_type": "tuning",
                "model_name": "Random Forest",
                "tuning_method": "RandomizedSearchCV",
                "selection_metric": SCORING,
            }
        )
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_iter", N_ITER)
        mlflow.log_param("best_cv_roc_auc", float(search.best_score_))
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(test_metrics)
        log_reports("Best RandomizedSearchCV Random Forest", best_model, prepared.x_test, prepared.y_test)

        signature = infer_signature(prepared.x_train.head(5), best_model.predict(prepared.x_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=prepared.x_train.head(2),
        )

        return {
            "run_id": run.info.run_id,
            "best_cv_roc_auc": float(search.best_score_),
            **search.best_params_,
            **test_metrics,
        }


def tune_random_forest(data_path: str) -> pd.DataFrame:
    """Run RandomizedSearchCV and log CV trials plus the best model."""

    configure_mlflow()
    prepared = prepare_data(data_path)
    pipeline = build_search_pipeline(prepared.preprocessor)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions(),
        n_iter=N_ITER,
        scoring=SCORING,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(prepared.x_train, prepared.y_train)

    results_df = cv_results_to_frame(search)
    best_summary = log_best_model(search, prepared)
    results_df["best_model_run_id"] = best_summary["run_id"]
    results_df["best_test_roc_auc"] = best_summary["roc_auc"]
    results_df["best_test_f1_score"] = best_summary["f1_score"]

    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/tuning_results.csv", index=False)
    log_cv_trials(results_df)

    print("Best RandomizedSearchCV model:")
    for key, value in best_summary.items():
        print(f"{key}: {value}")
    print("\nTop CV trials:")
    print(results_df.head(10).to_string(index=False))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Random Forest with RandomizedSearchCV and MLflow.")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to the churn CSV file.")
    args = parser.parse_args()
    tune_random_forest(args.data)

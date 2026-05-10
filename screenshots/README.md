# Screenshot Checklist

Capture these after starting the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Final screenshot set:

- `01_mlflow_experiment_overview.png`: MLflow experiment run table
- `02_best_model_metrics.png`: production model metrics
- `03_best_model_artifacts.png`: best run artifacts tree
- `04_randomized_search_tuning.png`: RandomizedSearchCV tuning run
- `05_model_registry_overview.png`: `ChurnPredictionModel` registry list
- `06_production_model_version.png`: production and staging model aliases
- `07_model_serving_terminal.png`: MLflow model serving terminal evidence
- `08_prediction_response_terminal.png`: real-time prediction response evidence
- `09_monitoring_metrics.png`: monitoring run metrics

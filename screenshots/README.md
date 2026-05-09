# Screenshot Checklist

Capture these after starting the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Recommended screenshots:

- `01_experiments_overview.png`: MLflow experiment run table
- `02_baseline_metrics.png`: baseline model metric comparison
- `03_best_run_artifacts.png`: best run artifacts, including model and reports
- `04_model_registry.png`: `ChurnPredictionModel` registry page
- `05_production_model.png`: Production stage or production alias
- `06_prediction_response.txt`: `serve_test.py` prediction response

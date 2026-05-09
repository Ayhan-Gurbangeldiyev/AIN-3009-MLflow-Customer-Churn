# 5-Minute Presentation Outline

## Slide 1: Title

Customer Churn Prediction with MLflow  
End-to-End Machine Learning Lifecycle Management

## Slide 2: Problem and Dataset

- Domain: Telecommunications
- Goal: Predict whether a customer will churn
- Dataset: Telco Customer Churn
- Target: `Churn`
- Business value: identify high-risk customers for retention campaigns

## Slide 3: Project Workflow

- Load and preprocess data
- Train baseline models
- Track experiments in MLflow
- Tune Random Forest
- Register best model
- Serve model locally
- Monitor production-like batches

## Slide 4: MLflow Experiment Tracking

- Show MLflow experiments page
- Compare Logistic Regression, Random Forest, and Gradient Boosting
- Explain logged parameters, metrics, artifacts, and model files

Screenshot to show: MLflow runs table and metrics comparison.

## Slide 5: Model Results

- Show baseline and tuning results
- Main metric: ROC-AUC
- Supporting metrics: Accuracy, Precision, Recall, F1-score
- Highlight RandomizedSearchCV: 20 sampled candidates, 3-fold CV, best CV ROC-AUC 0.8476
- Explain why ROC-AUC is useful for churn risk ranking

Screenshot to show: best run metrics and confusion matrix artifact.

## Slide 6: Model Registry

- Registered model name: `ChurnPredictionModel`
- Demonstrate versioning
- Show Staging and Production transition or aliases

Screenshot to show: Model Registry page.

## Slide 7: Deployment and Prediction

- Model served with MLflow Model Serving
- Endpoint: `http://127.0.0.1:5001/invocations`
- `serve_test.py` sends one customer record and receives a prediction

Screenshot to show: serving terminal and prediction response.

## Slide 8: Monitoring and Conclusion

- Monitoring script evaluates simulated production batches
- Logs batch accuracy, F1-score, ROC-AUC, churn rate, and feature summary metrics
- Conclusion: MLflow gives one workflow for tracking, packaging, registry, deployment, and monitoring

Screenshot to show: monitoring metrics run in MLflow.

## Demo Order

1. Open MLflow UI.
2. Show experiments and best model run.
3. Show registered model.
4. Show model serving terminal.
5. Run or show `serve_test.py` response.
6. Show monitoring batch metrics.

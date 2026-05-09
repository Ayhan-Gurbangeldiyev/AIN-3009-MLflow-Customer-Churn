# Customer Churn Prediction with MLflow

Course: AIN-3009 Delivering AI Applications with MLOps  
Project Type: Term Project

## 1. Introduction

This project implements an end-to-end machine learning lifecycle management system using MLflow. The goal is to show how experiment tracking, model training, hyperparameter tuning, model registry, deployment, and monitoring can be managed in a structured workflow.

## 2. Problem Definition

Customer churn is a critical business problem in telecommunications. The task is to predict whether a customer will leave the company based on demographic, service, account, and billing attributes. The target variable is `Churn`, where `Yes` indicates that the customer left.

## 3. Dataset Description

The project uses the Telco Customer Churn dataset. The dataset contains 7,043 customer records and 21 columns. Features include contract type, tenure, monthly charges, total charges, internet services, technical support, payment method, and customer demographics.

Key dataset facts:

- Target column: `Churn`
- Positive class: `Yes`
- Negative class: `No`
- Class distribution: 5,174 non-churn customers and 1,869 churn customers
- `TotalCharges` requires numeric conversion because it is stored as a text column

## 4. Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- MLflow Tracking
- MLflow Model Registry
- MLflow Model Serving
- SQLite backend store for MLflow metadata

## 5. MLflow Setup

MLflow is configured locally with a SQLite backend store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The project scripts create and use the experiment named `Telco Customer Churn`. Model artifacts, metrics, parameters, reports, and plots are logged for every relevant run.

## 6. Data Preprocessing

The preprocessing pipeline performs the following steps:

- Drops `customerID` because it is an identifier, not a predictive feature.
- Converts `TotalCharges` to numeric values.
- Encodes `Churn` as a binary target.
- Splits the dataset into train and test sets using stratification.
- Applies median imputation and scaling to numeric columns.
- Applies most-frequent imputation and one-hot encoding to categorical columns.

The preprocessing is implemented with Scikit-learn `Pipeline` and `ColumnTransformer`, which keeps training and inference transformations consistent.

## 7. Model Training

Three baseline models are trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each training run logs:

- Model name
- Model parameters
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- Serialized model artifact

## 8. Experiment Tracking

MLflow Tracking is used to compare baseline and tuning runs. This allows direct comparison of model metrics and makes it easy to identify the best candidate model.

The main comparison metric is ROC-AUC because churn prediction benefits from ranking customers by churn risk, not only predicting hard labels.

## 9. Hyperparameter Tuning

Random Forest is tuned using Scikit-learn `RandomizedSearchCV` with 3-fold cross-validation. This improves the tuning process by sampling a broader search space while keeping runtime manageable for a local project demo.

The search space includes:

- `classifier__n_estimators`
- `classifier__max_depth`
- `classifier__min_samples_split`
- `classifier__min_samples_leaf`
- `classifier__max_features`

Each sampled candidate is logged as a separate MLflow run with mean cross-validation ROC-AUC, standard deviation, rank, and hyperparameters. The best refitted model is logged as a separate MLflow model run with test-set metrics and artifacts. The final selected model is the run with the highest ROC-AUC score.

## 10. Model Registry

The best model is registered as:

```text
ChurnPredictionModel
```

The registry workflow demonstrates model versioning and lifecycle management. The selected model version is moved through Staging and Production where supported by the installed MLflow version. Aliases are also assigned for newer MLflow versions.

## 11. Model Deployment

The production model is deployed locally with MLflow Model Serving:

```bash
mlflow models serve -m "models:/ChurnPredictionModel/Production" -p 5001 --no-conda
```

The `serve_test.py` script sends a sample customer record to the `/invocations` endpoint and prints the prediction response.

## 12. Performance Monitoring

The monitoring script simulates production batches using held-out test data. For each batch, the script logs:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Actual churn rate
- Predicted churn rate
- Average tenure
- Average monthly charges

The script also simulates gradually noisier production outcomes to demonstrate how model performance could be tracked over time.

## 13. Results and Discussion

The baseline experiment produced the following model comparison:

- Random Forest: Accuracy 0.7601, F1-score 0.6286, ROC-AUC 0.8434
- Gradient Boosting: Accuracy 0.8062, F1-score 0.5895, ROC-AUC 0.8434
- Logistic Regression: Accuracy 0.7381, F1-score 0.6136, ROC-AUC 0.8413

The best tuning model was selected using RandomizedSearchCV:

- Cross-validation metric: ROC-AUC
- Cross-validation folds: 3
- Randomized search iterations: 20
- Best cross-validation ROC-AUC: 0.8476
- Best test ROC-AUC: 0.8411
- Best test F1-score: 0.6223
- Best parameters: 200 estimators, max depth 8, min samples split 2, min samples leaf 2, max features `sqrt`
- Best test metrics are stored in `reports/tuning_results.csv` and MLflow under `best_randomized_search_model`

The final registered production model remains the best test-set model in MLflow, registered as `ChurnPredictionModel`. The model serving endpoint was tested successfully with a sample customer and returned:

```json
{
  "predictions": [1]
}
```

The monitoring simulation logged four production-like batches. ROC-AUC decreased from 0.8424 in batch 1 to 0.7209 in batch 4 under increasing simulated outcome noise, demonstrating how MLflow can track performance changes over time.

Expected output files:

- `reports/baseline_results.csv`
- `reports/tuning_results.csv`
- `reports/registered_model.json`
- `reports/monitoring_metrics.csv`

## 14. Conclusion

This project demonstrates how MLflow can manage the complete lifecycle of a machine learning model, from experimentation to deployment and monitoring. The system provides a practical local workflow that can be extended to cloud storage, scheduled retraining, and production monitoring in a real-world MLOps environment.

## 15. Evidence and Screenshots

The following screenshots are included in the `screenshots/` directory:

- `01_experiments_runs.png`: MLflow experiment run list
- `02_best_run_metrics.png`: best RandomizedSearchCV model run
- `03_monitoring_run.png`: simulated monitoring run
- `04_model_registry.png`: registered model list
- `05_production_model.png`: production and staging aliases for `ChurnPredictionModel`

## 16. Reflection and Lessons Learned

This project shows that MLflow is useful not only for saving model metrics, but for organizing the full model lifecycle. Experiment tracking made it easier to compare baseline models and tuning results. The Model Registry provided a structured way to identify the production model. The monitoring simulation showed how model quality can be tracked over time, even when real production traffic is not available.

The main practical lesson is that MLOps is more than training a model. A useful ML system also needs repeatable preprocessing, logged experiments, reproducible artifacts, model versioning, deployment instructions, and monitoring evidence.

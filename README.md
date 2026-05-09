# Customer Churn Prediction with MLflow

Course: AIN-3009 Delivering AI Applications with MLOps  
Project: Term Project - Development and Evaluation of a Machine Learning Lifecycle Management System using MLflow
GitHub Repository: https://github.com/Ayhan-Gurbangeldiyev/AIN-3009-MLflow-Customer-Churn

This term project demonstrates an end-to-end machine learning lifecycle management system using MLflow. The selected domain is telecommunications, and the predictive task is customer churn prediction using the Telco Customer Churn dataset.

## Project Goals

- Track multiple ML experiments with MLflow.
- Train and compare Scikit-learn classification models.
- Tune hyperparameters and log every tuning trial.
- Register the best model in the MLflow Model Registry.
- Serve the production model for real-time predictions.
- Simulate production monitoring with batch-level metrics.

## Project Structure

```text
Project/
├── data/
│   └── telco_churn.csv
├── models/
├── notebooks/
├── presentation/
├── reports/
├── screenshots/
├── src/
│   ├── data_preprocessing.py
│   ├── monitor_model.py
│   ├── register_model.py
│   ├── serve_test.py
│   ├── train_models.py
│   └── tune_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

- Dataset: Telco Customer Churn
- File: `data/telco_churn.csv`
- Target column: `Churn`
- Rows: 7,043
- Columns: 21
- Positive class: customers who churned (`Churn = Yes`)

Important preprocessing steps:

- Drop `customerID`.
- Convert `TotalCharges` from string/object to numeric.
- Encode `Churn` as `Yes=1`, `No=0`.
- Impute numeric and categorical missing values.
- One-hot encode categorical columns.
- Split data with stratification.

## Setup

Run all commands from the project root:

```bash
cd "/Users/ayhan/Dersler/MLOps/HW and Project/Term Project/Project"
source .venv/bin/activate
pip install -r requirements.txt
```

## MLflow UI

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## 1. Validate Dataset

```bash
python src/data_preprocessing.py --data data/telco_churn.csv
```

Expected checks:

- `TotalCharges` is converted to numeric.
- `Churn` exists and is binary after preprocessing.
- Dataset shape is 7,043 rows and 21 columns before feature cleanup.

## 2. Train Baseline Models

```bash
python src/train_models.py --data data/telco_churn.csv
```

Models trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

Logged to MLflow:

- Parameters
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- Trained model artifact

Baseline results are also written to:

```text
reports/baseline_results.csv
```

## 3. Hyperparameter Tuning

```bash
python src/tune_model.py --data data/telco_churn.csv
```

The tuning script runs `RandomizedSearchCV` for Random Forest with 3-fold cross-validation and logs each sampled candidate as an MLflow run. It also logs the best refitted model as a separate MLflow run with test metrics and model artifacts. Results are written to:

```text
reports/tuning_results.csv
```

## 4. Register Best Model

```bash
python src/register_model.py --model-name ChurnPredictionModel
```

This script:

- Selects the best run by ROC-AUC.
- Registers it as `ChurnPredictionModel`.
- Moves the version through Staging and Production where supported.
- Also assigns MLflow aliases `staging` and `production` when available.

Registration summary is written to:

```text
reports/registered_model.json
```

## 5. Serve Production Model

Start model serving:

```bash
PATH=.venv/bin:$PATH MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/ChurnPredictionModel/Production" -p 5001 --no-conda
```

If using aliases in a newer MLflow setup:

```bash
PATH=.venv/bin:$PATH MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/ChurnPredictionModel@production" -p 5001 --no-conda
```

## 6. Test Prediction Endpoint

In a second terminal:

```bash
python src/serve_test.py --url http://127.0.0.1:5001/invocations
```

The script sends one sample customer record and prints the prediction response.

## 7. Simulate Monitoring

```bash
python src/monitor_model.py --model-uri "models:/ChurnPredictionModel/Production"
```

If using aliases:

```bash
python src/monitor_model.py --model-uri "models:/ChurnPredictionModel@production"
```

Monitoring logs batch-level metrics to MLflow:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Actual churn rate
- Predicted churn rate
- Average tenure
- Average monthly charges

Monitoring output is also written to:

```text
reports/monitoring_metrics.csv
```

## Screenshots for Submission

Save screenshots under `screenshots/`:

- MLflow experiments page
- Baseline model runs
- Metrics comparison
- Model artifacts
- Model Registry page
- Staging / Production transition
- Model serving terminal
- Prediction response
- Monitoring metrics

## Report and Presentation

Report draft:

```text
reports/project_report.md
```

Presentation outline:

```text
presentation/presentation_outline.md
```

Browser-based slide deck:

```text
presentation/index.html
```

Key MLflow proof screenshots are stored in:

```text
screenshots/
```

## Suggested Submission Package

After completing screenshots and final report:

```bash
zip -r PRJ-AyhanGurbangeldiyev-2020053.zip . -x ".venv/*" "__pycache__/*" "*.DS_Store"
```

# Presentation Script

## Slide 1 - Title
Hello, today I will present my MLOps term project: Customer Churn Prediction with MLflow. The goal is not only to train a model, but to manage the full machine learning lifecycle from experiments to deployment and monitoring.

## Slide 2 - Problem and Dataset
The selected problem is customer churn prediction in telecommunications. The model predicts whether a customer will leave. This is useful because companies can use churn risk to prioritize retention campaigns.

## Slide 3 - Workflow
This is the full lifecycle workflow. The project starts with preprocessing, then baseline training, hyperparameter tuning, model registry, serving, and finally simulated monitoring.

## Slide 4 - Data Preparation and Models
Preprocessing is implemented as a Scikit-learn pipeline. This is important because the same transformations are used during training and serving. I compared three baseline models before tuning.

## Slide 5 - Experiment Tracking
This screenshot shows the MLflow experiment. Each run logs parameters, metrics, artifacts, and model outputs, making the comparison auditable and reproducible.

## Slide 6 - Hyperparameter Tuning
For tuning, I used RandomizedSearchCV instead of a manual grid. It samples a broader parameter space while keeping runtime manageable. The best cross-validation ROC-AUC was about 0.8476.

## Slide 7 - Model Registry
The best model is registered as ChurnPredictionModel. The registry demonstrates model versioning and lifecycle management, including staging and production aliases.

## Slide 8 - Deployment and Monitoring
The production model is served locally with MLflow Models. The serve_test script sends a sample customer and receives a prediction. Monitoring then evaluates four simulated batches and logs metrics like F1-score and ROC-AUC.

## Slide 9 - Conclusion and Demo Checklist
To conclude, the project satisfies the main lifecycle requirements: tracking, training, tuning, registry, deployment, and monitoring. In the demo, I would show the MLflow experiment, best run, registry model, prediction response, and monitoring metrics.


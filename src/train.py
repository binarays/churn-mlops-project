# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import mlflow
import mlflow.sklearn
import joblib

# -----------------------------
# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/binarays/churn-mlops-project.mlflow")
mlflow.set_experiment("churn-experiment")

# -----------------------------
# Load processed data
data = pd.read_csv("data/processed/telco_customer_churn_data.csv")

X = data.drop("Churn", axis=1)
y = data["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# -----------------------------
# Train and log with MLflow
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # Log parameters (example: model hyperparams)
        mlflow.log_param("model_name", model_name)
        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
        if model_name == "XGBoost":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")

print("Training complete ✅")

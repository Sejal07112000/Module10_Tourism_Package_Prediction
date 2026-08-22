"""
Loads the train/test splits, tunes an XGBoost classifier with GridSearchCV,
tracks the experiment (params + metrics) with MLflow, evaluates the best
model, and saves it so the pipeline can commit it to the repository.
"""

import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_OUTPUT_PATH = "tourism_project/deployment/model.joblib"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


def build_pipeline() -> "make_pipeline":
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    return make_pipeline(preprocessor, model)


def train_model():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    # Falls back to a local sqlite-backed store if no tracking server is
    # reachable (e.g. when this script is run outside the CI workflow).
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("tourism-wellness-package")

    pipeline = build_pipeline()

    with mlflow.start_run():
        grid = GridSearchCV(pipeline, PARAM_GRID, cv=3, scoring="recall", n_jobs=-1)
        grid.fit(Xtrain, ytrain)

        mlflow.log_params(grid.best_params_)

        best_model = grid.best_estimator_
        preds = best_model.predict(Xtest)

        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1_score": f1_score(ytest, preds),
        }
        mlflow.log_metrics(metrics)

        print("Best parameters:")
        print(grid.best_params_)
        print("\nClassification report on test data:\n")
        print(classification_report(ytest, preds))

        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        mlflow.log_artifact(MODEL_OUTPUT_PATH)
        print(f"Best model saved to {MODEL_OUTPUT_PATH}")

    return best_model, metrics


if __name__ == "__main__":
    train_model()

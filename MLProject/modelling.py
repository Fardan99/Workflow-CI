
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_preprocessing")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

mlflow.sklearn.autolog(log_models=True)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

with mlflow.start_run(run_name="RandomForest_CI"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1_score": f1_score(y_test, predictions, zero_division=0)
    }

    mlflow.log_metrics(metrics)

    print("HASIL EVALUASI MODEL")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

print("MLflow Project berhasil menjalankan pelatihan model.")

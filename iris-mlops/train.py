import os
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

iris = load_iris(as_frame=True)
df = iris.data.copy()
df["target"] = iris.target

df = df.drop(index=params["remove_rows"], errors="ignore")

update = params["update"]
if update["enabled"]:
    df.loc[update["row"], update["feature"]] = update["value"]

df.to_csv("data/iris.csv", index=False)

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layers"]),
        max_iter=params["max_iter"],
        random_state=42
    ))
])

mlflow.set_experiment("Iris-Neural-Network")

with mlflow.start_run():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("removed_rows", str(params["remove_rows"]))
    mlflow.log_param("dataset_rows", len(df))

    mlflow.sklearn.log_model(
    model,
    name="iris_model",
    serialization_format="cloudpickle"
)
    joblib.dump(model, "models/iris_model.joblib")

    with open("metrics/results.json", "w") as file:
        json.dump({
            "accuracy": accuracy,
            "dataset_rows": len(df)
        }, file, indent=4)

    print("Dataset rows:", len(df))
    print("Accuracy:", round(accuracy, 4))

print("Model saved successfully!")
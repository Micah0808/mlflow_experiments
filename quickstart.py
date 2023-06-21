import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

import os
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts


# =================================================================================================
# Auto-logging
# =================================================================================================
mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

# =================================================================================================
# Manual-logging
# =================================================================================================
# Log a parameter (key-value pair)
log_param("config_value", randint(0, 100))

# Log a dictionary of parameters
log_params({"param1": randint(0, 100), "param2": randint(0, 100)})

# Log a metric; metrics can be updated throughout the run
log_metric("accuracy", random() / 2.0)
log_metric("accuracy", random() + 0.1)
log_metric("accuracy", random() + 0.2)

# Log an artifact (output file)
if not os.path.exists("outputs"):
    os.makedirs("outputs")
with open("outputs/test.txt", "w") as f:
    f.write("hello world!")
log_artifacts("outputs")

# =================================================================================================
# Storing a model in  MLflow
# =================================================================================================
# This just saves the model but does not log any metrics like autologging above.

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)

    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    print("Run ID: {}".format(run.info.run_id))

# =================================================================================================
# Load a model from a specific training run for inference
# =================================================================================================

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("runs:/77654a6641fb46ccb4578a9cc5aa16fa/model")
predictions = model.predict(X_test)
print(predictions)


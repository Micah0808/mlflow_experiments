# ML models

from typing import Union, Tuple, Any, Dict, Callable, NoReturn

# MLFlow autologging tries to spin up a Matplotlib GUI outside the main thread which leads to
# exceptions when running the MLFlow task in the Prefect flow. Therefore, I am disabling this here
# with matplotlib.use('Agg').
import matplotlib
import mlflow
import mlflow.xgboost

# Data wrangling
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import (
    fmin,
    hp,
    tpe,
    Trials,
    STATUS_OK
)
from hyperopt.pyll.base import scope
from mlflow import MlflowClient
from mlflow.entities import ModelVersion
from mlflow.models.signature import infer_signature

# ML Ops
from prefect import task, flow
from prefect_shell import ShellOperation

# ML modelling
from sklearn import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import Booster

matplotlib.use('Agg')

RANDOM_SEED = 0


# mlflow.set_tracking_uri('http://0.0.0.0:5000')


@task(log_prints=True)
def load_data(as_frame: bool = True) -> Union[pd.DataFrame, None]:
    """
    Function to load the breast cancer dataset and append the target values to the DataFrame.

    Args:
        as_frame (bool): Flag indicating whether to load data as pandas DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the breast cancer dataset with target values
                      appended, if `as_frame=True`. Otherwise, return None.
    """
    if as_frame:
        data = datasets.load_breast_cancer(as_frame=as_frame)
        data_df = data.data
        data_df['target'] = data.target
        print(f"{data_df.shape[0]} rows and {data_df.shape[1]} columns in the dataset.")
        return data_df
    else:
        print("This function only supports pandas DataFrame output.")
        return None


@task(log_prints=True)
def split_data(data: pd.DataFrame,
               test_size: float = 0.1,
               random_seed: Any = None) -> Tuple[pd.DataFrame, ...]:
    """
    Function to split the provided dataframe into training, validation and test sets.

    Args:
        data (pd.DataFrame): Input dataframe to be split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_seed (Any): The seed used by the random number generator.

    Returns:
        Tuple[pd.DataFrame, ...]: A tuple containing the training, validation and test sets
        (including target values).
    """

    # Splitting the dataset into training/validation and holdout sets
    train_val, test = train_test_split(
        data,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed
    )

    # Creating X, y for training/validation set
    X_train_val = train_val.drop(columns='target')
    y_train_val = train_val.target

    # Creating X, y for test set
    X_test = test.drop(columns='target')
    y_test = test.target

    # Splitting training/testing set to create training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        stratify=y_train_val,
        shuffle=True,
        random_state=random_seed
    )
    print(f"Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns.")
    print(f"Test set has {X_test.shape[0]} rows and {X_test.shape[1]} columns.")

    return X_train, X_val, X_test, y_train, y_val, y_test


@task
def preprocess_data(X_train: pd.DataFrame,
                    X_val: pd.DataFrame,
                    X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to preprocess data using the PowerTransformer.

    Args:
        X_train (DataFrame): The training dataset.
        X_val (DataFrame): The validation dataset.
        X_test (DataFrame): The test dataset.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: A tuple containing the preprocessed training,
        validation and test datasets.
    """

    # Preprocessing data
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_transformed = power.fit_transform(X_train)
    X_val_transformed = power.transform(X_val)
    X_test_transformed = power.transform(X_test)

    return X_train_transformed, X_val_transformed, X_test_transformed


@task
def run_training_flow(experiment_id: str,
                      training_setup: Callable,
                      search_space: Dict) -> Dict:
    """
        Runs the training flow for an XGBoost model using hyperparameter optimization.

        This function executes the training setup within an MLflow run context, where each
        hyperparameter configuration is logged as a child run of a parent run called
        "xgboost_models". It requires Spark to be configured and installed to run.

        Args:
            experiment_id (str): The id of the MLflow experiment.
            training_setup (Callable): The function to setup and execute the training.
            search_space (Dict): The hyperparameter search space for the XGBoost model.

        Returns:
            Dict: A dictionary of the best parameters for the XGBoost model.
        """
    # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
    # A reasonable value for parallelism is the square root of max_evals.

    # Will need spark configured and installed to run. Add this to fmin function below like so:
    trials = Trials()

    # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a
    # child run of a parent run called "xgboost_models" .
    with mlflow.start_run(experiment_id=experiment_id, run_name='xgboost_models'):
        xgboost_best_params = fmin(
            fn=training_setup,
            space=search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=10
        )


@task(log_prints=True)
def load_best_model_from_runs(experiment_id: str) -> Union[Booster, None]:
    """
    Function to load the best XGBoost model from the runs in the given MLflow experiment.

    Args:
        experiment_id (str): The id of the MLflow experiment.

    Returns:
        Booster: The best XGBoost model from the runs in the MLflow experiment, if any exist.
                 Otherwise, return None.
    """
    runs_df = mlflow.search_runs(experiment_ids=experiment_id,
                                 order_by=['metrics.validation_aucroc DESC'])
    if runs_df.empty:
        print("No runs found in the experiment.")
        return None
    else:
        best_run = runs_df.iloc[0]
        best_run_id = best_run['run_id']

        # Loading model from best run
        best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

        return best_model, best_run, best_run_id


@task(log_prints=True)
def evaluate_model_on_holdout_set(model: Booster,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray) -> Dict[str, float]:
    """
    Function to evaluate the performance of a model on a holdout set.

    Args:
        model (Booster): The XGBoost model to be evaluated.
        X_test (np.ndarray): The feature values of the holdout set.
        y_test (np.ndarray): The true labels of the holdout set.

    Returns:
        Dict[str, float]: A dictionary with the evaluation metrics and their values.
    """
    # Predicting and evaluating model on holdout set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
    test_precision = precision_score(y_test, y_test_pred).round(3)
    test_recall = recall_score(y_test, y_test_pred).round(3)
    test_f1 = f1_score(y_test, y_test_pred).round(3)
    test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

    metrics = {
        "Testing Accuracy": test_accuracy,
        "Testing Precision": test_precision,
        "Testing Recall": test_recall,
        "Testing F1": test_f1,
        "Testing AUCROC": test_aucroc
    }

    print(f'Testing Accuracy: {test_accuracy}')
    print(f'Testing Precision: {test_precision}')
    print(f'Testing Recall: {test_recall}')
    print(f'Testing F1: {test_f1}')
    print(f'Testing AUCROC: {test_aucroc}')

    return metrics


@task(log_prints=True)
def register_and_update_model(run_id: str,
                              model_description: str,
                              model_version_description: str,
                              model_name: str):
    """
    Function to register an MLflow model and update its information.

    Args:
        run_id (str): The id of the MLflow run where the model was trained.
        model_description (str): The description to be set for the registered model.
        model_version_description (str): The description to be set for the version of the
                                         registered model.
        model_name (str): The name to be given to the registered model.

    Returns:
        Tuple[ModelVersion, str, str]: A tuple containing the ModelVersion object for the registered
                                       model, the model description and the model version
                                       description.
    """
    model_details = mlflow.register_model(f'runs:/{run_id}/artifacts/model', model_name)

    client = MlflowClient()
    client.update_registered_model(
        name=model_details.name,
        description=model_description
    )

    client.update_model_version(
        name=model_details.name,
        version=model_details.version,
        description=model_version_description
    )

    print("Model Details:", model_details)
    print("Model Description:", model_description)
    print("Model Version Description:", model_version_description)

    return client, model_details, model_description, model_version_description


@task
def stage_best_model_to_production(client: MlflowClient,
                                   model_details: ModelVersion) -> NoReturn:
    """
    Function to transition a model version to the production stage.

    Args:
        client (MlflowClient): An instance of MlflowClient.
        model_details (ModelVersion): The details of the model version to be transitioned.

    Returns:
        None
    """
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage='Production'
    )


# TODO need to change serving code to fetch the best model based off of id


@task
def serve_model_locally():
    ShellOperation(commands=[
        """
        curl https://pyenv.run | bash
        """,
        """
        python -m  pip install virtualenv
        """,
        """
        PATH="$HOME/.pyenv/bin:$PATH"
        """,
        """
        mlflow models serve -m "/Users/micahcearns/Desktop/IT_Projects/mlflow_project/mlruns/422345479337653825/0b408f18e5b84484917fc8f05ef396cc/artifacts/model" --port 5004 &
        """
    ]).run()


@task
def test_served_model():
    ShellOperation(commands=[
        """
        curl -d '{"dataframe_split": {"columns": ["mean radius", "mean texture", 
        "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"], 
        "data": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}}' \
        -H 'Content-Type: application/json' -X POST localhost:5004/invocations
        """
    ]).run()


@flow
def main_flow():
    df = load_data(as_frame=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data=df,
        test_size=0.1,
        random_seed=RANDOM_SEED
    )
    X_train_transformed, X_val_transformed, X_test_transformed = preprocess_data(
        X_train, X_val, X_test
    )

    # Setting search space for xgboost model
    search_space = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 100)),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'subsample': hp.uniform('subsample', .5, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
        'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
        'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
        'gamma': hp.loguniform('gamma', -10, 10),
        'use_label_encoder': False,
        'verbosity': 0,
        'random_state': RANDOM_SEED
    }

    try:
        EXPERIMENT_ID = mlflow.create_experiment('xgboost-hyperopt')
    except:
        EXPERIMENT_ID = dict(mlflow.get_experiment_by_name('xgboost-hyperopt'))['experiment_id']

    def train_model(params):
        """
        Creates a hyperopt training model function that sweeps through params in a nested run
        Args:
            params: hyperparameters selected from the search space
        Returns:
            hyperopt status and the loss metric value
        """

        # With MLflow autologging, hyperparameters and the trained model are automatically logged to
        # MLflow. This sometimes doesn't log everything you may want so I usually log my own metrics
        # and params just in case.
        mlflow.xgboost.autolog()

        with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
            # Training xgboost classifier
            model = xgb.XGBClassifier(**params)
            model = model.fit(X_train, y_train)

            # Predicting values for training and validation data, and getting prediction probabilities
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            y_val_pred = model.predict(X_val)
            y_val_pred_proba = model.predict_proba(X_val)[:, 1]

            # Evaluating model metrics for training set predictions and validation set predictions
            # Creating training and validation metrics dictionaries to make logging in mlflow easier
            metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc']
            # Training evaluation metrics
            train_accuracy = accuracy_score(y_train, y_train_pred).round(3)
            train_precision = precision_score(y_train, y_train_pred).round(3)
            train_recall = recall_score(y_train, y_train_pred).round(3)
            train_f1 = f1_score(y_train, y_train_pred).round(3)
            train_aucroc = roc_auc_score(y_train, y_train_pred_proba).round(3)
            training_metrics = {
                'Accuracy': train_accuracy,
                'Precision': train_precision,
                'Recall': train_recall,
                'F1': train_f1,
                'AUCROC': train_aucroc
            }
            training_metrics_values = list(training_metrics.values())

            # Validation evaluation metrics
            val_accuracy = accuracy_score(y_val, y_val_pred).round(3)
            val_precision = precision_score(y_val, y_val_pred).round(3)
            val_recall = recall_score(y_val, y_val_pred).round(3)
            val_f1 = f1_score(y_val, y_val_pred).round(3)
            val_aucroc = roc_auc_score(y_val, y_val_pred_proba).round(3)
            validation_metrics = {
                'Accuracy': val_accuracy,
                'Precision': val_precision,
                'Recall': val_recall,
                'F1': val_f1,
                'AUCROC': val_aucroc
            }
            validation_metrics_values = list(validation_metrics.values())

            # Logging model signature, class, and name
            signature = infer_signature(X_train_transformed, y_val_pred)
            mlflow.xgboost.log_model(model, 'model', signature=signature)
            mlflow.set_tag('estimator_name', model.__class__.__name__)
            mlflow.set_tag('estimator_class', model.__class__)

            # Logging each metric
            for name, metric in list(zip(metric_names, training_metrics_values)):
                mlflow.log_metric(f'training_{name}', metric)
            for name, metric in list(zip(metric_names, validation_metrics_values)):
                mlflow.log_metric(f'validation_{name}', metric)

            # Set the loss to -1*validation auc roc so fmin maximizes it
            return {'status': STATUS_OK, 'loss': -1 * validation_metrics['AUCROC']}

    run_training_flow(experiment_id=EXPERIMENT_ID, training_setup=train_model, search_space=search_space)
    best_model, best_run, best_run_id = load_best_model_from_runs(experiment_id=EXPERIMENT_ID)
    evaluate_model_on_holdout_set(model=best_model, X_test=X_test, y_test=y_test)

    model_name = 'BreastCancerClassification-XGBHP'
    model_description = """
        This model classifies breast cancer as malignant or benign given certain numerical features of cell nuclei such as
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry
        j) fractal dimension ("coastline approximation" - 1).
        """
    model_version_description = """
        This model version is the first XGBoost model trained with HyperOpt for bayesian 
        hyperparameter tuning.
        """

    client, model_details, model_description, model_version_description = (
        register_and_update_model(run_id=best_run_id,
                                  model_description=model_description,
                                  model_version_description=model_version_description,
                                  model_name=model_name)
    )

    stage_best_model_to_production(client=client, model_details=model_details)
    serve_model_locally()
    test_served_model()


if __name__ == '__main__':
    main_flow()


# TODO
#  Experiment with setting up scheduling and deploying to production
#  Use Prefect-Shell to deploy a MLFlow docker image container to GCP:
#   - https://prefecthq.github.io/prefect-shell/
#   - https://prefecthq.github.io/prefect-gcp/
#  Once complete look into setting up CI/CD with something like GitLab or Buddy


# # =================================================================================================
# # Serving locally to test predictions
# # =================================================================================================
# # Run this if problems with pyenv
# # curl https://pyenv.run | bash
# # python -m  pip install virtualenv
# # PATH="$HOME/.pyenv/bin:$PATH"
#
# # # To serve
# # mlflow models serve - m "/Users/micahcearns/Desktop/IT_Projects/mlflow_project/mlruns" \
# #                         "/422345479337653825/0b408f18e5b84484917fc8f05ef396cc/artifacts/model" - \
# #                         -port 5002
#
# # To send a request
# curl -d '{"dataframe_split": {
# "columns": ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"],
# "data": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}}' \
# -H 'Content-Type: application/json' -X POST localhost:5002/invocations
#
# # =================================================================================================
# # Creating a Docker image
# # =================================================================================================

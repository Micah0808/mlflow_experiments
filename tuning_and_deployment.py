from math import exp
from sklearn import datasets
from pandas_profiling import ProfileReport
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
from hyperopt import (
    fmin,
    hp,
    tpe,
    rand,
    SparkTrials,
    Trials,
    STATUS_OK
)
from hyperopt.pyll.base import scope

RANDOM_SEED = 0
# mlflow.set_tracking_uri('http://0.0.0.0:5000')

# =================================================================================================
# Data setup
# =================================================================================================
data = datasets.load_breast_cancer(as_frame=True)
data_df = data.data
data_df['target'] = data.target
print(data_df)

# # =================================================================================================
# # Pandas profiling
# # =================================================================================================
# data_profile = ProfileReport(data_df)
# data_profile.to_file(f'data_profile.html')

# =================================================================================================
# Train/test/split and pre-process
# =================================================================================================
# Splitting the dataset into training/validation and holdout sets
train_val, test = train_test_split(
    data_df,
    test_size=0.1,
    shuffle=True,
    random_state=RANDOM_SEED
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
    random_state=RANDOM_SEED
)

# Preprocessing data
power = PowerTransformer(method='yeo-johnson', standardize=True)
X_train = power.fit_transform(X_train)
X_val = power.transform(X_val)
X_test = power.transform(X_test)
print(X_train)

# =================================================================================================
# Hyperparameter tuning
# =================================================================================================
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
    Creates a hyperopt training model funciton that sweeps through params in a nested run
    Args:
        params: hyperparameters selected from the search space
    Returns:
        hyperopt status and the loss metric value
    """
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to
    # MLflow. This sometimes doesn't log everything you may want so I usually log my own metrics
    # and params just in case.
    mlflow.xgboost.autolog()

    #
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
        signature = infer_signature(X_train, y_val_pred)
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


# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.

# Will need spark configured and installed to run. Add this to fmin function below like so:
trials = Trials()

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a
# child run of a parent run called "xgboost_models" .
with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_models'):
    xgboost_best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=50
    )

# =================================================================================================
# Testing on the test set
# =================================================================================================
# Querying mlflow api instead of using web UI. Sorting by validation aucroc and then getting top
# run for best run.
runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.validation_aucroc DESC'])
best_run = runs_df.iloc[0]
best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']

# Loading model from best run
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

# Predicting and evaluating best model on holdout set
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
test_precision = precision_score(y_test, y_test_pred).round(3)
test_recall = recall_score(y_test, y_test_pred).round(3)
test_f1 = f1_score(y_test, y_test_pred).round(3)
test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

# =================================================================================================
# Registering the model into the Model Registry
# =================================================================================================
# Registering the model
model_details = mlflow.register_model(f'runs:/{best_run_id}/artifacts/model',
                                      'BreastCancerClassification-XGBHP')

# And updating its information
client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="""This model classifies breast cancer as malignant or benign given certain numerical features of cell nuclei such as 
  a) radius (mean of distances from center to points on the perimeter)
  b) texture (standard deviation of gray-scale values)
  c) perimeter
  d) area
  e) smoothness (local variation in radius lengths)
  f) compactness (perimeter^2 / area - 1.0)
  g) concavity (severity of concave portions of the contour)
  h) concave points (number of concave portions of the contour)
  i) symmetry
  j) fractal dimension ("coastline approximation" - 1)."""
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description='This model version is the first XGBoost model trained with HyperOpt for bayesian hyperparameter tuning.'
)

# =================================================================================================
# Staging the model for production
# =================================================================================================
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production'
)




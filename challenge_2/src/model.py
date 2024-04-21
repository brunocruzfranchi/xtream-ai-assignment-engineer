import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

DEFAULT_MODEL_PARAM = {
    "learning_rate": 0.04595961020010484,
    "n_estimators": 181,
    "max_depth": 5,
    "min_child_weight": 10,
    "colsample_bytree": 1.0,
    "subsample": 0.9,
    "gamma": 0,
    "reg_lambda": 0.1,
}


def train_model(X_train, y_train) -> xgb.XGBRegressor:
    """
    Trains an XGBoost regression model using the given training data.

    Parameters:
        X_train: The input features for training.
        y_train: The target values for training.

    Returns:
        xgb_model (xgb.XGBRegressor): The trained XGBoost regression model.
    """

    xgb_model = xgb.XGBRegressor(
        **DEFAULT_MODEL_PARAM,
        nthread=-1,
        verbosity=0,
    )

    xgb_model.fit(X_train, y_train)

    return xgb_model


def eval_model(model: xgb.XGBRegressor, X, y) -> float:
    """Evaluate the performance of a machine learning model.

    Parameters:
        model (xgb.XGBRegressor): The trained machine learning model.
        X : The input features for evaluation.
        y : The true labels for evaluation.

    Returns:
        float: The root mean squared error (RMSE) between the true labels and the predicted labels.
    """

    y_pred = model.predict(X)

    return np.sqrt(mean_squared_error(y_pred, y))


def save_model(model: xgb.XGBRegressor, output_dir: Path):
    """
    Save the XGBoost model to the specified output directory.

    Args:
        model: The XGBoost model to be saved.
        output_dir (Path): The directory where the model will be saved.

    Returns:
        Path: The path to the saved model file.
    """

    model_name = f"xgboost_model-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    model_output_dir = Path.joinpath(output_dir, model_name)

    model.save_model(model_output_dir)

    return model_output_dir


def training_pipeline(X_train, y_train, X_test, y_test, output_dir: Path):
    """Train a model using the provided training data and evaluate its performance on the test data.
    Save the trained model to the specified output directory.

    Args:
        X_train: The training data features.
        y_train: The training data labels.
        X_test: The test data features.
        y_test: The test data labels.
        output_dir (Path): The directory to save the trained model.
    """

    model = train_model(X_train, y_train)

    train_eval_rmse = eval_model(model, X_train, y_train)

    test_eval_rmse = eval_model(model, X_test, y_test)

    print(f"\033[92mINFO:\033[0m Train RMSE: {train_eval_rmse}")
    print(f"\033[92mINFO:\033[0m Test RMSE: {test_eval_rmse}")

    model_path = save_model(model, output_dir)

    print(f"\033[92mINFO:\033[0m The model has been saved to {model_path}")

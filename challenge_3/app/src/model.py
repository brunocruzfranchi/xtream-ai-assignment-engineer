import numpy as np
import xgboost as xgb

DEFAULT_PARAMETERS_ORDER = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "cut_category",
    "color_category",
    "clarity_category",
    "z_depth",
    "table_width",
]


class RegressorModel:
    regressor = None

    def __init__(self, model_path: str):
        self.regressor = xgb.XGBRegressor()
        self.regressor.load_model(model_path)

    def predict(self, data):
        return self.regressor.predict(data)

    def preprocess_input(self, input_dict: dict):

        input_copy = input_dict.copy()

        input_copy.pop("cut")
        input_copy.pop("color")
        input_copy.pop("clarity")

        list_input = [
            input_copy[feature] for feature in DEFAULT_PARAMETERS_ORDER
        ]

        return np.asarray([list_input])

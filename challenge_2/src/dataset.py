from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PARAMETERS = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "price",
    "x",
    "y",
    "z",
]

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


def check_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Checks the given dataset for missing columns and adds them with default value 0 if necessary.

    Args:
        dataset (pd.DataFrame): The dataset to be checked.

    Raises:
        ValueError: If the 'price' column is not found in the dataset.

    Returns:
        pd.DataFrame: The dataset with missing columns added and default values set.
    """

    missing_columns = set(DEFAULT_PARAMETERS) - set(dataset.columns)

    if "price" not in dataset.columns:
        raise ValueError("Price column not found in dataset")

    for column in missing_columns:
        print(
            f"\033[33mWARNING:\033[0m Column {column} not found in dataset, adding it with default value 0"
        )
        dataset[column] = 0

    return dataset


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the given dataset by applying filters and transformations.

    Args:
        dataset (pd.DataFrame): The input dataset to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed dataset with the target variable separated.
    """

    filter_dataset = dataset[dataset["price"] > 0]

    filter_dataset = filter_dataset[
        (filter_dataset["x"] > 0)
        | (filter_dataset["y"] > 0)
        | (filter_dataset["z"] > 0)
    ]

    try:
        filter_dataset["z_depth"] = filter_dataset["depth"] / (
            filter_dataset["z"] * 100
        )
    except:
        filter_dataset["z_depth"] = 0

    try:
        filter_dataset["table_width"] = filter_dataset["table"] / (
            filter_dataset["x"] * 100
        )
    except:
        filter_dataset["table_width"] = 0

    cut_mapping = {
        "Fair": 1,
        "Good": 2,
        "Very Good": 3,
        "Premium": 4,
        "Ideal": 5,
    }
    filter_dataset["cut_category"] = filter_dataset["cut"].map(cut_mapping)

    color_grading_scale = list(map(chr, range(ord("D"), ord("Z") + 1)))[::-1]
    color_mapping = {k: v for v, k in enumerate(color_grading_scale, 1)}
    filter_dataset["color_category"] = filter_dataset["color"].map(
        color_mapping
    )

    clarity_grading_scale = [
        "FL",
        "IF",
        "VVS1",
        "VVS2",
        "VS1",
        "VS2",
        "SI1",
        "SI2",
        "I1",
        "I2",
        "I3",
    ][::-1]
    clarity_mapping = {k: v for v, k in enumerate(clarity_grading_scale, 1)}

    filter_dataset["clarity_category"] = filter_dataset["clarity"].map(
        clarity_mapping
    )

    final_dataset = filter_dataset.drop(columns=["cut", "color", "clarity"])

    X = final_dataset.drop(columns=["price"])
    X = X[DEFAULT_PARAMETERS_ORDER]

    y = final_dataset["price"]

    return X, y


def dataset_pipeline(dataset_path: Path):
    """
    Process the dataset located at the given path.

    Args:
        dataset_path (Path): The path to the dataset file.

    Returns:
        tuple: A tuple containing the preprocessed features (X) and the target variable (y).
    """

    dataset = pd.read_csv(dataset_path)

    final_dataset = check_dataset(dataset)

    X, y = preprocess_dataset(final_dataset)

    return X, y


def split_data(X, y, test_size=0.3, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

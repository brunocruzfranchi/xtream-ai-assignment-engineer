import argparse
from pathlib import Path

from src.dataset import dataset_pipeline, split_data
from src.model import training_pipeline


def main(args_in: list[str] | None = None) -> None:

    parser = argparse.ArgumentParser(
        description="Train diamond price prediction model - XGBoost"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="path where the dataset is stored",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="directory to export the trained model to; default: based on input",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="proportion of the dataset to include in the test split; default: 0.3",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to be used in multiple functions; default: 42",
    )

    args = parser.parse_args(args_in)

    if args.dataset is None:
        raise ValueError("Dataset path is required")

    if args.output is None:
        args.output = args.dataset.parent
        print(
            f"\033[92mINFO:\033[0m Output directory not provided, using default ({args.output})"
        )

    X, y = dataset_pipeline(args.dataset)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    training_pipeline(X_train, y_train, X_test, y_test, args.output)


if __name__ == "__main__":
    main()

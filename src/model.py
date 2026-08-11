import os
from typing import Tuple

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


from src.feature_engineering import FEATURE_NAMES


DEFAULT_DATASET_PATH = os.path.join(
    "data",
    "training_data.csv",
)

DEFAULT_MODEL_PATH = os.path.join(
    "models",
    "bfs_shortest_path_model.joblib",
)


def load_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:

    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Generate it first using:\n"
            "python -m src.dataset_generator"
        )

    dataframe = pd.read_csv(
        dataset_path
    )

    required_columns = (
        FEATURE_NAMES + ["label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    return dataframe


def prepare_data(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    X = dataframe[
        FEATURE_NAMES
    ].copy()

    y = dataframe[
        "label"
    ].astype(int)

    return X, y


def create_model() -> RandomForestClassifier:

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("=" * 60)
    print("                 MODEL EVALUATION")
    print("=" * 60)

    print()

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()

    print(
        "Classification Report:"
    )

    print(
        report
    )

    print(
        "Confusion Matrix:"
    )

    print(
        matrix
    )

    print("=" * 60)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_model(
    model: RandomForestClassifier,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:

    directory = os.path.dirname(
        model_path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    joblib.dump(
        model,
        model_path,
    )

    print()

    print(
        f"Model saved to: {model_path}"
    )


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
) -> RandomForestClassifier:

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    return model


def print_dataset_information(
    dataframe: pd.DataFrame,
) -> None:

    print()
    print("=" * 60)
    print("                 DATASET INFORMATION")
    print("=" * 60)

    print()

    print(
        f"Rows    : {len(dataframe)}"
    )

    print(
        f"Columns : {len(dataframe.columns)}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Features:"
    )

    for index, feature in enumerate(
        FEATURE_NAMES,
        start=1,
    ):

        print(
            f"  {index:2d}. {feature}"
        )

    print("=" * 60)


def train_and_save(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
) -> RandomForestClassifier:

    print("=" * 60)

    print(
        "          BFS MACHINE LEARNING TRAINING"
    )

    print("=" * 60)

    dataframe = load_dataset(
        dataset_path
    )

    print_dataset_information(
        dataframe
    )

    X, y = prepare_data(
        dataframe
    )

    if y.nunique() < 2:

        raise ValueError(
            "The dataset contains only one class. "
            "Both label 0 and label 1 are required "
            "for classification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print()

    print(
        "Training Random Forest..."
    )

    model = train_model(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        model_path,
    )

    return model


if __name__ == "__main__":

    train_and_save()

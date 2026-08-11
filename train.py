"""
train.py

End-to-end training pipeline:

    1. Generate a set of TRAINING trees.
    2. Generate a separate, disjoint set of TESTING trees.
    3. Turn each set into labeled feature rows.
    4. Train a RandomForestClassifier on the training rows.
    5. Evaluate on the testing rows (classification metrics).
    6. Save the trained model to disk with joblib.

Training trees and testing trees are generated independently (different
random seeds), so the model is evaluated on trees it has never seen
during training. This avoids the model simply memorizing specific trees.
"""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.dataset_generator import generate_trees, generate_dataset_from_trees
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import evaluate_classification, print_classification_report

DEFAULT_TRAIN_TREES = 240
DEFAULT_TEST_TREES = 60
DEFAULT_PAIRS_PER_TREE = 5
DEFAULT_MIN_NODES = 6
DEFAULT_MAX_NODES = 15
DEFAULT_MIN_WEIGHT = 1
DEFAULT_MAX_WEIGHT = 20

MODEL_PATH = "models/shortest_path_model.joblib"
DATASET_PATH = "data/training_data.csv"


def train_model(train_trees=DEFAULT_TRAIN_TREES, test_trees=DEFAULT_TEST_TREES,
                 pairs_per_tree=DEFAULT_PAIRS_PER_TREE, min_nodes=DEFAULT_MIN_NODES,
                 max_nodes=DEFAULT_MAX_NODES, min_weight=DEFAULT_MIN_WEIGHT,
                 max_weight=DEFAULT_MAX_WEIGHT, save_dataset_csv=True, verbose=True):
    """
    Run the full training pipeline and return
    (model, metrics, train_trees_list, test_trees_list, X_test, y_test).
    """
    if verbose:
        print("STEP 1: Generating training trees...")
    train_tree_list = generate_trees(
        train_trees, min_nodes, max_nodes, min_weight, max_weight, seed=42
    )
    if verbose:
        print("Completed.\n")

    if verbose:
        print("STEP 2: Generating testing trees (separate, unseen trees)...")
    test_tree_list = generate_trees(
        test_trees, min_nodes, max_nodes, min_weight, max_weight, seed=99999
    )
    if verbose:
        print("Completed.\n")

    if verbose:
        print("STEP 3: Generating training dataset...")
    train_df = generate_dataset_from_trees(train_tree_list, pairs_per_tree, seed=42)
    test_df = generate_dataset_from_trees(test_tree_list, pairs_per_tree, seed=99999)
    if verbose:
        print(f"Samples generated: {len(train_df) + len(test_df):,}\n")

    if save_dataset_csv:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        full_df = train_df.copy()
        full_df.to_csv(DATASET_PATH, index=False)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    if verbose:
        print("STEP 4: Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    if verbose:
        print("Training completed.\n")

    if verbose:
        print("STEP 5: Evaluating model...")
    metrics = evaluate_classification(model, X_test, y_test)
    if verbose:
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}%\n")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if verbose:
        print(f"Model saved to {MODEL_PATH}\n")

    if verbose:
        print_classification_report(
            metrics, train_trees, test_trees, len(X_train), len(X_test)
        )

    return model, metrics, train_tree_list, test_tree_list, X_test, y_test


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run training first (menu option 4 "
            f"or `python main.py --train`)."
        )
    return joblib.load(path)


if __name__ == "__main__":
    train_model()
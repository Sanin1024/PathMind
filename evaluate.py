"""
evaluate.py

Model evaluation utilities:
    - Classification-level metrics (accuracy, precision, recall, F1, confusion matrix)
    - Feature importance reporting + chart
    - Path-level evaluation (exact path match rate, destination success rate,
      average cost error) comparing ML predictions against Dijkstra ground truth
      on entirely unseen trees.
"""

import random

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.feature_engineering import FEATURE_COLUMNS
from src.dijkstra import dijkstra, path_cost
from src.predict import predict_shortest_path
from src.dataset_generator import generate_trees


def evaluate_classification(model, X_test, y_test):
    """Compute standard classification metrics on a held-out test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }
    return metrics


def print_classification_report(metrics, train_trees, test_trees, train_samples, test_samples):
    cm = metrics["confusion_matrix"]
    print("=" * 44)
    print("       MODEL EVALUATION")
    print("=" * 44)
    print()
    print(f"Training Trees: {train_trees}")
    print(f"Testing Trees : {test_trees}")
    print()
    print(f"Training Samples: {train_samples:,}")
    print(f"Testing Samples : {test_samples:,}")
    print()
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score : {metrics['f1'] * 100:.2f}%")
    print()
    print("Confusion Matrix:")
    print()
    print("                 Predicted")
    print("                 0       1")
    print(f"Actual 0        {cm[0][0]:<7} {cm[0][1]:<7}")
    print(f"Actual 1        {cm[1][0]:<7} {cm[1][1]:<7}")
    print()


def feature_importance_report(model):
    """Return a dict of feature_name -> importance (fraction, sums to 1.0)."""
    importances = model.feature_importances_
    return dict(sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True))


def print_feature_importance(importance_dict):
    print("=" * 44)
    print("          FEATURE IMPORTANCE")
    print("=" * 44)
    print()
    print(f"{'Feature':<32}{'Importance':>10}")
    print("-" * 50)
    for feature, importance in importance_dict.items():
        print(f"{feature:<32}{importance * 100:>9.2f}%")
    print()


def plot_feature_importance(importance_dict, output_path="outputs/feature_importance.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(importance_dict.keys())
    values = [v * 100 for v in importance_dict.values()]

    plt.figure(figsize=(9, 6))
    plt.barh(features[::-1], values[::-1], color="#3B82F6")
    plt.xlabel("Importance (%)")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def path_level_evaluation(model, num_test_trees=50, min_nodes=6, max_nodes=15,
                           min_weight=1, max_weight=20, pairs_per_tree=3, seed=2024):
    """
    Evaluate the model at the PATH level on entirely fresh, unseen trees
    (generated with a distinct seed from any training/classification data).

    Returns a dict of aggregate metrics.
    """
    rng = random.Random(seed)
    trees = generate_trees(
        num_test_trees, min_nodes, max_nodes, min_weight, max_weight, seed=seed
    )

    total_pairs = 0
    exact_matches = 0
    destination_successes = 0
    optimal_costs = []
    ml_costs = []
    cost_errors = []

    for tree in trees:
        node_ids = list(tree.keys())
        if len(node_ids) < 2:
            continue
        for _ in range(pairs_per_tree):
            source, destination = rng.sample(node_ids, 2)

            true_distance, true_path, _ = dijkstra(tree, source, destination)
            if true_path is None:
                continue

            result = predict_shortest_path(tree, source, destination, model)
            total_pairs += 1

            if result["success"]:
                destination_successes += 1
                ml_cost = result["cost"]
                ml_costs.append(ml_cost)
                optimal_costs.append(true_distance)
                cost_errors.append(abs(ml_cost - true_distance))

                if result["path"] == true_path:
                    exact_matches += 1

    metrics = {
        "test_trees": num_test_trees,
        "total_pairs": total_pairs,
        "exact_matches": exact_matches,
        "path_accuracy": (exact_matches / total_pairs) if total_pairs else 0.0,
        "destination_success_rate": (destination_successes / total_pairs) if total_pairs else 0.0,
        "average_optimal_cost": (sum(optimal_costs) / len(optimal_costs)) if optimal_costs else 0.0,
        "average_ml_cost": (sum(ml_costs) / len(ml_costs)) if ml_costs else 0.0,
        "average_cost_error": (sum(cost_errors) / len(cost_errors)) if cost_errors else 0.0,
    }
    return metrics


def print_path_level_evaluation(metrics):
    print("=" * 44)
    print("       PATH-LEVEL EVALUATION")
    print("=" * 44)
    print()
    print(f"Test Trees             : {metrics['test_trees']}")
    print()
    print(f"Exact Path Matches     : {metrics['exact_matches']} / {metrics['total_pairs']}")
    print(f"Path Accuracy          : {metrics['path_accuracy'] * 100:.2f}%")
    print()
    print(f"Destination Success    : {metrics['destination_success_rate'] * 100:.2f}%")
    print()
    print(f"Average Optimal Cost   : {metrics['average_optimal_cost']:.2f}")
    print(f"Average ML Cost        : {metrics['average_ml_cost']:.2f}")
    print()
    print(f"Average Cost Error     : {metrics['average_cost_error']:.2f}")
    print()
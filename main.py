"""
main.py

Interactive PowerShell entry point for the Weighted-Binary-Tree-ML project.

Run:
    python main.py                launch interactive menu
    python main.py --train        train the model
    python main.py --evaluate     evaluate the trained model
    python main.py --demo         run the full end-to-end demo
    python main.py --dijkstra     run Dijkstra on a sample tree
    python main.py --predict      run ML prediction on a sample tree
    python main.py --visualize    visualize a sample tree
"""

import argparse
import os
import sys

from src.tree_generator import (
    generate_weighted_binary_tree,
    validate_tree,
    render_tree_text,
    get_all_node_ids,
)
from src.dijkstra import dijkstra
from src.train import train_model, load_model, MODEL_PATH
from src.evaluate import (
    feature_importance_report,
    print_feature_importance,
    plot_feature_importance,
    path_level_evaluation,
    print_path_level_evaluation,
)
from src.predict import predict_shortest_path
from src.visualize import draw_tree


SAMPLE_TREE = {
    0: {"left": 1, "right": 2, "left_weight": 4, "right_weight": 2},
    1: {"left": 3, "right": 4, "left_weight": 3, "right_weight": 5},
    2: {"left": 5, "right": 6, "left_weight": 7, "right_weight": 1},
    3: {"left": None, "right": None, "left_weight": None, "right_weight": None},
    4: {"left": None, "right": None, "left_weight": None, "right_weight": None},
    5: {"left": None, "right": None, "left_weight": None, "right_weight": None},
    6: {"left": None, "right": None, "left_weight": None, "right_weight": None},
}


def print_header(title):
    print("=" * 44)
    print(title.center(44))
    print("=" * 44)


def print_menu():
    print()
    print_header("WEIGHTED BINARY TREE ML PROJECT")
    print()
    print("1. Create Manual Binary Tree")
    print("2. Generate Random Binary Tree")
    print("3. Run Dijkstra")
    print("4. Train ML Model")
    print("5. Evaluate ML Model")
    print("6. Predict Shortest Path Using ML")
    print("7. Compare Dijkstra vs ML")
    print("8. Visualize Tree")
    print("9. Run Complete Demo")
    print("10. Exit")
    print()


def read_int(prompt, allow_negative_one=False):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if not allow_negative_one and value < -1:
                print("Please enter a valid non-negative integer.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def create_manual_tree():
    print()
    print_header("ENTER WEIGHTED BINARY TREE")
    print()
    num_nodes = read_int("Enter number of nodes: ")

    tree = {}
    for node_id in range(num_nodes):
        tree[node_id] = {"left": None, "right": None, "left_weight": None, "right_weight": None}

    for node_id in range(num_nodes):
        print()
        print(f"Node {node_id}")
        left = read_int("Left child (-1 if none): ", allow_negative_one=True)
        left_weight = None
        if left != -1:
            left_weight = read_int("Left weight: ")
        right = read_int("Right child (-1 if none): ", allow_negative_one=True)
        right_weight = None
        if right != -1:
            right_weight = read_int("Right weight: ")

        tree[node_id]["left"] = None if left == -1 else left
        tree[node_id]["left_weight"] = left_weight
        tree[node_id]["right"] = None if right == -1 else right
        tree[node_id]["right_weight"] = right_weight

    try:
        validate_tree(tree)
    except ValueError as e:
        print(f"\nInvalid tree: {e}")
        return None

    print("\nTree created successfully.\n")
    print(render_tree_text(tree))
    return tree


def generate_random_tree_interactive():
    print()
    num_nodes = read_int("Number of nodes: ")
    min_weight = read_int("Minimum weight: ")
    max_weight = read_int("Maximum weight: ")
    seed_raw = input("Random seed (blank for random): ").strip()
    seed = int(seed_raw) if seed_raw else None

    tree = generate_weighted_binary_tree(num_nodes, min_weight, max_weight, seed)
    print("\nGenerated tree:\n")
    print(render_tree_text(tree))
    return tree


def run_dijkstra_interactive(tree):
    print()
    source = read_int("Enter source node: ")
    destination = read_int("Enter destination node: ")

    distance, path, _ = dijkstra(tree, source, destination)

    print()
    print_header("DIJKSTRA RESULT")
    print()
    print("Binary Tree:\n")
    print(render_tree_text(tree, dijkstra_path=path))
    print()
    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print()

    if path is None:
        print("No path found (destination unreachable).")
        return None, None, source, destination

    print("Shortest Path:\n")
    print(" -> ".join(str(n) for n in path))
    print()
    print("Edge Costs:\n")
    for i in range(len(path) - 1):
        # Recompute the individual edge weight for display.
        a, b = path[i], path[i + 1]
        info_a = tree[a]
        w = None
        if info_a["left"] == b:
            w = info_a["left_weight"]
        elif info_a["right"] == b:
            w = info_a["right_weight"]
        else:
            info_b = tree[b]
            if info_b["left"] == a:
                w = info_b["left_weight"]
            elif info_b["right"] == a:
                w = info_b["right_weight"]
        print(f"{a} -> {b} = {w}")
    print()
    print(f"Total Cost:\n\n{distance}")
    print()
    print("Dijkstra completed successfully.")
    return distance, path, source, destination


def train_interactive():
    print()
    train_model()


def evaluate_interactive():
    print()
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(str(e))
        return

    importance = feature_importance_report(model)
    print_feature_importance(importance)
    os.makedirs("outputs", exist_ok=True)
    chart_path = plot_feature_importance(importance)
    print(f"Feature importance chart saved to {chart_path}\n")

    metrics = path_level_evaluation(model)
    print_path_level_evaluation(metrics)


def predict_interactive(tree):
    print()
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(str(e))
        return None, None, None, None

    source = read_int("Enter source node: ")
    destination = read_int("Enter destination node: ")

    result = predict_shortest_path(tree, source, destination, model)

    print()
    print_header("ML PATH PREDICTION")
    print()
    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print()

    for step in result["steps"]:
        print(f"Current Node: {step['current_node']}\n")
        for i, cand in enumerate(step["candidates"], start=1):
            print(f"Candidate {i} ({cand['candidate_node']}):")
            print(f"Edge Weight: {cand['edge_weight']}")
            print(f"ML Probability: {cand['probability']:.2f}")
            print()
        print(f"ML Selected:\n{step['chosen']}\n")

    if result["success"]:
        print("Destination reached.\n")
        print_header("ML PREDICTED PATH")
        print()
        print(" -> ".join(str(n) for n in result["path"]))
        print()
        print(f"Predicted Cost: {result['cost']}")
    else:
        print(f"Prediction FAILED: {result['reason']}")

    return result, model, source, destination


def compare_interactive(tree):
    print()
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(str(e))
        return

    source = read_int("Enter source node: ")
    destination = read_int("Enter destination node: ")

    d_distance, d_path, _ = dijkstra(tree, source, destination)
    result = predict_shortest_path(tree, source, destination, model)

    print()
    print_header("DIJKSTRA vs ML COMPARISON")
    print()
    print(f"Source      : {source}")
    print(f"Destination : {destination}")
    print()
    print("-" * 44)
    print("DIJKSTRA")
    print("-" * 44)
    print()
    print("Path:")
    print(" -> ".join(str(n) for n in d_path) if d_path else "No path found")
    print()
    print(f"Cost:\n{d_distance}")
    print()
    print("-" * 44)
    print("MACHINE LEARNING")
    print("-" * 44)
    print()
    print("Path:")
    if result["success"]:
        print(" -> ".join(str(n) for n in result["path"]))
        print()
        print(f"Cost:\n{result['cost']}")
    else:
        print(f"FAILED ({result['reason']})")
    print()
    print("-" * 44)
    print("COMPARISON")
    print("-" * 44)
    print()

    if result["success"] and d_path is not None:
        exact_match = result["path"] == d_path
        cost_match = result["cost"] == d_distance
        cost_error = abs(result["cost"] - d_distance)
        cost_error_pct = (cost_error / d_distance * 100) if d_distance else 0.0

        print(f"Exact Path Match : {'YES' if exact_match else 'NO'}")
        print(f"Cost Match       : {'YES' if cost_match else 'NO'}")
        print()
        print(f"Dijkstra Cost    : {d_distance}")
        print(f"ML Cost          : {result['cost']}")
        print()
        print(f"Cost Error       : {cost_error}")
        print(f"Cost Error (%)   : {cost_error_pct:.2f}%")
    else:
        print("Exact Path Match : NO (ML failed to reach destination)")

    print()
    print("=" * 44)

    return d_path, d_distance, result


def visualize_interactive(tree, dijkstra_path=None, ml_path=None):
    print()
    os.makedirs("outputs", exist_ok=True)
    output_path = draw_tree(tree, dijkstra_path=dijkstra_path, ml_path=ml_path)
    print(f"Tree visualization saved to {output_path}")


def run_complete_demo():
    print()
    print_header("COMPLETE PROJECT DEMO")
    print()

    print("STEP 1: Generating training trees...")
    print("STEP 2: Generating training dataset...")
    print("STEP 3: Training Random Forest...")
    model, metrics, train_trees, test_trees, X_test, y_test = train_model(verbose=False)
    print("Completed.\n")

    print("STEP 4: Evaluating model...")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"F1 Score: {metrics['f1'] * 100:.2f}%\n")

    print("STEP 5: Generating unseen test tree...")
    demo_tree = generate_weighted_binary_tree(num_nodes=7, min_weight=1, max_weight=10, seed=7)
    node_ids = get_all_node_ids(demo_tree)
    source, destination = node_ids[0], node_ids[-1]
    print("Completed.\n")

    print("STEP 6: Running Dijkstra...")
    d_distance, d_path, _ = dijkstra(demo_tree, source, destination)
    print("Completed.\n")

    print("STEP 7: Running ML prediction...")
    result = predict_shortest_path(demo_tree, source, destination, model)
    print("Completed.\n")

    print("STEP 8: Comparing results...\n")
    print_header("FINAL RESULT")
    print()
    print("Dijkstra Path:")
    print(" -> ".join(str(n) for n in d_path) if d_path else "None")
    print()
    print(f"Dijkstra Cost:\n{d_distance}")
    print()
    print("ML Path:")
    if result["success"]:
        print(" -> ".join(str(n) for n in result["path"]))
        print()
        print(f"ML Cost:\n{result['cost']}")
        path_match = result["path"] == d_path
        cost_match = result["cost"] == d_distance
    else:
        print(f"FAILED ({result['reason']})")
        path_match = False
        cost_match = False
    print()
    print(f"Path Match:\n{'YES' if path_match else 'NO'}")
    print()
    print(f"Cost Match:\n{'YES' if cost_match else 'NO'}")
    print()
    print("=" * 44)
    print("DEMONSTRATION COMPLETE".center(44))
    print("=" * 44)

    os.makedirs("outputs", exist_ok=True)
    draw_tree(demo_tree, dijkstra_path=d_path,
              ml_path=result["path"] if result["success"] else None)


def interactive_menu():
    current_tree = None
    last_dijkstra_path = None
    last_ml_path = None

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            tree = create_manual_tree()
            if tree is not None:
                current_tree = tree
        elif choice == "2":
            current_tree = generate_random_tree_interactive()
        elif choice == "3":
            if current_tree is None:
                current_tree = SAMPLE_TREE
                print("\nNo tree loaded yet - using the built-in sample tree.")
            _, last_dijkstra_path, _, _ = run_dijkstra_interactive(current_tree)
        elif choice == "4":
            train_interactive()
        elif choice == "5":
            evaluate_interactive()
        elif choice == "6":
            if current_tree is None:
                current_tree = SAMPLE_TREE
                print("\nNo tree loaded yet - using the built-in sample tree.")
            result, _, _, _ = predict_interactive(current_tree)
            if result and result["success"]:
                last_ml_path = result["path"]
        elif choice == "7":
            if current_tree is None:
                current_tree = SAMPLE_TREE
                print("\nNo tree loaded yet - using the built-in sample tree.")
            d_path, _, result = compare_interactive(current_tree)
            last_dijkstra_path = d_path
            if result and result["success"]:
                last_ml_path = result["path"]
        elif choice == "8":
            if current_tree is None:
                current_tree = SAMPLE_TREE
                print("\nNo tree loaded yet - using the built-in sample tree.")
            visualize_interactive(current_tree, last_dijkstra_path, last_ml_path)
        elif choice == "9":
            run_complete_demo()
        elif choice == "10":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number from 1 to 10.")


def main():
    parser = argparse.ArgumentParser(description="Weighted Binary Tree ML Project")
    parser.add_argument("--train", action="store_true", help="Train the ML model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the ML model")
    parser.add_argument("--demo", action="store_true", help="Run the complete demo")
    parser.add_argument("--dijkstra", action="store_true", help="Run Dijkstra on the sample tree")
    parser.add_argument("--predict", action="store_true", help="Run ML prediction on the sample tree")
    parser.add_argument("--visualize", action="store_true", help="Visualize the sample tree")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_interactive()
    elif args.demo:
        run_complete_demo()
    elif args.dijkstra:
        run_dijkstra_interactive(SAMPLE_TREE)
    elif args.predict:
        predict_interactive(SAMPLE_TREE)
    elif args.visualize:
        _, path, _, _ = run_dijkstra_interactive(SAMPLE_TREE)
        visualize_interactive(SAMPLE_TREE, dijkstra_path=path)
    else:
        interactive_menu()


if __name__ == "__main__":
    sys.exit(main() or 0)
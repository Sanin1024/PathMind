import os
from typing import Dict, List, Optional, Set

import joblib
import pandas as pd

from src.bfs import bfs_shortest_path, format_path
from src.feature_engineering import (
    FEATURE_NAMES,
    extract_feature_vector,
)
from src.model import DEFAULT_MODEL_PATH


Graph = Dict[int, List[int]]


def load_trained_model(
    model_path: str = DEFAULT_MODEL_PATH,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train it first using:\n"
            "python -m src.model"
        )

    return joblib.load(model_path)


def read_integer(
    prompt: str,
    minimum: Optional[int] = None,
) -> int:

    while True:

        try:
            value = int(
                input(prompt).strip()
            )

            if minimum is not None and value < minimum:
                print(
                    f"Please enter a value >= {minimum}."
                )
                continue

            return value

        except ValueError:
            print(
                "Please enter a valid integer."
            )


def create_user_graph() -> Graph:

    print()
    print("=" * 60)
    print("                 GRAPH INPUT")
    print("=" * 60)

    print()

    num_nodes = read_integer(
        "Enter number of nodes: ",
        minimum=2,
    )

    graph: Graph = {
        node: []
        for node in range(num_nodes)
    }

    print()
    print(
        f"Nodes are numbered from 0 to {num_nodes - 1}."
    )

    print()
    print(
        "Enter the number of edges."
    )

    num_edges = read_integer(
        "Number of edges: ",
        minimum=0,
    )

    print()
    print(
        "Enter each edge as: node1 node2"
    )

    print(
        "Example: 0 1"
    )

    print()

    edges_added = 0

    while edges_added < num_edges:

        edge_text = input(
            f"Edge {edges_added + 1}/{num_edges}: "
        ).strip()

        parts = edge_text.split()

        if len(parts) != 2:

            print(
                "Enter exactly two node numbers."
            )

            continue

        try:
            node_a = int(parts[0])
            node_b = int(parts[1])

        except ValueError:

            print(
                "Both nodes must be integers."
            )

            continue

        if (
            node_a < 0
            or node_a >= num_nodes
            or node_b < 0
            or node_b >= num_nodes
        ):

            print(
                f"Nodes must be between "
                f"0 and {num_nodes - 1}."
            )

            continue

        if node_a == node_b:

            print(
                "Self-loops are not allowed."
            )

            continue

        if node_b in graph[node_a]:

            print(
                "That edge already exists."
            )

            continue

        # Undirected graph.
        graph[node_a].append(
            node_b
        )

        graph[node_b].append(
            node_a
        )

        edges_added += 1

    for node in graph:

        graph[node] = sorted(
            set(graph[node])
        )

    return graph


def read_source_destination(
    graph: Graph,
) -> tuple[int, int]:

    print()
    print("=" * 60)
    print("              SOURCE / DESTINATION")
    print("=" * 60)

    print()

    max_node = max(
        graph.keys()
    )

    while True:

        source = read_integer(
            "Enter source node: ",
            minimum=0,
        )

        if source not in graph:

            print(
                f"Node must be between 0 and {max_node}."
            )

            continue

        break

    while True:

        destination = read_integer(
            "Enter destination node: ",
            minimum=0,
        )

        if destination not in graph:

            print(
                f"Node must be between 0 and {max_node}."
            )

            continue

        break

    return source, destination


def predict_candidate(
    model,
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    candidate_node: int,
    visited: Optional[Set[int]] = None,
) -> int:

    features = extract_feature_vector(
        graph=graph,
        source=source,
        destination=destination,
        current_node=current_node,
        candidate_node=candidate_node,
        visited=visited,
    )

    dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(
        dataframe
    )

    return int(
        prediction[0]
    )


def predict_candidate_probability(
    model,
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    candidate_node: int,
    visited: Optional[Set[int]] = None,
) -> float:

    features = extract_feature_vector(
        graph=graph,
        source=source,
        destination=destination,
        current_node=current_node,
        candidate_node=candidate_node,
        visited=visited,
    )

    dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    probabilities = model.predict_proba(
        dataframe
    )

    classes = list(
        model.classes_
    )

    if 1 in classes:

        class_index = classes.index(
            1
        )

        return float(
            probabilities[0][class_index]
        )

    return 0.0


def choose_ml_next_node(
    model,
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    visited: Set[int],
) -> Optional[int]:

    candidates = [
        node
        for node in sorted(
            graph.get(
                current_node,
                [],
            )
        )
        if node not in visited
    ]

    if not candidates:
        return None

    predictions = []

    for candidate in candidates:

        prediction = predict_candidate(
            model=model,
            graph=graph,
            source=source,
            destination=destination,
            current_node=current_node,
            candidate_node=candidate,
            visited=visited,
        )

        probability = predict_candidate_probability(
            model=model,
            graph=graph,
            source=source,
            destination=destination,
            current_node=current_node,
            candidate_node=candidate,
            visited=visited,
        )

        predictions.append(
            (
                prediction,
                probability,
                candidate,
            )
        )

    positive_candidates = [
        item
        for item in predictions
        if item[0] == 1
    ]

    if positive_candidates:

        positive_candidates.sort(
            key=lambda item: (
                -item[1],
                item[2],
            )
        )

        return positive_candidates[0][2]

    predictions.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    return predictions[0][2]


def ml_shortest_path(
    model,
    graph: Graph,
    source: int,
    destination: int,
) -> Optional[List[int]]:

    if source == destination:
        return [source]

    current = source

    path = [source]

    visited: Set[int] = {
        source
    }

    max_steps = len(graph)

    for _ in range(max_steps):

        if current == destination:
            return path

        next_node = choose_ml_next_node(
            model=model,
            graph=graph,
            source=source,
            destination=destination,
            current_node=current,
            visited=visited,
        )

        if next_node is None:
            return None

        if next_node in visited:
            return None

        path.append(
            next_node
        )

        visited.add(
            next_node
        )

        current = next_node

    return None


def print_graph(
    graph: Graph,
) -> None:

    print()

    print(
        "Graph:"
    )

    for node in sorted(graph):

        print(
            f"  {node}: {graph[node]}"
        )


def compare_paths(
    bfs_path: Optional[List[int]],
    ml_path: Optional[List[int]],
) -> None:

    bfs_length = (
        len(bfs_path) - 1
        if bfs_path
        else None
    )

    ml_length = (
        len(ml_path) - 1
        if ml_path
        else None
    )

    same_path = (
        bfs_path == ml_path
    )

    same_destination = (
        bfs_path is not None
        and ml_path is not None
        and bfs_path[-1] == ml_path[-1]
    )

    print()
    print("=" * 60)
    print("                    RESULT")
    print("=" * 60)

    print()

    print(
        f"BFS shortest path : "
        f"{format_path(bfs_path)}"
    )

    print(
        f"ML predicted path : "
        f"{format_path(ml_path)}"
    )

    print()

    print(
        f"BFS path length   : {bfs_length}"
    )

    print(
        f"ML path length    : {ml_length}"
    )

    print(
        f"Same path         : {same_path}"
    )

    print(
        f"Reached destination: {same_destination}"
    )

    print()

    if same_path:

        print(
            "SUCCESS: ML reproduced the BFS shortest path."
        )

    elif same_destination:

        print(
            "PARTIAL SUCCESS: ML reached the destination "
            "but selected a different path."
        )

    else:

        print(
            "FAILED: ML did not reach the destination."
        )

    print()

    print("=" * 60)


def main() -> None:

    print()
    print("=" * 60)
    print("        BFS MACHINE LEARNING PREDICTOR")
    print("=" * 60)

    print()

    print(
        "The trained ML model will predict the route."
    )

    print(
        "BFS will also calculate the true shortest route"
    )

    print(
        "so that both results can be compared."
    )

    model = load_trained_model()

    graph = create_user_graph()

    source, destination = (
        read_source_destination(
            graph
        )
    )

    print_graph(
        graph
    )

    print()

    print(
        f"Source      : {source}"
    )

    print(
        f"Destination : {destination}"
    )

    print()

    print(
        "Calculating BFS shortest path..."
    )

    bfs_path = bfs_shortest_path(
        graph,
        source,
        destination,
    )

    if bfs_path is None:

        print()

        print(
            "No path exists between "
            f"{source} and {destination}."
        )

        return

    print(
        "BFS calculation completed."
    )

    print()

    print(
        "Calculating ML predicted path..."
    )

    ml_path = ml_shortest_path(
        model=model,
        graph=graph,
        source=source,
        destination=destination,
    )

    print(
        "ML prediction completed."
    )

    compare_paths(
        bfs_path,
        ml_path,
    )


if __name__ == "__main__":
    main()

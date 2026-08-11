import os
import random
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.bfs import bfs_shortest_path
from src.feature_engineering import (
    FEATURE_NAMES,
    create_training_row,
)


Graph = Dict[int, List[int]]


DEFAULT_OUTPUT_PATH = os.path.join(
    "data",
    "training_data.csv",
)


def generate_connected_graph(
    num_nodes: int,
    edge_probability: float = 0.25,
    seed: Optional[int] = None,
) -> Graph:

    if num_nodes < 2:
        raise ValueError(
            "num_nodes must be at least 2."
        )

    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError(
            "edge_probability must be between 0 and 1."
        )

    rng = random.Random(seed)

    graph: Graph = {
        node: []
        for node in range(num_nodes)
    }

    # First create a random spanning tree.
    # This guarantees that the graph is connected.
    for node in range(1, num_nodes):

        parent = rng.randint(
            0,
            node - 1,
        )

        graph[node].append(parent)

        graph[parent].append(node)

    # Add additional random edges.
    for node_a in range(num_nodes):

        for node_b in range(
            node_a + 1,
            num_nodes,
        ):

            if node_b in graph[node_a]:
                continue

            if rng.random() < edge_probability:

                graph[node_a].append(
                    node_b
                )

                graph[node_b].append(
                    node_a
                )

    for node in graph:

        graph[node] = sorted(
            set(graph[node])
        )

    return graph


def choose_source_destination(
    graph: Graph,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:

    nodes = list(
        graph.keys()
    )

    if len(nodes) < 2:
        return None

    for _ in range(100):

        source = rng.choice(nodes)

        destination = rng.choice(nodes)

        if source == destination:
            continue

        path = bfs_shortest_path(
            graph,
            source,
            destination,
        )

        if path is not None:
            return source, destination

    return None


def generate_training_rows(
    num_graphs: int = 100,
    nodes_per_graph: int = 20,
    edge_probability: float = 0.20,
    seed: Optional[int] = 42,
) -> List[Dict[str, float]]:

    if num_graphs <= 0:
        raise ValueError(
            "num_graphs must be greater than 0."
        )

    if nodes_per_graph < 2:
        raise ValueError(
            "nodes_per_graph must be at least 2."
        )

    rng = random.Random(seed)

    rows: List[Dict[str, float]] = []

    for graph_index in range(
        num_graphs
    ):

        graph_seed = rng.randint(
            0,
            1_000_000_000,
        )

        graph = generate_connected_graph(
            num_nodes=nodes_per_graph,
            edge_probability=edge_probability,
            seed=graph_seed,
        )

        pair = choose_source_destination(
            graph,
            rng,
        )

        if pair is None:
            continue

        source, destination = pair

        shortest_path = bfs_shortest_path(
            graph,
            source,
            destination,
        )

        if shortest_path is None:
            continue

        current_node = source

        visited: Set[int] = {
            source
        }

        # Generate training examples along the
        # actual BFS shortest path.
        for next_node in shortest_path[1:]:

            candidates = sorted(
                graph.get(
                    current_node,
                    [],
                )
            )

            for candidate_node in candidates:

                if candidate_node in visited:
                    continue

                row = create_training_row(
                    graph=graph,
                    source=source,
                    destination=destination,
                    current_node=current_node,
                    candidate_node=candidate_node,
                    visited=visited,
                )

                row["graph_id"] = float(
                    graph_index
                )

                rows.append(
                    row
                )

            visited.add(
                next_node
            )

            current_node = next_node

    return rows


def save_dataset(
    rows: List[Dict[str, float]],
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:

    if not rows:
        raise ValueError(
            "No training rows were generated."
        )

    dataframe = pd.DataFrame(
        rows
    )

    os.makedirs(
        os.path.dirname(output_path)
        or ".",
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe


def generate_dataset(
    num_graphs: int = 100,
    nodes_per_graph: int = 20,
    edge_probability: float = 0.20,
    seed: Optional[int] = 42,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:

    print(
        "Generating training graphs..."
    )

    rows = generate_training_rows(
        num_graphs=num_graphs,
        nodes_per_graph=nodes_per_graph,
        edge_probability=edge_probability,
        seed=seed,
    )

    print(
        f"Generated {len(rows)} training rows."
    )

    dataframe = save_dataset(
        rows,
        output_path,
    )

    print(
        f"Dataset saved to: {output_path}"
    )

    return dataframe


def print_dataset_summary(
    dataframe: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 60
    )
    print(
        "              DATASET SUMMARY"
    )
    print(
        "=" * 60
    )

    print()

    print(
        f"Rows       : {len(dataframe)}"
    )

    print(
        f"Columns    : {len(dataframe.columns)}"
    )

    if "label" in dataframe.columns:

        label_counts = (
            dataframe["label"]
            .value_counts()
            .sort_index()
        )

        print()

        print(
            "Labels:"
        )

        for label, count in label_counts.items():

            print(
                f"  {int(label)}: {count}"
            )

    print()

    print(
        "Columns:"
    )

    for column in dataframe.columns:

        print(
            f"  - {column}"
        )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "       BFS ML DATASET GENERATOR"
    )

    print(
        "=" * 60
    )

    dataframe = generate_dataset(
        num_graphs=100,
        nodes_per_graph=20,
        edge_probability=0.20,
        seed=42,
        output_path=DEFAULT_OUTPUT_PATH,
    )

    print_dataset_summary(
        dataframe
    )

    print()

    print(
        "First 5 rows:"
    )

    print(
        dataframe.head().to_string(
            index=False
        )
    )

    print()

    print(
        "Dataset generation completed successfully."
    )

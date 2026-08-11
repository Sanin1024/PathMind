from collections import deque
from typing import Dict, List, Optional, Set


Graph = Dict[int, List[int]]


FEATURE_NAMES = [
    "source",
    "destination",
    "current_node",
    "candidate_node",
    "source_candidate_distance",
    "candidate_destination_distance",
    "current_destination_distance",
    "current_degree",
    "candidate_degree",
    "destination_degree",
    "candidate_is_destination",
    "candidate_is_neighbor_of_destination",
    "candidate_is_visited",
    "candidate_is_source",
    "candidate_distance_from_current",
    "destination_reachable_from_candidate",
]


def node_degree(
    graph: Graph,
    node: int,
) -> int:

    return len(
        graph.get(node, [])
    )


def is_neighbor(
    graph: Graph,
    node_a: int,
    node_b: int,
) -> int:

    return int(
        node_b in graph.get(
            node_a,
            [],
        )
    )


def bfs_distances(
    graph: Graph,
    source: int,
) -> Dict[int, int]:

    if source not in graph:
        return {}

    queue = deque([source])

    distances: Dict[int, int] = {
        source: 0
    }

    while queue:

        current = queue.popleft()

        for neighbor in graph.get(
            current,
            [],
        ):

            if neighbor in distances:
                continue

            distances[neighbor] = (
                distances[current] + 1
            )

            queue.append(neighbor)

    return distances


def is_reachable(
    graph: Graph,
    source: int,
    destination: int,
) -> bool:

    distances = bfs_distances(
        graph,
        source,
    )

    return destination in distances


def extract_candidate_features(
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    candidate_node: int,
    visited: Optional[Set[int]] = None,
) -> Dict[str, float]:

    if visited is None:
        visited = set()

    source_distances = bfs_distances(
        graph,
        source,
    )

    candidate_distances = bfs_distances(
        graph,
        candidate_node,
    )

    current_distances = bfs_distances(
        graph,
        current_node,
    )

    source_candidate_distance = (
        source_distances.get(
            candidate_node,
            -1,
        )
    )

    candidate_destination_distance = (
        candidate_distances.get(
            destination,
            -1,
        )
    )

    current_destination_distance = (
        current_distances.get(
            destination,
            -1,
        )
    )

    features = {
        "source": float(source),

        "destination": float(destination),

        "current_node": float(current_node),

        "candidate_node": float(candidate_node),

        "source_candidate_distance": float(
            source_candidate_distance
        ),

        "candidate_destination_distance": float(
            candidate_destination_distance
        ),

        "current_destination_distance": float(
            current_destination_distance
        ),

        "current_degree": float(
            node_degree(
                graph,
                current_node,
            )
        ),

        "candidate_degree": float(
            node_degree(
                graph,
                candidate_node,
            )
        ),

        "destination_degree": float(
            node_degree(
                graph,
                destination,
            )
        ),

        "candidate_is_destination": float(
            candidate_node == destination
        ),

        "candidate_is_neighbor_of_destination": float(
            is_neighbor(
                graph,
                candidate_node,
                destination,
            )
        ),

        "candidate_is_visited": float(
            candidate_node in visited
        ),

        "candidate_is_source": float(
            candidate_node == source
        ),

        "candidate_distance_from_current": float(
            1
            if candidate_node in graph.get(
                current_node,
                [],
            )
            else -1
        ),

        "destination_reachable_from_candidate": float(
            destination in candidate_distances
        ),
    }

    return features


def features_to_vector(
    features: Dict[str, float],
) -> List[float]:

    return [
        float(features[name])
        for name in FEATURE_NAMES
    ]


def extract_feature_vector(
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    candidate_node: int,
    visited: Optional[Set[int]] = None,
) -> List[float]:

    features = extract_candidate_features(
        graph=graph,
        source=source,
        destination=destination,
        current_node=current_node,
        candidate_node=candidate_node,
        visited=visited,
    )

    return features_to_vector(
        features
    )


def generate_candidate_rows(
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    visited: Optional[Set[int]] = None,
) -> List[Dict[str, float]]:

    if visited is None:
        visited = set()

    rows = []

    for candidate in sorted(
        graph.get(
            current_node,
            [],
        )
    ):

        if candidate in visited:
            continue

        features = extract_candidate_features(
            graph=graph,
            source=source,
            destination=destination,
            current_node=current_node,
            candidate_node=candidate,
            visited=visited,
        )

        rows.append(
            features
        )

    return rows


def get_optimal_next_node(
    graph: Graph,
    current_node: int,
    destination: int,
    distances_to_destination: Optional[
        Dict[int, int]
    ] = None,
) -> Optional[int]:

    if current_node == destination:
        return destination

    if distances_to_destination is None:

        distances_to_destination = bfs_distances(
            graph,
            destination,
        )

    candidates = []

    for neighbor in sorted(
        graph.get(
            current_node,
            [],
        )
    ):

        distance = distances_to_destination.get(
            neighbor,
            float("inf"),
        )

        if distance != float("inf"):

            candidates.append(
                (
                    distance,
                    neighbor,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return candidates[0][1]


def label_candidate(
    graph: Graph,
    current_node: int,
    candidate_node: int,
    destination: int,
    distances_to_destination: Optional[
        Dict[int, int]
    ] = None,
) -> int:

    optimal_node = get_optimal_next_node(
        graph=graph,
        current_node=current_node,
        destination=destination,
        distances_to_destination=distances_to_destination,
    )

    return int(
        candidate_node == optimal_node
    )


def create_training_row(
    graph: Graph,
    source: int,
    destination: int,
    current_node: int,
    candidate_node: int,
    visited: Optional[Set[int]] = None,
) -> Dict[str, float]:

    if visited is None:
        visited = set()

    features = extract_candidate_features(
        graph=graph,
        source=source,
        destination=destination,
        current_node=current_node,
        candidate_node=candidate_node,
        visited=visited,
    )

    distances_to_destination = bfs_distances(
        graph,
        destination,
    )

    label = label_candidate(
        graph=graph,
        current_node=current_node,
        candidate_node=candidate_node,
        destination=destination,
        distances_to_destination=distances_to_destination,
    )

    features["label"] = float(
        label
    )

    return features


def get_feature_names() -> List[str]:

    return list(
        FEATURE_NAMES
    )


def get_feature_count() -> int:

    return len(
        FEATURE_NAMES
    )


if __name__ == "__main__":

    test_graph: Graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1, 5],
        5: [2, 4, 6],
        6: [5],
    }

    source = 0

    destination = 6

    current_node = 0

    candidate_node = 2

    visited = {
        source
    }

    print(
        "=" * 60
    )

    print(
        "        FEATURE ENGINEERING TEST"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Testing candidate:"
    )

    print(
        f"Source      : {source}"
    )

    print(
        f"Destination : {destination}"
    )

    print(
        f"Current     : {current_node}"
    )

    print(
        f"Candidate   : {candidate_node}"
    )

    print()

    features = extract_candidate_features(
        graph=test_graph,
        source=source,
        destination=destination,
        current_node=current_node,
        candidate_node=candidate_node,
        visited=visited,
    )

    print(
        "Features:"
    )

    for name in FEATURE_NAMES:

        print(
            f"  {name}: "
            f"{features[name]}"
        )

    print()

    print(
        "Feature count:"
    )

    print(
        get_feature_count()
    )

    print()

    print(
        "Feature vector:"
    )

    print(
        features_to_vector(
            features
        )
    )

    print()

    print(
        "Candidate label:"
    )

    print(
        label_candidate(
            graph=test_graph,
            current_node=current_node,
            candidate_node=candidate_node,
            destination=destination,
        )
    )

    print()

    print(
        "Feature engineering test completed successfully."
    )

    print(
        "=" * 60
    )

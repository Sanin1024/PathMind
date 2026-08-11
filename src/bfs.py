from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Graph = Dict[int, List[int]]


def bfs_traversal(
    graph: Graph,
    source: int,
) -> Tuple[List[int], Dict[int, Optional[int]], Dict[int, int]]:

    if source not in graph:
        raise ValueError(
            f"Source node {source} does not exist in the graph."
        )

    queue = deque([source])

    visited: Set[int] = {source}

    parent: Dict[int, Optional[int]] = {
        source: None
    }

    distance: Dict[int, int] = {
        source: 0
    }

    order: List[int] = []

    while queue:

        current = queue.popleft()

        order.append(current)

        for neighbor in sorted(
            graph.get(current, [])
        ):

            if neighbor in visited:
                continue

            visited.add(neighbor)

            parent[neighbor] = current

            distance[neighbor] = (
                distance[current] + 1
            )

            queue.append(neighbor)

    return order, parent, distance


def reconstruct_path(
    parent: Dict[int, Optional[int]],
    source: int,
    destination: int,
) -> Optional[List[int]]:

    if destination not in parent:
        return None

    path: List[int] = []

    current: Optional[int] = destination

    while current is not None:

        path.append(current)

        if current == source:
            break

        current = parent.get(current)

    if not path or path[-1] != source:
        return None

    path.reverse()

    return path


def bfs_shortest_path(
    graph: Graph,
    source: int,
    destination: int,
) -> Optional[List[int]]:

    if source not in graph:
        raise ValueError(
            f"Source node {source} does not exist in the graph."
        )

    if destination not in graph:
        raise ValueError(
            f"Destination node {destination} does not exist in the graph."
        )

    if source == destination:
        return [source]

    queue = deque([source])

    visited: Set[int] = {source}

    parent: Dict[int, Optional[int]] = {
        source: None
    }

    while queue:

        current = queue.popleft()

        if current == destination:
            break

        for neighbor in sorted(
            graph.get(current, [])
        ):

            if neighbor in visited:
                continue

            visited.add(neighbor)

            parent[neighbor] = current

            queue.append(neighbor)

    if destination not in parent:
        return None

    return reconstruct_path(
        parent,
        source,
        destination,
    )


def bfs_distance(
    graph: Graph,
    source: int,
    destination: int,
) -> Optional[int]:

    if source not in graph:
        raise ValueError(
            f"Source node {source} does not exist in the graph."
        )

    if destination not in graph:
        raise ValueError(
            f"Destination node {destination} does not exist in the graph."
        )

    if source == destination:
        return 0

    queue = deque([source])

    distance: Dict[int, int] = {
        source: 0
    }

    while queue:

        current = queue.popleft()

        for neighbor in sorted(
            graph.get(current, [])
        ):

            if neighbor in distance:
                continue

            distance[neighbor] = (
                distance[current] + 1
            )

            if neighbor == destination:
                return distance[neighbor]

            queue.append(neighbor)

    return None


def get_bfs_tree(
    graph: Graph,
    source: int,
) -> Dict[int, Optional[int]]:

    _, parent, _ = bfs_traversal(
        graph,
        source,
    )

    return parent


def is_valid_path(
    graph: Graph,
    path: Optional[List[int]],
) -> bool:

    if not path:
        return False

    for node in path:

        if node not in graph:
            return False

    for index in range(
        len(path) - 1
    ):

        current = path[index]

        next_node = path[index + 1]

        if next_node not in graph.get(
            current,
            [],
        ):
            return False

    return True


def path_length(
    path: Optional[List[int]],
) -> int:

    if not path:
        return 0

    return len(path) - 1


def format_path(
    path: Optional[List[int]],
) -> str:

    if not path:
        return "No path"

    return " -> ".join(
        str(node)
        for node in path
    )


def print_bfs_result(
    graph: Graph,
    source: int,
    destination: Optional[int] = None,
) -> None:

    order, parent, distance = bfs_traversal(
        graph,
        source,
    )

    print()
    print("=" * 50)
    print("                 BFS RESULT")
    print("=" * 50)

    print()
    print(f"Source node : {source}")

    print()
    print("BFS traversal:")
    print(format_path(order))

    print()
    print("Distances:")

    for node in sorted(distance):

        print(
            f"  {source} -> {node} = "
            f"{distance[node]} edge(s)"
        )

    if destination is not None:

        path = reconstruct_path(
            parent,
            source,
            destination,
        )

        print()
        print(f"Destination : {destination}")

        if path is None:

            print(
                "Shortest path: No path"
            )

            print(
                "Status       : Destination unreachable"
            )

        else:

            print(
                f"Shortest path: "
                f"{format_path(path)}"
            )

            print(
                f"Path length  : "
                f"{path_length(path)} edge(s)"
            )

    print("=" * 50)


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

    source_node = 0

    destination_node = 6

    print("Testing BFS...")

    print()
    print("Graph:")

    for node in sorted(test_graph):

        print(
            f"{node}: {test_graph[node]}"
        )

    order, parent, distance = bfs_traversal(
        test_graph,
        source_node,
    )

    print()
    print("BFS Traversal:")

    print(
        format_path(order)
    )

    print()
    print("Shortest Path:")

    path = bfs_shortest_path(
        test_graph,
        source_node,
        destination_node,
    )

    print(
        format_path(path)
    )

    print()
    print("Shortest Distance:")

    distance_value = bfs_distance(
        test_graph,
        source_node,
        destination_node,
    )

    print(
        distance_value
    )

    print()
    print("Path Valid:")

    print(
        is_valid_path(
            test_graph,
            path,
        )
    )

    print()
    print(
        "BFS test completed successfully."
    )

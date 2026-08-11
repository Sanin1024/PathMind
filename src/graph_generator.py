"""
Graph Generator for BFS Machine Learning Project.

Generates random unweighted graphs that can be used for:

* BFS shortest-path generation
* ML training-data generation
* BFS model testing

The graphs are represented as adjacency dictionaries:

{
0: [1, 2],
1: [0, 3],
2: [0, 4],
...
}
"""

import random
from typing import Dict, List, Optional, Tuple

Graph = Dict[int, List[int]]

def generate_random_graph(
num_nodes: int = 10,
edge_probability: float = 0.3,
seed: Optional[int] = None,
ensure_connected: bool = True,
) -> Graph:
"""
Generate a random undirected, unweighted graph.

```
Args:
    num_nodes:
        Number of nodes in the graph.

    edge_probability:
        Probability of creating an edge between two nodes.
        Must be between 0 and 1.

    seed:
        Optional random seed for reproducible graphs.

    ensure_connected:
        If True, first creates a spanning tree so that every
        node is reachable from every other node.

Returns:
    Graph represented as:
        {node: [neighbor1, neighbor2, ...]}
"""

if num_nodes < 2:
    raise ValueError("num_nodes must be at least 2.")

if not 0 <= edge_probability <= 1:
    raise ValueError("edge_probability must be between 0 and 1.")

rng = random.Random(seed)

graph: Graph = {
    node: []
    for node in range(num_nodes)
}

def add_edge(u: int, v: int) -> None:
    """Add an undirected edge."""
    if v not in graph[u]:
        graph[u].append(v)

    if u not in graph[v]:
        graph[v].append(u)

# ---------------------------------------------------------
# Step 1: Guarantee connectivity using a random spanning tree
# ---------------------------------------------------------
if ensure_connected:
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)

    for i in range(1, num_nodes):
        current = nodes[i]

        # Connect the current node to any previously
        # connected node.
        parent_index = rng.randint(0, i - 1)
        parent = nodes[parent_index]

        add_edge(current, parent)

# ---------------------------------------------------------
# Step 2: Add additional random edges
# ---------------------------------------------------------
for u in range(num_nodes):
    for v in range(u + 1, num_nodes):

        # Skip edges that already exist.
        if v in graph[u]:
            continue

        if rng.random() < edge_probability:
            add_edge(u, v)

# Keep neighbors ordered for reproducible processing.
for node in graph:
    graph[node].sort()

return graph
```

def generate_connected_graph(
num_nodes: int = 10,
edge_probability: float = 0.3,
seed: Optional[int] = None,
) -> Graph:
"""
Generate a guaranteed-connected random graph.
"""

```
return generate_random_graph(
    num_nodes=num_nodes,
    edge_probability=edge_probability,
    seed=seed,
    ensure_connected=True,
)
```

def generate_disconnected_graph(
num_nodes: int = 10,
edge_probability: float = 0.2,
seed: Optional[int] = None,
) -> Graph:
"""
Generate a random graph without forcing connectivity.

```
Some node pairs may therefore be unreachable.
"""

return generate_random_graph(
    num_nodes=num_nodes,
    edge_probability=edge_probability,
    seed=seed,
    ensure_connected=False,
)
```

def add_edge(graph: Graph, u: int, v: int) -> None:
"""
Add an undirected edge to an existing graph.
"""

```
if u not in graph:
    graph[u] = []

if v not in graph:
    graph[v] = []

if v not in graph[u]:
    graph[u].append(v)

if u not in graph[v]:
    graph[v].append(u)

graph[u].sort()
graph[v].sort()
```

def remove_edge(graph: Graph, u: int, v: int) -> None:
"""
Remove an undirected edge from an existing graph.
"""

```
if u in graph and v in graph[u]:
    graph[u].remove(v)

if v in graph and u in graph[v]:
    graph[v].remove(u)
```

def get_edges(graph: Graph) -> List[Tuple[int, int]]:
"""
Return all undirected edges without duplicates.

```
Example:
    [(0, 1), (0, 2), (1, 3)]
"""

edges: List[Tuple[int, int]] = []

for u in sorted(graph):
    for v in graph[u]:
        if u < v:
            edges.append((u, v))

return edges
```

def get_num_edges(graph: Graph) -> int:
"""
Return the number of undirected edges.
"""

```
return len(get_edges(graph))
```

def get_neighbors(graph: Graph, node: int) -> List[int]:
"""
Return the neighbors of a node.
"""

```
if node not in graph:
    raise ValueError(f"Node {node} does not exist in the graph.")

return list(graph[node])
```

def graph_to_string(graph: Graph) -> str:
"""
Convert a graph into a readable adjacency-list string.
"""

```
lines = []

for node in sorted(graph):
    neighbors = ", ".join(
        str(neighbor)
        for neighbor in sorted(graph[node])
    )

    lines.append(f"{node}: [{neighbors}]")

return "\n".join(lines)
```

def print_graph(graph: Graph) -> None:
"""
Print the graph in adjacency-list format.
"""

```
print("\nGraph:")
print("-" * 40)
print(graph_to_string(graph))
print("-" * 40)
```

if **name** == "**main**":
# Simple standalone test.
graph = generate_connected_graph(
num_nodes=10,
edge_probability=0.25,
seed=42,
)

```
print_graph(graph)

print(f"Nodes : {len(graph)}")
print(f"Edges : {get_num_edges(graph)}")
```

"""
predict.py

Uses the trained Random Forest model to greedily predict a shortest path
between a source and destination node in a (possibly unseen) weighted
binary tree.

At every step, the model looks at the current node's candidate neighbors
(its tree children and, if available, its tree parent) and predicts the
probability that each candidate is the correct next step. The candidate
with the highest predicted probability is chosen, and the walk repeats
until the destination is reached, a dead end is hit, or a maximum number
of steps is exceeded (to guard against infinite loops).
"""

import pandas as pd

from src.dijkstra import build_adjacency
from src.feature_engineering import (
    build_feature_row,
    compute_depths,
    compute_subtree_sizes,
    FEATURE_COLUMNS,
)


class PredictionFailure(Exception):
    """Raised when the ML model fails to find a valid path to the destination."""
    pass


def predict_shortest_path(tree, source, destination, model, verbose=False):
    """
    Predict a path from `source` to `destination` using the trained model.

    Parameters
    ----------
    tree : dict
        The binary tree.
    source : int
        Starting node id.
    destination : int
        Target node id.
    model : sklearn estimator
        A trained classifier exposing predict_proba().
    verbose : bool
        If True, return per-step candidate probabilities for display.

    Returns
    -------
    dict with keys:
        "path": list of node ids (or None on failure)
        "cost": total path cost (or None on failure)
        "success": bool
        "steps": list of per-step debug info (only meaningful if verbose)
        "reason": failure reason string (if any)
    """
    if source not in tree:
        raise ValueError(f"Source node {source} does not exist in the tree")
    if destination not in tree:
        raise ValueError(f"Destination node {destination} does not exist in the tree")

    adjacency = build_adjacency(tree)
    depths = compute_depths(tree)
    subtree_sizes = compute_subtree_sizes(tree)

    max_steps = max(len(tree) * 2, 10)

    path = [source]
    visited = set([source])
    current_node = source
    previous_node = None
    current_path_cost = 0
    steps_info = []

    if source == destination:
        return {
            "path": [source],
            "cost": 0,
            "success": True,
            "steps": steps_info,
            "reason": None,
        }

    for _ in range(max_steps):
        candidates = [
            (nid, w) for nid, w in adjacency[current_node]
            if nid != previous_node
        ]

        if not candidates:
            return {
                "path": path,
                "cost": None,
                "success": False,
                "steps": steps_info,
                "reason": f"No available candidates from node {current_node}",
            }

        rows = []
        for candidate_node, edge_weight in candidates:
            row = build_feature_row(
                current_node=current_node,
                candidate_node=candidate_node,
                edge_weight=edge_weight,
                current_path_cost=current_path_cost,
                tree=tree,
                depths=depths,
                subtree_sizes=subtree_sizes,
            )
            rows.append(row)

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probabilities = model.predict_proba(X)[:, 1]

        step_record = {
            "current_node": current_node,
            "candidates": [
                {"candidate_node": c[0], "edge_weight": c[1], "probability": float(p)}
                for c, p in zip(candidates, probabilities)
            ],
        }

        best_index = int(probabilities.argmax())
        chosen_node, chosen_weight = candidates[best_index]
        step_record["chosen"] = chosen_node
        steps_info.append(step_record)

        if chosen_node in visited:
            return {
                "path": path,
                "cost": None,
                "success": False,
                "steps": steps_info,
                "reason": f"Model re-selected already-visited node {chosen_node} (possible loop)",
            }

        path.append(chosen_node)
        visited.add(chosen_node)
        current_path_cost += chosen_weight
        previous_node = current_node
        current_node = chosen_node

        if current_node == destination:
            return {
                "path": path,
                "cost": current_path_cost,
                "success": True,
                "steps": steps_info,
                "reason": None,
            }

    return {
        "path": path,
        "cost": None,
        "success": False,
        "steps": steps_info,
        "reason": "Maximum number of steps exceeded without reaching destination",
    }
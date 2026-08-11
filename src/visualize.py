"""
visualize.py

Draws the binary tree using matplotlib, showing node ids, edge weights,
and (optionally) highlighting the Dijkstra shortest path and the ML
predicted path with different line styles so they remain distinguishable
even without color.
"""

from src.feature_engineering import compute_depths


def _compute_positions(tree, root=0):
    """
    Compute (x, y) coordinates for each node using a simple recursive
    in-order-style layout so the tree renders without overlapping nodes.
    """
    depths = compute_depths(tree, root)
    positions = {}
    next_x = [0]

    def assign(node_id):
        info = tree[node_id]
        if info["left"] is not None:
            assign(info["left"])
        x = next_x[0]
        next_x[0] += 1
        positions[node_id] = (x, -depths[node_id])
        if info["right"] is not None:
            assign(info["right"])

    assign(root)
    return positions


def _path_edges(path):
    if not path:
        return set()
    return set(zip(path[:-1], path[1:])) | set(zip(path[1:], path[:-1]))


def draw_tree(tree, dijkstra_path=None, ml_path=None, root=0,
              output_path="outputs/tree_visualization.png", title="Weighted Binary Tree"):
    """
    Render the tree to a PNG file at `output_path`.

    dijkstra_path : list of node ids or None
    ml_path       : list of node ids or None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = _compute_positions(tree, root)
    dijkstra_edges = _path_edges(dijkstra_path)
    ml_edges = _path_edges(ml_path)

    fig, ax = plt.subplots(figsize=(max(8, len(tree)), 6))

    # Draw all edges first (default style).
    for node_id, info in tree.items():
        x1, y1 = positions[node_id]
        for child_key, weight_key in (("left", "left_weight"), ("right", "right_weight")):
            child = info[child_key]
            if child is None:
                continue
            x2, y2 = positions[child]
            weight = info[weight_key]

            is_dijkstra = (node_id, child) in dijkstra_edges
            is_ml = (node_id, child) in ml_edges

            if is_dijkstra and is_ml:
                ax.plot([x1, x2], [y1, y2], color="#8B5CF6", linewidth=3.2,
                        linestyle="-", zorder=3, label="_nolegend_")
            elif is_dijkstra:
                ax.plot([x1, x2], [y1, y2], color="#2563EB", linewidth=3,
                        linestyle="-", zorder=2, label="_nolegend_")
            elif is_ml:
                ax.plot([x1, x2], [y1, y2], color="#DC2626", linewidth=2.4,
                        linestyle="--", zorder=2, label="_nolegend_")
            else:
                ax.plot([x1, x2], [y1, y2], color="#B0B0B0", linewidth=1,
                        linestyle=":", zorder=1, label="_nolegend_")

            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(weight), fontsize=9, color="#333333",
                    backgroundcolor="white", ha="center", va="center", zorder=4)

    # Draw nodes on top of edges.
    for node_id, (x, y) in positions.items():
        ax.scatter([x], [y], s=650, color="#1F2937", zorder=5)
        ax.text(x, y, str(node_id), fontsize=10, color="white",
                ha="center", va="center", zorder=6, fontweight="bold")

    # Legend (manual, since we suppressed auto-labels above).
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#B0B0B0", lw=1, linestyle=":", label="Tree edge"),
        Line2D([0], [0], color="#2563EB", lw=3, linestyle="-", label="Dijkstra path"),
        Line2D([0], [0], color="#DC2626", lw=2.4, linestyle="--", label="ML predicted path"),
        Line2D([0], [0], color="#8B5CF6", lw=3.2, linestyle="-", label="Both (match)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
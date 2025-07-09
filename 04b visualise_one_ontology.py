import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load JSON
with open("ontology_output/gcq_ontology_topdown_refined.json", encoding="utf-8") as f:
    ontology_data = json.load(f)
cluster_id = "Cluster 8"
cluster_ontology = ontology_data.get(cluster_id, {}).get("refined_ontology", {})


def draw_arrows_with_node_border(ax, G, pos, node_radius=0.045, edge_color="lightgreen", arrowsize=20):

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Vector from source to target
        vec = np.array([x2 - x1, y2 - y1])
        dist = np.linalg.norm(vec)
        if dist == 0:
            continue

        # Normalize vector
        vec_norm = vec / dist

        # Shorten line so arrows start/end at node edge, not center
        start = np.array([x1, y1]) + vec_norm * node_radius
        end = np.array([x2, y2]) - vec_norm * node_radius

        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle='-|>',
            mutation_scale=arrowsize,
            color=edge_color,
            linewidth=2,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
        ax.add_patch(arrow)

# Build directed graph
G = nx.DiGraph()

# Add nodes
for node in cluster_ontology.get("classes", []):
    G.add_node(node)

# Add directed edges and labels
edge_labels = {}
for subj, pred, obj in cluster_ontology.get("relationships", []):
    G.add_edge(subj, obj)
    edge_labels[(subj, obj)] = pred

pos = nx.spring_layout(G, seed=42, k=3, iterations=100)

# Plotting
plt.figure(figsize=(15, 10))
ax = plt.gca()

nx.draw_networkx_nodes(G, pos, node_color="#e754b1", node_size=1500, edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=12, font_color="Black")
#nx.draw_networkx_edges(G, pos, edge_color="lightgreen", arrows=True, arrowsize=30)
draw_arrows_with_node_border(ax, G, pos, arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="gray")

x_values, y_values = zip(*pos.values())

# Compute limits with padding
x_margin = (max(x_values) - min(x_values)) * 0.1  # 10% margin
y_margin = (max(y_values) - min(y_values)) * 0.1

plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)

# Remove axis
plt.axis("off")
plt.tight_layout()
plt.title(f"Ontology Graph {cluster_id}", fontsize=14)
plt.show()

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load JSON
with open("ontology_output/gcq_ontology_topdown_refined_with_keywords.json", encoding="utf-8") as f:
    ontology_data = json.load(f)

# Select cluster
cluster_id = "Cluster 8"
cluster_ontology = ontology_data.get(cluster_id, {}).get("refined_ontology", {})

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

pos = nx.spring_layout(G, seed=42)  # seed for reproducibility

# Plotting
plt.figure(figsize=(14, 10))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color="#e754b1", node_size=1500, edgecolors='black')

# Draw labels on nodes
nx.draw_networkx_labels(G, pos, font_size=10, font_color="Black")

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color="lightgreen", arrows=True, arrowsize=20)

# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="gray")

# Remove axis
plt.axis("off")
plt.tight_layout()
plt.title(f"Ontology Graph {cluster_id}", fontsize=14)
plt.show()

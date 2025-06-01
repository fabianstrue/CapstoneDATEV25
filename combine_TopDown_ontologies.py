from pyvis.network import Network
import json

# Load merged ontology
with open("merged_TopDown_ontology.json", "r", encoding="utf-8") as f:
    ontology = json.load(f)

# IMPORTANT: Set notebook=False to avoid the missing template error
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)

# Add classes as nodes
for cls in ontology["classes"]:
    net.add_node(str(cls), label=str(cls), shape="ellipse", color="lightblue")

# Add edges (relationships)
for rel in ontology["relationships"]:
    if len(rel) != 3:
        continue  # Skip invalid entries
    subj, pred, obj = map(str, rel)
    if subj not in net.node_ids:
        net.add_node(subj, label=subj)
    if obj not in net.node_ids:
        net.add_node(obj, label=obj)
    net.add_edge(subj, obj, label=pred)

# Save HTML file (notebook=False avoids Jinja template issue)
net.show("ontology_graph.html")

print("✅ Graph created: Open 'ontology_graph.html' in your browser.")

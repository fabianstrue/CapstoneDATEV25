#Zur Info: Für Streamlit muss es ein reiner Pythion file (.py) sein, daher kein Notebook (.ipynb)

import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

#Example Questions (fixed with Cluster 0 and Cluster 10 for testing)
cq_data = [
    {
        "cq": "Welche Rechte hat ein Kabelnetzbetreiber zur Kontrolle der Signaldurchleitung?",
        "cluster": "0",
    },
    {
        "cq": "Wie kann ein Eigentümer seine Rechte bei Beeinträchtigung durch Dritte durchsetzen?",
        "cluster": "10",
    },
]

#Load Ontology Data
with open("ontology_output/gcq_ontology_topdown_refined_small.json") as f:
    ontology_data = json.load(f)

#Once Again Embeddings for Competency Questions
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
cq_embeddings = model.encode([item["cq"] for item in cq_data], convert_to_tensor=True)

#Dashboard UI Setup
st.title("📚 DATEV Legal Assistant Dashboard")
query = st.text_input("🔍 Type in your legal question:")

#Interactive Ontology-Visualisation Attempt
def show_ontology_graph(ontology):
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")

    #Knods (Klassen)
    classes = ontology.get("refined_ontology", {}).get("classes", [])
    for node in classes:
        net.add_node(node, label=node, color="#1f78b4")

    #Edges (Relationen)
    relations = ontology.get("refined_ontology", {}).get("relationships", [])
    for subj, pred, obj in relations:
        net.add_edge(subj, obj, label=pred)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_path = tmp_file.name

    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=550, scrolling=True)

#Find Similar Competency Questions
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, cq_embeddings)[0]
    top_k = min(5, len(cq_data))
    top_indices = np.argsort(-cosine_scores.cpu())[:top_k]

    st.subheader("✨ Similar Competency Questions:")
    for idx in top_indices:
        result = cq_data[int(idx)]
        cq_text = result["cq"]
        cluster = result["cluster"]
        label_key = f"Cluster {cluster}"

        if st.button(f"➞ {cq_text}", key=f"btn_{idx}"):
            st.markdown(f"**Selected Question:** {cq_text}")
            st.markdown(f"_Cluster: {label_key}_")

            if label_key in ontology_data:
                st.subheader("📘 Related Sub-Ontology:")
                ontology = ontology_data[label_key]
                show_ontology_graph(ontology)
            else:
                st.warning("No Ontology Found For This Label.")

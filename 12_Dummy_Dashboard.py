import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# === Beispiel-Daten (angepasst an Cluster 0 und Cluster 10) ===
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

# === Lade Ontologie-Daten ===
with open("ontology_output/gcq_ontology_topdown_refined_small.json") as f:
    ontology_data = json.load(f)

# === Embedding vorbereiten ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
cq_embeddings = model.encode([item["cq"] for item in cq_data], convert_to_tensor=True)

# === Dashboard UI ===
st.title("📚 DATEV Legal Assistant Dashboard")
query = st.text_input("🔍 Gib eine juristische Frage ein:")

# === Interaktive Ontologie-Visualisierung ===
def show_ontology_graph(ontology):
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")

    # Knoten (Klassen)
    classes = ontology.get("refined_ontology", {}).get("classes", [])
    for node in classes:
        net.add_node(node, label=node, color="#1f78b4")

    # Kanten (Relationen)
    relations = ontology.get("refined_ontology", {}).get("relationships", [])
    for subj, pred, obj in relations:
        net.add_edge(subj, obj, label=pred)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_path = tmp_file.name

    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=550, scrolling=True)

# === Ähnliche Fragen finden ===
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, cq_embeddings)[0]
    top_k = min(5, len(cq_data))
    top_indices = np.argsort(-cosine_scores.cpu())[:top_k]

    st.subheader("✨ Ähnliche Competency Questions:")
    for idx in top_indices:
        result = cq_data[int(idx)]
        cq_text = result["cq"]
        cluster = result["cluster"]
        label_key = f"Cluster {cluster}"

        if st.button(f"➞ {cq_text}", key=f"btn_{idx}"):
            st.markdown(f"**Ausgewählte Frage:** {cq_text}")
            st.markdown(f"_Cluster: {label_key}_")

            if label_key in ontology_data:
                st.subheader("📘 Zugehörige Sub-Ontologie:")
                ontology = ontology_data[label_key]
                show_ontology_graph(ontology)
            else:
                st.warning("Keine Ontologie für dieses Label gefunden.")

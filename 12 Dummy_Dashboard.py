import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

# Beispiel-CQs mit zugehörigen Clustern
cq_data = [
    {"cq": "Was ist ein Vertrag?", "cluster": "Cluster 10", "label": "Vertragsrecht"},
    {"cq": "Wann verjährt ein Anspruch?", "cluster": "Cluster 41", "label": "Versicherungsrecht"},
]

# Lade Ontologie-Datei
with open("ontology_output/gcq_ontology_topdown_refined_small.json", "r", encoding="utf-8") as f:
    ontology_data = json.load(f)

# CQ-Embeddings vorbereiten
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
cq_embeddings = model.encode([item["cq"] for item in cq_data], convert_to_tensor=True)

# Streamlit UI
st.title("📚 DATEV Legal Assistant Dashboard")
query = st.text_input("🔍 Gib eine juristische Frage ein:")

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
        label = result["label"]

        if st.button(f"➡ {cq_text}"):
            st.markdown(f"**Ausgewählte Frage:** {cq_text}")
            st.markdown(f"_Cluster: {cluster}, Thema: {label}_")

            # Zeige passende Ontologie
            if cluster in ontology_data:
                st.subheader("📘 Zugehörige Sub-Ontologie:")
                st.json(ontology_data[cluster])
            else:
                st.warning("Keine Ontologie für dieses Cluster gefunden.")

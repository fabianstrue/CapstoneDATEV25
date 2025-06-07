import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load Clustered CQs & Labels
with open("dashboard_data/clustered_CQs_hdbscan.json", "r", encoding="utf-8") as f:
    clustered_cqs = json.load(f)

with open("dashboard_data/cluster_keywords.json", "r", encoding="utf-8") as f:
    cluster_labels = json.load(f)

# Create list of all CQs with cluster info
cq_data = []
for cluster_id, cqs in clustered_cqs.items():
    label = ", ".join(cluster_labels.get(cluster_id, []))
    for cq in cqs:
        cq_data.append({
            "cluster": cluster_id,
            "cq": cq,
            "label": label
        })

# Load embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
cq_embeddings = model.encode([item["cq"] for item in cq_data], convert_to_tensor=True)

# Dashboard UI
st.title("📘 DATEV Legal Assistant Dashboard")
query = st.text_input("🔍 Gib eine juristische Frage ein:")

# Initialisierung, damit kein Fehler entsteht, wenn query leer ist
top_indices = []

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, cq_embeddings)[0]
    top_k = min(5, len(cq_data))
    top_indices = np.argsort(-cosine_scores.cpu())[:top_k]

    st.subheader("📌 Ähnliche Competency Questions:")
    for idx in top_indices:
        result = cq_data[int(idx)]
        st.markdown(f"- **{result['cq']}**  \n _(Cluster: {result['cluster']} – Thema: {result['label']})_")
import re
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util

#Load our clustered CQs
with open("competency_questions_output/clustered_CQs_hdbscan.json", "r", encoding="utf-8") as f:
    clustered_cqs = json.load(f)

#Flatten CQ list with cluster info
flat_cqs = []
for cluster_id, questions in clustered_cqs.items():
    for q in questions:
        flat_cqs.append((q, cluster_id))

#Sentence Embedding Model for Finding Similar Questions 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
corpus = [q for q, _ in flat_cqs]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

#Streamlit Dashboard UI Setuo
st.set_page_config(page_title="Legal CQ Finder", layout="wide")
st.title("🔍 Juristische Frage & Ontologie-Cluster")

user_input = st.text_input("📌 Ihre Frage eingeben:")

if user_input:
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu().numpy()

    top_k = min(5, len(corpus))
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    sorted_results = top_results[np.argsort(-cos_scores[top_results])]

    st.subheader("📄 Ähnlichste Kompetenzfragen:")
    for idx in sorted_results:
        cq, cluster = flat_cqs[idx]
        st.markdown(f"**Frage:** {cq}")
        st.markdown(f"🗂️ **Cluster:** {cluster}")
        st.markdown("---")

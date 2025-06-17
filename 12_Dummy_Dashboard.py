#streamlit run 12_Dummy_Dashboard.py --server.fileWatcherType none

import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# Page configuration
st.set_page_config(
    page_title="Legal Assistant Dashboard",
    layout="wide"
)

# --- COMPANY INFO PANEL IN SIDEBAR ---
# Sidebar fixed full height with company info, description, feature list and footer
st.sidebar.markdown("""
<div style="
    position: fixed;
    top: 0;
    left: 0;
    width: 280px;
    height: 100vh;
    background-color: #EFEFEF;
    padding: 1rem;
    overflow: auto;
    box-sizing: border-box;
">
  <h3 style="margin: 0; color: #222222;">Your Company Name</h3>
  <p style="font-size: 0.9rem; color: #444444; margin-top: 0.5rem;">
    We are an advanced research tool for legal professionals, meticulously engineered to streamline case analysis and make every aspect of legal research faster, more precise, and deeply efficient.
  </p>
  <div style="margin-top: 1rem;">
    <ul style="list-style-type: none; padding: 0; margin: 0;">
      <li style="margin-bottom: 0.5rem;"><strong>Trained on 6000+ documents</strong></li>
      <li style="margin-bottom: 0.5rem;"><strong>AI powered</strong></li>
      <li style="margin-bottom: 1rem;"><strong>Developed by AI professionals and lawyers</strong></li>
    </ul>
  </div>
  <div style="
      position: absolute;
      bottom: 1rem;
      left: 1rem;
      font-size: 0.8rem;
      color: #444444;
  ">
    for more information contact info@datavation.de
  </div>
</div>
""", unsafe_allow_html=True)

# Global CSS for clean, light modern design
st.markdown("""
<style>
  /* App background and container */
  .reportview-container, .stApp {
    background-color: #FFFFFF;
    color: #333333;
    padding: 2rem;
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  /* Remove default top stripe */
  .stApp > header {
    border: none;
    background-color: #FFFFFF;
  }
  /* Sidebar style override to ensure contrast */
  .stSidebar {
    background-color: #FFFFFF;
    padding: 0;
  }
  /* Title style */
  h1, h2 {
    margin-bottom: 0.5rem;
    color: #222222;
    text-align: left !important;
  }
  /* Text input: white background, green border */
  .stTextInput>div>div>input {
    background-color: #FFFFFF;
    border: 2px solid #70BA30 !important;
    border-radius: 6px;
    padding: 0.5rem;
    color: #333333 !important;
  }
  /* Question buttons: left-aligned, full-width, white background with pink border */
  .stButton>button {
    background-color: #FFFFFF;
    color: #E862A1;
    padding: 0.6rem 1.2rem;
    border-radius: 6px;
    border: 2px solid #E862A1;
    font-size: 0.9rem;
    text-align: left !important;
    width: 100% !important;
  }
  .stButton>button:hover {
    background-color: #FDE8EE;
  }
  /* Remove default footer */
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Load ontology data
with open("ontology_output/gcq_ontology_topdown_refined_with_keywords.json", encoding="utf-8") as f:
    ontology_data = json.load(f)

# Example competency questions
cq_data = [
    {"cq": "Unter welchen Voraussetzungen kann Lärm eine Mietminderung rechtfertigen?", "cluster": "8"},
    {"cq": "Welche Rolle spielt § 536 Abs. 1 BGB bei der Beurteilung von Lärm als Mangel der Mietsache?", "cluster": "8"},
    {"cq": "Welche Bedeutung hat die Art des Lärms für die Entscheidung, ob eine Gefahrenlage vorliegt, die Schutzmaßnahmen erfordert?", "cluster": "8"},
    {"cq": "Welche rechtlichen Maßstäbe sind anzuwenden, um zu entscheiden, ob Lärm eine Mietminderung rechtfertigt?", "cluster": "8"},
    {"cq": "Unter welchen Voraussetzungen muss eine Krankenversicherung die Kosten für eine künstlichen Befruchtung als medizinisch notwendige Heilbehandlung erstatten?", "cluster": "8"},
    {"cq": "Unter welchen Voraussetzungen und welche Rechte hat ein Miterfinder an einer Erfindung, insbesondere im Hinblick auf die Lizenzierung, Vergütung und Nutzung der Erfindung durch andere Beteiligten?", "cluster": "12"},
    {"cq": "Unter welchen Voraussetzungen kann ein Kunde einen Anspruch auf Rückzahlung von Zahlungen aufgrund unwirksamer Gaspreiserhöhungen geltend machen?", "cluster": "15"}
]

# Load model and embeddings
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
cq_embeddings = model.encode([q['cq'] for q in cq_data], convert_to_tensor=True)

# Logos at top right
_, col1, col2 = st.columns([8,1,1], gap="small")
with col1:
    st.image("pictures/logoDatev.png", width=80)
with col2:
    st.image("pictures/logoDatavation.png", width=80)
_with_, col1, col2 = None, _, _  # keep cols alignment
# Title and input
st.title("Legal Assistant Dashboard")
query = st.text_input("Enter your legal question:")

# Function to show ontology graph
def show_graph(ontology):
     net = Network(height="500px", width="100%", bgcolor="#000000", font_color="#000000")
     for node in ontology.get("refined_ontology", {}).get("classes", []):
         net.add_node(node, label=node, color="#E862A1")
     for subj, pred, obj in ontology.get("refined_ontology", {}).get("relationships", []):
         net.add_edge(subj, obj, label=pred, color="#70BA30")
     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
     net.save_graph(tmp.name)
     html = open(tmp.name, encoding="utf-8").read()
     components.html(html, height=550)

# Find and display similar questions
if query:
    emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(emb, cq_embeddings)[0]
    top5 = np.argsort(-scores.cpu())[:5]
    st.subheader("Similar Competency Questions")
    for i in top5:
        text = cq_data[int(i)]['cq']
        if st.button(text, key=f"btn_{i}"):
            st.markdown(f"**Selected Question:** {text}")
            label = f"Cluster {cq_data[int(i)]['cluster']}"
            ont = ontology_data.get(label)
            if ont:
                st.subheader("Related Ontology")
                show_graph(ont)
            else:
                st.warning("No ontology found for this cluster.")

# Footer note fixed at bottom left (smaller font)
st.markdown("""
<div style="
    position: fixed;
    bottom: 10px;
    left: 10px;
    font-size: 0.7rem;
    color: #888888;
">
  for more information contact info@datavation.de
</div>
""", unsafe_allow_html=True)
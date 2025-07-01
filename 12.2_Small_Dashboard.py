import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
#streamlit run /Users/jule/Desktop/CapstoneDATEV25/12.2_Small_Dashboard.py
# Page configuration
st.set_page_config(
    page_title="Legal Assistant Dashboard",
    layout="wide"
)

# --- COMPANY INFO PANEL IN SIDEBAR ---
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
  <h3 style="margin: 0; color: #222222;">Datavation</h3>
  <p style="font-size: 0.9rem; color: #444444; margin-top: 0.5rem; margin-bottom: 1.5rem;">
    We are an advanced research tool for legal professionals, engineered to streamline case analysis and make every aspect of legal research faster, more precise, and efficient.
  </p>
  <div style="margin-top: 2rem; color: #000000;">
    <ul style="list-style: none; padding: 0; margin: 0;">
      <li style="margin-bottom: 0.2rem;">► Trained on 6000+ documents</li>
      <li style="margin-bottom: 0.2rem;">► AI powered</li>
      <li style="margin-bottom: 0.2rem;">► Developed by AI professionals and lawyers</li>
    </ul>
  </div>
  <div style="
      position: absolute;
      bottom: 1rem;
      left: 1rem;
      font-size: 0.5rem;
      color: #444444;
  ">
    For more information contact: info@datavation.de
  </div>
</div>
""", unsafe_allow_html=True)

# Global CSS for clean, light modern design
st.markdown("""
<style>
  .reportview-container, .stApp {
    background-color: #FFFFFF;
    color: #333333;
    padding: 2rem;
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  .stApp > header {
    border: none;
    background-color: #FFFFFF;
  }
  .stSidebar {
    background-color: #FFFFFF;
    padding: 0;
  }
  h1, h2, .stSubheader {
    margin-bottom: 0.5rem;
    color: #222222;
    text-align: left !important;
  }
  .stTextInput>div>div>input {
    background-color: #FFFFFF;
    border: 2px solid #70BA30 !important;
    border-radius: 6px;
    padding: 0.5rem;
    color: #333333 !important;
  }
  .stTextInput>div>div>input::placeholder {
    color: #888888 !important;
  }
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

# TOP RIGHT: Logos + taglines
col_main, col_logos = st.columns([8, 2], gap="small")

with col_logos:
    # Logos (shown using st.image for reliability)
    logo_col1, logo_col2 = st.columns(2)
    with logo_col1:
        st.image("pictures/logoDatev.png", width=100)
    with logo_col2:
        st.image("pictures/logoDatavation.png", width=100)

    # Taglines directly below logos
    st.markdown("""
    <div style="text-align: right; line-height: 1.3; margin-top: 0.1rem;">
        <div style="color: #70BA30; font-size: 0.9rem; font-weight: 500;">Shaping the future.</div>
        <div style="color: #E862A1; font-size: 0.9rem; font-weight: 500;">Together.</div>
    </div>
    """, unsafe_allow_html=True)


# Title and input
st.title("Legal Research Assistant Dashboard")
# Subtitle (short explanation)
st.markdown("""
<p style="font-size: 1.1rem; color: #444444; margin-top: -0.5rem; margin-bottom: 1.5rem;">
Explore and understand complex legal documents in seconds. This dashboard lets you explore legal documents in a visual way, so you can find what matters faster and make sense of complex info more easily.
</p>
""", unsafe_allow_html=True)
# Input field with placeholder text
query = st.text_input("", placeholder="Enter your legal question or keywords here")

# Function to show ontology graph
def show_graph(ontology):
     # Initialize directed network with white background and more size
     net = Network(
         height="700px",
         width="100%",
         bgcolor="#FFFFFF",
         font_color="#000000",
         directed=True,
         notebook=False
     )
     # Use physics for better spacing
     net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
     # Add nodes
     for node in ontology.get("refined_ontology", {}).get("classes", []):
         net.add_node(node, label=node, color="#E862A1", font={'size':40})
     # Add directed edges with arrows
     for subj, pred, obj in ontology.get("refined_ontology", {}).get("relationships", []):
         net.add_edge(subj, obj, label=pred, color="#70BA30", arrows='to')
     # Generate HTML and render
     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
     net.save_graph(tmp.name)
     html = open(tmp.name, encoding="utf-8").read()
     components.html(html, height=750)

# Find and display similar questions
if query:
    emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(emb, cq_embeddings)[0]
    top5 = np.argsort(-scores.cpu())[:5]
    st.subheader("Top-5 Most Similar Questions")
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

# Footer note fixed at bottom left
st.markdown("""
<div style="
    position: fixed;
    bottom: 10px;
    left: 10px;
    font-size: 0.5rem;
    color: #888888;
">
  For more information contact: info@datavation.de
</div>
""", unsafe_allow_html=True)

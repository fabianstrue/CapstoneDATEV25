import os
import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stopwords_de = stopwords.words("german")


# === 1. Lade Textdatei ===
input_path = Path("competency_questions_output/competency_questions_all_documents.txt")
with open(input_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# === 2. Zerlege Dokumente ===
documents = re.split(r"\nDokument:\s", full_text)
rthema_to_cqs = defaultdict(list)

for doc in documents:
    if not doc.strip():
        continue
    rthema_match = re.search(r"Rechtsthema:\s(.+)", doc)
    if not rthema_match:
        continue
    rthemen = [t.strip() for t in rthema_match.group(1).split(",")]

    questions = re.findall(r"\*\*Frage:\*\*\s*(.+)", doc)
    for thema in rthemen:
        rthema_to_cqs[thema].extend(questions)

# === 3. Lade Embedding-Modell ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === 4. Erstelle Ausgabeordner ===
os.makedirs("cluster_output_by_rthema", exist_ok=True)

# === 5. Clustering + Labeling pro rthema ===
summary = {}

for thema, questions in rthema_to_cqs.items():
    if len(questions) < 10:
        continue  # nur ausreichend große Gruppen

    print(f"🔹 Clustering für '{thema}' mit {len(questions)} Fragen...")

    # Embeddings & Reduktion
    embeddings = model.encode(questions)
    embeddings = normalize(embeddings)

    reducer = umap.UMAP(n_neighbors=30, n_components=3, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(reduced)

    # Cluster-Ergebnis verarbeiten
    clustered_questions = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clustered_questions[f"Cluster {label}"].append(questions[idx])

    # TF-IDF für Cluster-Label
    vectorizer = TfidfVectorizer(stop_words=stopwords_de)
    vectorizer.fit(questions)
    feature_names = np.array(vectorizer.get_feature_names_out())

    labeled_clusters = {}
    for cluster_id, cqs in clustered_questions.items():
        tfidf = vectorizer.transform(cqs).mean(axis=0).A1
        keywords = feature_names[tfidf.argsort()[-7:][::-1]]
        labeled_clusters[cluster_id] = {
            "size": len(cqs),
            "top_keywords": keywords.tolist(),
            "questions": cqs,
        }

    # Speichere Datei
    out_path = Path("cluster_output_by_rthema") / f"clusters_{thema.replace('/', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled_clusters, f, indent=2, ensure_ascii=False)

    summary[thema] = {
        "clusters": len(labeled_clusters),
        "total_cqs": len(questions)
    }

# === 6. Speichere Übersicht ===
with open("cluster_output_by_rthema/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n✅ Clustering erfolgreich abgeschlossen!")

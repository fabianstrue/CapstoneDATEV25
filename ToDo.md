Relevanz-Score zu CQ ausgeben lassen, um später zu priorisieren

Semantic Duplicate Detection (Sentence-Transformers + cosine > 0.9), nicht nur String-Gleichheit

# Clustering

Nachbearbeitung der Cluster
# Noise-Cluster (label = -1) prüfen:
# - Noise-Subset extrahieren: python<br>noise_mask = (cluster_labels == -1)<br>noise_cqs = cqs_df[noise_mask]
# - Embedding-Abstand zum nächsten Cluster berechnen	Grenzfälle“ erkennen	Für jedes Noise-CQ: 1. Finde k = 10 nächstgelegene Nachbarn jenseits des Noise-Clusters (FAISS/Annoy). 2. Speichere den minimalen Cosine-Abstand d_min
# 3 · Grenzwerte festlegen	Triage automatisieren	- d_min ≤ 0.25 → Quasi-Mitglieder: automatisch dem nächstgelegenen Cluster zuordnen.
4 · Semantische Stichprobe	echte Ausreißer identifizieren	Ziehe aus jeder d_min-Kategorie 10 % Zufalls-CQs für eine Domänen-Expertin. - Markiere „relevant“ / „irrelevant“. - Gib Feedback zum Schwellenwert.
5 · Re-Clustering der Review-Kandidaten	seltene Themen zusammenfassen	Führe auf den Review-Kandidaten hierarchisches Agglomerative Clustering mit sehr kleiner distance_threshold (z.3) durch – häufig entstehen 2-3 Minicluster („Haftungsquote bei Tierhalterhaftung“ etc.).
# 6 · Metadata ergänzen	spätere Nachvollziehbarkeit	- Flag source = noise_promoted oder source = noise_discarded in eurer CQ-Tabelle.
7 · Regression-Test	künftiges LLM-Update absichern	Speichere die „promoted“ Noise-CQs in einer Whitelist. Bei einem neuen LLM-Run wird vor dem Löschen der Noise-CQs geprüft, ob ein String-oder Cosine-Match gegen diese Whitelist existiert; andernfalls Warnung.


Manche Noise-CQs sind lang und komplex (mehrere Nebensätze). Kürze sie testweise oder paraphrasiere mit dem LLM → oft sinkt d_min, und sie erhalten Anschluss an ein bestehendes Cluster.



Ein Rest-Noise-Rate von 10-15 % ist völlig okay; versucht nicht, alles krampfhaft einzusortieren – die Ontologie soll echte, häufige Informationsbedürfnisse priorisieren.


Representative extraction: Mediod oder höchste relevanz


Kritische cq markieren
Metrik	Formel / Tool	Zielwert
Intra-Cluster cosine-Mean	höher besser	> 0.65
Inter-Cluster cosine-Min	niedriger besser	< 0.4
Noise-Rate		< 25 %
Domain-Acceptance	Fachlich bestätigte Cluster ÷ Gesamt	> 80 %







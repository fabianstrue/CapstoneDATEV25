import os
import json
import random
import time
from mistralai import Mistral

# Input + Output paths
input_path = r"Data/clean/filtered_output.json"
output_dir = "competency_questions_output"
os.makedirs(output_dir, exist_ok=True)

# Gemeinsame Ausgabedatei
combined_output_path = os.path.join(output_dir, "competency_questions_all_documents.txt")

# Projektzweck
ontology_purpose = "To build an ontology for legal and regulatory compliance analysis from legal documents, more specifically judgments."

# System-Prompt
system_message = """
You are an experienced ontology and knowledge engineer. Your task is to create competency questions based on documents that will be used in a later stage to create an ontology. Together with the document, you are given a short purpose description for the ontology that must be created.

Remember the definition and characteristics of competency questions:
Competency questions (CQs) are specific questions that an ontology should be able to answer once it is complete. Use them to define the scope and validate the design of the ontology.

Follow these key aspects when generating competency questions:

1.⁠ ⁠Align questions with the ontology purpose: Reflect the domain and purpose, and address critical use cases.
2.⁠ ⁠Make questions clear and unambiguous: Use concise, natural language to avoid misunderstandings.
3.⁠ ⁠Ensure questions are specific and testable: Focus on precise aspects of the domain and make sure they can be validated with data or reasoning.
4.⁠ ⁠Vary and categorize questions: Your questions should vary in structure (!) and begin with different forms (!) such as:
•⁠  ⁠Unter welchen Umständen..., Wie..., Warum..., Was..., Welche..., Wann..., Unter welchen Bedingungen..., Welche Konsequenzen hat..., Was passiert wenn..., Wie wird..., Was ist..., Wie lange dauert es..., Welche Rechte hat... etc.
5.⁠ ⁠Do NOT always start with Wer...

Include a mix of:
•⁠  ⁠Retrieval (e.g. Was ist ein Mietvertrag?)
•⁠  ⁠Reasoning (e.g. Warum ist eine bestimmte Regelung anwendbar?)
•⁠  ⁠Consistency (e.g. Ist eine doppelte Kündigung im selben Zeitraum gültig?)
•⁠  ⁠Actor-based (e.g. Wer ist verantwortlich für die Zustellung der Kündigung?)
•⁠  ⁠Procedural (e.g. Wie wird ein Widerspruch korrekt eingelegt?)


Example output:

Frage: Wann liegt ein wichtiger Grund für eine außerordentliche Kündigung eines Mietverhältnisses vor?
Quelle: ["Ein wichtiger Grund im Sinne des § 543 Abs. 1 BGB liegt vor, wenn dem kündigenden Teil unter Berücksichtigung aller Umstände des Einzelfalls und unter Abwägung der beiderseitigen Interessen die Fortsetzung des Mietverhältnisses nicht zugemutet werden kann."]

Frage: Wie wirkt sich eine schuldhafte Pflichtverletzung eines Gesellschafters auf seine Haftung in der GbR aus?
Quelle: ["Ein Gesellschafter haftet für die von ihm schuldhaft verursachten Schäden gegenüber der Gesellschaft gemäß § 280 Abs. 1 BGB."]

Frage: Was gilt bei der Ermittlung der ortsüblichen Vergleichsmiete im Rahmen eines Mieterhöhungsverlangens?
Quelle: ["Die ortsübliche Vergleichsmiete ist anhand eines qualifizierten Mietspiegels oder durch Sachverständigengutachten zu ermitteln, § 558 BGB."]

Frage: Warum entfällt der Pflichtteilsergänzungsanspruch bei freiwilligem Verzicht auf ein Erbe?
Quelle: ["Ein Pflichtteilsergänzungsanspruch entfällt, wenn ein Pflichtteilsberechtigter wirksam auf seinen Pflichtteil verzichtet hat (§ 2346 BGB)."]

"""

# Prompt-Generator
def build_user_prompt(purpose, text):
    return f"""
Abstract description of the document contents:
These documents are court rulings (Urteile) from the German Federal Court of Justice (Bundesgerichtshof), issued between 2000 and 2020 and classified as higher court jurisprudence (obere Rechtsprechung). 
The rulings cover a variety of legal areas including summarized legal themes, civil procedure law (Zivilverfahrensrecht), general contract law (SchuldrechtAT), damages (Schadensersatz), commercial and corporate law (Handelsrecht Gesellschaftsrecht), general provisions of the German Civil Code (BGBAT), family law (Familienrecht), tenancy and lease law (Miete Pacht), property law (Sachenrecht), insurance law (Versicherungsrecht), purchase, exchange, and leasing (Kauf Tausch Leasing), inheritance and gift law (Erbschaft Schenkung), IT law (EDV-Recht), obligations (Schuldverhältnisse), contracts for work and services (Werkvertrag), residential property law (Wohnungseigentum), miscellaneous law (Sonstiges Recht), and travel contract law (Reisevertrag).

General purpose of the ontology:
{purpose}

Document:
{text[:10000]}

Please generate 5-10 competency questions in German that reflects a specific legal obligation, condition, actor or rule. Use different question forms like Wer, Was, Wie, Warum, Wann etc. Do not repeat structure.

Your task:
•⁠  ⁠Generate 5-10 competency questions (CQs) in German based on this document.
•⁠  ⁠For each CQ, include the original sentence(s) you used as justification.
•⁠  ⁠Focus on legal reasoning by formulating questions that involve legal interpretation, applicability of rules, conditions, or exceptions.
•⁠  ⁠Include obligations by addressing duties, rights, or responsibilities of parties involved.
•⁠  ⁠Use facts by referencing concrete legal situations, case details, or normative statements from the document.
•⁠  ⁠Only list questions and citations. Do not explain or comment.
"""

# JSON laden
with open(input_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Zufällige Auswahl von 20 Dokumenten verarbeiten
random.seed(69)
sampled_documents = dict(random.sample(list(documents.items()), 20))

# Bestehende Datei leeren
with open(combined_output_path, "w", encoding="utf-8") as f:
    f.write("")

# Dokumente verarbeiten
for i, (doc_id, doc_data) in enumerate(sampled_documents.items()):
    # Extrahiere relevante Textabschnitte aus dem Urteil
    text_data = doc_data.get("text", {}).get("entscheidungsinhalt", {})
    # Leitsatz inkludieren?
    tenor = text_data.get("tenor", "")
    tatbestand = text_data.get("tatbestand", "")
    gruende_list = text_data.get("gruende", {}).get("gruende", [])

    # Alle Texte kombinieren, inklusive .get("#text")-Fix für dicts
    text_parts = [
        "Entscheidungsformel:\n" + tenor.strip() if isinstance(tenor, str) else "",
        "Sachverhalt:\n" + tatbestand.get("#text", "").strip() if isinstance(tatbestand, dict) else tatbestand.strip() if isinstance(tatbestand, str) else "",
        "Begründung:\n" + "\n".join(gruende_list).strip() if gruende_list else ""
    ]

    text = "\n\n".join(part for part in text_parts if part).strip()

    if not text:
        print(f"Skipping {doc_id} (no usable text)")
        continue

    user_prompt = build_user_prompt(ontology_purpose, text)
    api_key = "ujI60UR6Fe5jel48SAtfnMiN5Skxfwhq"
    model = "open-mistral-nemo"
    client = Mistral(api_key=api_key)

    try:
        response = client.chat.complete(
            model = model,
            random_seed = 69,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature = 0.69,
        )

        output = response.choices[0].message.content.strip()

        # Nur einfache Formatbereinigung
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        frage = next((l for l in lines if l.lower().startswith("frage:")), lines[0])
        quelle = next((l for l in lines if "quelle" in l.lower()), "(Quelle: Unbekannt)")

        formatted_entry = f"Dokument: {doc_id}\n{frage}\n{quelle}\n\n"

# Speichern
        with open(combined_output_path, "a", encoding="utf-8") as f_out:
            f_out.write(formatted_entry)

        print(f"Gespeichert: {doc_id}")
        time.sleep(0.1)  # Kurze Pause nach jedem API-Call

    except Exception as e:
        print(f"Fehler bei {doc_id}: {e}")
"""
# Erster Versuch für den Loop, funktioniert aber noch nicht:
# Nur einfache Formatbereinigung
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        fragen = []
        quellen = []

# Iteriere durch die Zeilen und finde CQ/Frage + Quelle
        for idx, line in enumerate(lines):
            if line.lower().startswith("frage:"):
                frage = line
                # Nächste Zeile mit "Quelle" finden
                quelle = "(Quelle: Unbekannt)"
                for j in range(idx + 1, len(lines)):
                    if "quelle" in lines[j].lower():
                        quelle = lines[j]
                        break
                fragen.append((frage, quelle))

        # Formatierter Eintrag für alle CQs des Dokuments
        formatted_entry = f"Dokument: {doc_id}\n"
        for i, (frage, quelle) in enumerate(fragen, 1):
            formatted_entry += f"{i}. {frage}\n{quelle}\n\n"
"""


import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# ------------------------------
# Instellingen en OpenAI setup
# ------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"

# ------------------------------
# Genereer embeddings voor vragen
# ------------------------------
def get_embeddings(texts):
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(input=batch, model=EMBED_MODEL)
        for item in response.data:
            embeddings.append(item.embedding)
    return np.array(embeddings)

# ------------------------------
# Cluster vergelijkbare vragen
# ------------------------------
def cluster_questions(embeddings, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# ------------------------------
# Genereer FAQ per cluster
# ------------------------------
def generate_faq_question_answer(group):
    vragen = group["Omschrijving"].tolist()
    antwoorden = group["Antwoord"].tolist()
    prompt = f"""
U bent helpdeskmedewerker. Onderstaande meldingen gaan over hetzelfde onderwerp.
Vat ze samen in één duidelijke vraag, gevolgd door een helder antwoord in de u-vorm, in begrijpelijke taal zonder jargon.

Meldingen:
{chr(10).join(['- ' + v for v in vragen[:10]])}

Antwoorden:
{chr(10).join(['- ' + a for a in antwoorden[:10]])}

Genereer één duidelijke FAQ-vraag met daarna een volledig antwoord dat begrijpelijk is voor een vrijwilliger of medewerker zonder veel digitale kennis.
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    output = response.choices[0].message.content
    if "Antwoord:" in output:
        vraag, antwoord = output.split("Antwoord:", 1)
        return vraag.strip(), antwoord.strip()
    else:
        return output.strip(), ""

# ------------------------------
# Bouw de FAQ-dataset
# ------------------------------
def build_faq(df, labels):
    df["cluster"] = labels
    faq_data = []
    for cluster_id, group in df.groupby("cluster"):
        vraag, antwoord = generate_faq_question_answer(group)
        faq_data.append({
            "Systeem": "",
            "Subthema": "",
            "Categorie": "",
            "Omschrijving melding": vraag,
            "Toelichting melding": "",
            "Soort melding": "FAQ gegenereerd",
            "Antwoord of oplossing": antwoord,
            "Afbeelding": ""
        })
    return pd.DataFrame(faq_data)

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("IPAL FAQ Generator (Streamlit-versie)")

uploaded_file = st.file_uploader("Upload tickets.xlsx (met kolommen 'Omschrijving' en 'Antwoord')", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if "Omschrijving" not in df.columns or "Antwoord" not in df.columns:
        st.error("Bestand moet kolommen bevatten met de namen 'Omschrijving' en 'Antwoord'")
    else:
        st.success(f"{len(df)} regels geladen. Start verwerking hieronder.")
        if st.button("Genereer FAQ"):
            with st.spinner("Verwerken en genereren van FAQ..."):
                embeddings = get_embeddings(df["Omschrijving"].tolist())
                labels = cluster_questions(embeddings, n_clusters=30)
                faq_df = build_faq(df, labels)
                st.success("FAQ gegenereerd!")
                st.download_button(
                    label="Download FAQ als Excel",
                    data=faq_df.to_excel(index=False),
                    file_name="faq_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

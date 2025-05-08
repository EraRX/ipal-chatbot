import os
import io
import traceback
import base64
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import openai

# ───────────────────────── API Key ─────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(page_title="IPAL Helpdesk Chatbot", layout="wide")

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
:root {
  --accent: #2A44AD;
  --accent2: #5A6AF4;
  --bg: #ADD8E6;
  --text: #0F274A;
  --white: #ffffff;
  --shadow: rgba(0, 0, 0, 0.1);
}
html, body, .stApp {
  background-color: var(--bg);
  font-family: 'Inter', sans-serif;
}
.topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  padding: 10px 20px;
  border-radius: 0 0 16px 16px;
  box-shadow: 0 4px 12px var(--shadow);
}
.topbar img {
  height: 40px;
  margin: 4px 12px;
}
.topbar h1 {
  flex-basis: 100%;
  text-align: center;
  color: white;
  font-size: 1.5rem;
  margin-top: 10px;
}
.card {
  background: var(--white);
  border: 1px solid #ccc;
  padding: 16px;
  border-radius: 10px;
  margin-bottom: 20px;
  box-shadow: 0 2px 6px var(--shadow);
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Header ─────────────────────────
ip = base64.b64encode(open("logo.png","rb").read()).decode()
db = base64.b64encode(open("logo-docbase-icon.png","rb").read()).decode()
ex = base64.b64encode(open("Exact.png","rb").read()).decode()
header = f"""
<div class='topbar'>
  <img src='data:image/png;base64,{ip}' alt='IPAL'>
  <img src='data:image/png;base64,{db}' alt='DocBase'>
  <img src='data:image/png;base64,{ex}' alt='Exact'>
  <h1>🔍 IPAL Helpdesk Chatbot</h1>
</div>
"""
st.markdown(header, unsafe_allow_html=True)

# ───────────────────────── Excel inlezen ─────────────────────────
try:
    df = pd.read_excel("faq.xlsx")
except Exception:
    st.error("Fout bij inlezen van faq.xlsx")
    st.stop()

cols = ['Systeem','Subthema','Categorie','Omschrijving melding','Toelichting melding','Soort melding','Antwoord of oplossing']
df['zoek'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)

# ───────────────────────── Chatbox UI ─────────────────────────
st.markdown("---")
st.subheader("💬 Chat met de IPAL Helpdesk")
vraag = st.text_input("Stel hier je vraag over Exact of DocBase")

if vraag:
    # Zoek in df
    hits = df[df['zoek'].str.contains(vraag, case=False, na=False)]

    if hits.empty:
        st.warning("⚠️ Geen antwoord gevonden in FAQ. Probeer je vraag specifieker te stellen.")
    else:
        top3 = hits['Antwoord of oplossing'].dropna().head(3).tolist()
        context = "\n".join(f"- {a}" for a in top3)

        system_prompt = "Je bent een helpdeskassistent. Geef korte, duidelijke antwoorden op basis van de voorbeelden." 
        user_prompt = f"Vraag: {vraag}\n\nAntwoorden in FAQ:\n{context}\n\nAntwoord:" 

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=300
            )
            antwoord = resp.choices[0].message.content.strip()
            st.markdown(f"""
<div class='card'>
<strong>💬 Antwoord:</strong><br>
{antwoord}
</div>
""", unsafe_allow_html=True)
        except Exception as e:
            st.error("Fout bij genereren van antwoord via OpenAI")
            st.code(traceback.format_exc())

# ───────────────────────── Vrij zoeken fallback ─────────────────────────
st.markdown("---")
st.subheader("📚 Doorzoek handmatig de FAQ")
zoekterm = st.text_input("🔍 Zoek in de volledige FAQ")
if zoekterm:
    res = df[df['zoek'].str.contains(zoekterm, case=False, na=False)]
    st.write(f"{len(res)} resultaat/resultaten gevonden:")
    for _, r in res.iterrows():
        st.markdown(f"""
<div class='card'>
<strong>💬 Antwoord:</strong><br>{r['Antwoord of oplossing'] or '-'}<hr>
📁 <b>Systeem:</b> {r['Systeem']}<br>
🗂️ <b>Subthema:</b> {r['Subthema']}<br>
📌 <b>Categorie:</b> {r['Categorie']}<br>
📝 <b>Omschrijving:</b> {r['Omschrijving melding']}<br>
ℹ️ <b>Toelichting:</b> {r['Toelichting melding']}<br>
🏷️ <b>Soort melding:</b> {r['Soort melding']}
</div>
""", unsafe_allow_html=True)

    buf = io.BytesIO()
    res.drop(columns=['zoek'], errors='ignore').to_excel(buf, index=False)
    st.download_button("📥 Download resultaten als Excel", buf.getvalue(), "zoekresultaten.xlsx")

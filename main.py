import os
import io
import traceback
import base64
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import openai

# ───────────────────────── API Key voor Streamlit Cloud ─────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ───────────────────────── Streamlit Pagina ─────────────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ───────────────────────── CSS & HEADER ─────────────────────────
css = '''
<style>
:root {
  --accent: #2A44AD;
  --accent2: #5A6AF4;
  --page-bg: #ADD8E6;
  --card-bg: #FFFFFF;
  --text-color: #0F274A;
  --border-color: #E0E0E0;
}
html, body, .stApp {
  background-color: var(--page-bg) !important;
  color: var(--text-color) !important;
  font-family: 'Inter', sans-serif;
}
.topbar {
  width: 100vw; position: relative; left: 50%; transform: translateX(-50%);
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  padding: 10px 20px; display:flex; flex-wrap:wrap;
  align-items:center; justify-content:center; gap:12px;
  border-radius:0 0 20px 20px; box-shadow:0 4px 10px rgba(0,0,0,0.12);
}
.topbar img { height:48px; }
.topbar h1 {
  flex:1 1 auto; text-align:center; color:#FFF;
  font-size:1.6rem; font-weight:600; margin:0;
}
.stSelectbox>div, .stTextInput>div>div {
  background-color:#FFF !important;
  color: var(--text-color) !important;
  border:1px solid var(--border-color) !important;
  border-radius:8px; padding:6px 10px;
}
.stSelectbox label, .stTextInput label, .stRadio label {
  color: var(--accent) !important; font-weight:600;
}
.stButton>button, .stDownloadButton>button {
  background-color: var(--accent) !important;
  color:#FFF !important; font-weight:600;
  border-radius:8px !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background-color: var(--accent2) !important;
}
.card {
  background: var(--card-bg);
  border:1px solid var(--border-color);
  border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);
  padding:16px; margin:16px 0;
}
</style>
''' 
st.markdown(css, unsafe_allow_html=True)

# Logo's in header
ip = base64.b64encode(open("logo.png","rb").read()).decode()
db = base64.b64encode(open("logo-docbase-icon.png","rb").read()).decode()
ex = base64.b64encode(open("Exact.png","rb").read()).decode()
header_html = f'''<div class="topbar">
  <img src="data:image/png;base64,{ip}" alt="IPAL">
  <img src="data:image/png;base64,{db}" alt="DocBase">
  <img src="data:image/png;base64,{ex}" alt="Exact">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>'''
st.markdown(header_html, unsafe_allow_html=True)

# ───────────────────────── Authentication ─────────────────────────
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title('🔐 Helpdesk Toegang')
    pwd = st.text_input('Voer wachtwoord in:', type='password')
    if not pwd:
        st.stop()
    if pwd != 'ipal2024':
        st.error('Onjuist wachtwoord.')
        st.stop()
    # correct
    st.session_state.auth = True
    st.success('Toegang verleend! De FAQ wordt geladen...')
    st.stop()

# ───────────────────────── Data Loading ─────────────────────────
try:
    df = pd.read_excel('faq.xlsx')
except Exception:
    st.error('Fout bij inlezen van faq.xlsx')
    st.code(traceback.format_exc())
    st.stop()

# Combineer kolommen voor zoek
cols = ['Systeem','Subthema','Categorie','Omschrijving melding','Toelichting melding','Soort melding','Antwoord of oplossing']
df['zoek'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)

# ───────────────────────── Search Interface ─────────────────────────
st.markdown('---')
mode = st.radio('🔎 Kies zoekmethode', ['🎯 Gefilterd', '🔍 Vrij zoeken'], horizontal=True)

if mode == '🎯 Gefilterd':
    st.subheader('🎯 Gefilterde zoekopdracht')
    temp = df.copy()
    for field in cols[:-1]:
        sel = st.selectbox(field, sorted(temp[field].dropna().unique()), key=field)
        temp = temp[temp[field] == sel]
    results = temp
else:
    st.subheader('🔍 Vrij zoeken')
    q = st.text_input('Zoekterm:')
    results = df[df['zoek'].str.contains(q, case=False, na=False)] if q else pd.DataFrame()

# ───────────────────────── Results Display ─────────────────────────
if results.empty:
    st.info('Geen resultaten gevonden.')
else:
    st.write(f'### 📄 {len(results)} resultaat/resultaten gevonden')
    for _, row in results.iterrows():
        card_html = f"""
<div class='card'>
  <strong>💬 Antwoord:</strong><br>{row['Antwoord of oplossing'] or '-'}<hr>
  📁 <b>Systeem:</b> {row['Systeem']}<br>
  🗂️ <b>Subthema:</b> {row['Subthema']}<br>
  📌 <b>Categorie:</b> {row['Categorie']}<br>
  📝 <b>Omschrijving melding:</b> {row['Omschrijving melding']}<br>
  ℹ️ <b>Toelichting:</b> {row['Toelichting melding']}<br>
  🏷️ <b>Soort:</b> {row['Soort melding']}
</div>"""
        height = 260 + (len(str(row['Antwoord of oplossing']))//80)*18
        components.html(card_html, height=height, scrolling=False)

    # RAG: natuurlijke taal generatie
    vraag = st.text_input('🗣️ Formuleer je vraag in eigen woorden:')
    if vraag:
        top3 = results['Antwoord of oplossing'].dropna().head(3).tolist()
        context = "\n".join(f"- {a}" for a in top3)
        system = "Je bent een helpdeskassistent. Geef een helder, vloeiend antwoord in volledige zinnen." 
        user = f"Gebruikersvraag:\n{vraag}\n\nKennis uit FAQ:\n{context}\n\nFormuleer een zelfstandig antwoord op de vraag."
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user},
                ],
                temperature=0.3,
                max_tokens=300
            )
            antwoord = resp.choices[0].message.content.strip()
            st.markdown("**💬 Geformuleerd antwoord:**")
            st.write(antwoord)
        except Exception as e:
            st.error("Fout bij genereren van antwoord.")
            st.code(traceback.format_exc())

# ───────────────────────── Download ─────────────────────────
if mode == '🔍 Vrij zoeken' and not results.empty:
    buf = io.BytesIO()
    results.drop(columns=['zoek'], errors='ignore').to_excel(buf, index=False)
    st.download_button('📥 Download Excel', buf.getvalue(), 'zoekresultaten.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

import streamlit as st
import pandas as pd
import base64, io, traceback
import streamlit.components.v1 as components

# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ───────────────────────── Assets Loading ─────────────────────────
ipal_logo  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc_logo   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ───────────────────────── Header & Global CSS ─────────────────────────
components.html(f"""
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
  --accent: #2A44AD;
  --accent2: #5A6AF4;
  --page-bg: #EAF2FF;
  --card-bg: #FFFFFF;
  --text: #0F274A;
  --border: #E0E0E0;
}}
html, body, .stApp {{
  margin: 0; padding: 0;
  background: var(--page-bg);
  font-family: 'Inter', sans-serif;
  color: var(--text);
}}
.topbar {{
  position: relative; left: 50%; transform: translateX(-50%);
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  padding: calc(10px + env(safe-area-inset-top)) 24px 10px;
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: 20px; border-radius: 0 0 20px 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}
.topbar img {{ height: 60px; }}
.topbar h1 {{
  flex-basis: 100%; text-align: center;
  color: #FFFFFF; margin: 4px 0 0;
  font-size: 1.8rem; font-weight: 600;
}}
@media (min-width: 768px) {{ .topbar h1 {{ flex-basis: auto; margin-left: 18px; }} }}
.stSelectbox > div {{
  background: #000000 !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px;
  padding: 8px 12px;
}}
.stSelectbox option {{ background: #000000; color: var(--accent) !important; }}
.stSelectbox label, .stTextInput label, .stRadio label {{ color: var(--accent) !important; font-weight: 600; }}
.stButton > button, .stDownloadButton > button {{
  background: var(--accent) !important;
  color: #FFFFFF !important;
  font-weight: 600;
  border-radius: 8px !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: var(--accent2) !important; }}
.card {{
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.04);
  padding: 22px;
  margin: 20px 0;
}}
</style>
<div class="topbar">
  <img src="data:image/png;base64,{ipal_logo}" alt="IPAL">
  <img src="data:image/png;base64,{doc_logo}" alt="DocBase">
  <img src="data:image/png;base64,{exact_logo}" alt="Exact">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
""", height=140, scrolling=False)

# ───────────────────────── Authentication ─────────────────────────
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title('🔐 Helpdesk Toegang')
    pw = st.text_input('Voer wachtwoord in:', type='password')
    if pw == 'ipal2024':
        st.session_state.auth = True
    else:
        if pw:
            st.error('Onjuist wachtwoord.')
        st.stop()

# ───────────────────────── Data Loading ─────────────────────────
try:
    df = pd.read_excel('faq.xlsx')
except Exception:
    st.error('Fout bij inlezen van faq.xlsx')
    st.code(traceback.format_exc())
    st.stop()

cols = ['Systeem','Subthema','Categorie','Omschrijving melding','Toelichting melding','Soort melding','Antwoord of oplossing']
df['zoek'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)

# ───────────────────────── Search Interface ─────────────────────────
st.markdown('---')
mode = st.radio('🔎 Kies zoekmethode', ['🎯 Gefilterd', '🔍 Vrij zoeken'], horizontal=True)

if mode == '🎯 Gefilterd':
    st.subheader('🎯 Gefilterde zoekopdracht')
    tmp = df.copy()
    for field in cols[:-1]:  # exclude answer
        sel = st.selectbox(field, sorted(tmp[field].dropna().unique()), key=field)
        tmp = tmp[tmp[field] == sel]
    res = tmp
else:
    st.subheader('🔍 Vrij zoeken')
    q = st.text_input('Zoekterm:')
    res = df[df['zoek'].str.contains(q, case=False, na=False)] if q else pd.DataFrame()

# ───────────────────────── Results Display ─────────────────────────
if res.empty:
    st.info('Geen resultaten gevonden.')
else:
    st.write(f'### 📄 {len(res)} resultaat/resultaten gevonden')
    def render(r):
        html = f"""
<div class='card'>
  <strong>💬 Antwoord:</strong><br>{r['Antwoord of oplossing'] or '-'}<hr>
  📁 <b>Systeem:</b> {r['Systeem']}<br>
  🗂️ <b>Subthema:</b> {r['Subthema']}<br>
  📌 <b>Categorie:</b> {r['Categorie']}<br>
  📝 <b>Omschrijving melding:</b> {r['Omschrijving melding']}<br>
  ℹ️ <b>Toelichting melding:</b> {r['Toelichting melding']}<br>
  🏷️ <b>Soort melding:</b> {r['Soort melding']}
</div>"""
        h = 260 + (len(str(r['Antwoord of oplossing'])) // 80) * 18
        components.html(html, height=h, scrolling=False)
    res.apply(render, axis=1)

# ───────────────────────── Download Button ─────────────────────────
if mode == '🔍 Vrij zoeken' and not res.empty:
    buffer = io.BytesIO()
    res.drop(columns=['zoek'], errors='ignore').to_excel(buffer, index=False)
    st.download_button('📥 Download Excel', buffer.getvalue(), 'zoekresultaten.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

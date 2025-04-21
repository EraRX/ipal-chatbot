import streamlit as st
import pandas as pd
import base64, io, traceback
import streamlit.components.v1 as components

# ─────────────────── Page Config ───────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ─────────────────── CSS + Header ───────────────────
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
#root > div:nth-child(1) {
  padding-top: env(safe-area-inset-top);
}
.topbar {
  width: 100vw;
  position: relative; left: 50%; transform: translateX(-50%);
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  padding: 10px 20px;
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 12px;
  border-radius: 0 0 20px 20px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}
.topbar img { height: 48px; }
.topbar h1 {
  flex: 1 1 auto; text-align: center; color: #fff;
  font-size: 1.6rem; font-weight: 600; margin: 0;
}
.stSelectbox>div, .stTextInput>div>div {
  background-color: #FFFFFF !important;
  color: var(--text-color) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px; padding: 6px 10px;
}
.stSelectbox label, .stTextInput label, .stRadio label {
  color: var(--accent) !important; font-weight: 600;
}
.stButton>button, .stDownloadButton>button {
  background-color: var(--accent) !important; color: #fff !important;
  font-weight: 600; border-radius: 8px !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background-color: var(--accent2) !important;
}
.card {
  background: var(--card-bg); border: 1px solid var(--border-color);
  border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  padding: 16px; margin: 16px 0;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Header logos
dir_ipal = "logo.png"
dir_doc  = "logo-docbase-icon.png"
dir_ex   = "Exact.png"
ipal_logo  = base64.b64encode(open(dir_ipal, "rb").read()).decode()
doc_logo   = base64.b64encode(open(dir_doc,  "rb").read()).decode()
exact_logo = base64.b64encode(open(dir_ex,   "rb").read()).decode()
header_html = f"""
<div class='topbar'>
  <img src='data:image/png;base64,{ipal_logo}' alt='IPAL'>
  <img src='data:image/png;base64,{doc_logo}' alt='DocBase'>
  <img src='data:image/png;base64,{exact_logo}' alt='Exact'>
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────── Authentication ───────────────────
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title('🔐 Helpdesk Toegang')
    pwd = st.text_input('Voer wachtwoord in:', type='password')
    if pwd:
        if pwd == 'ipal2024':
            st.session_state.auth = True
            st.experimental_rerun()
        else:
            st.error('Onjuist wachtwoord.')
    st.stop()

# ─────────────────── Data Loading ───────────────────
try:
    df = pd.read_excel('faq.xlsx')
except Exception:
    st.error('Fout bij inlezen van faq.xlsx')
    st.code(traceback.format_exc())
    st.stop()

cols = ['Systeem','Subthema','Categorie','Omschrijving melding','Toelichting melding','Soort melding','Antwoord of oplossing']
df['zoek'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)

# ─────────────────── Search Interface ───────────────────
st.markdown('---')
mode = st.radio('🔎 Kies zoekmethode', ['🎯 Gefilterd', '🔍 Vrij zoeken'], horizontal=True)

if mode == '🎯 Gefilterd':
    st.subheader('🎯 Gefilterde zoekopdracht')
    tmp = df.copy()
    for field in cols[:-1]:
        sel = st.selectbox(field, sorted(tmp[field].dropna().unique()), key=field)
        tmp = tmp[tmp[field] == sel]
    res = tmp
else:
    st.subheader('🔍 Vrij zoeken')
    q = st.text_input('Zoekterm:')
    res = df[df['zoek'].str.contains(q, case=False, na=False)] if q else pd.DataFrame()

# ─────────────────── Results Display ───────────────────
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
  ℹ️ <b>Toelichting:</b> {r['Toelichting melding']}<br>
  🏷️ <b>Soort melding:</b> {r['Soort melding']}
</div>"""
        height = 260 + (len(str(r['Antwoord of oplossing'])) // 80) * 18
        components.html(html, height=height, scrolling=False)
    res.apply(render, axis=1)

# ─────────────────── Download Button ───────────────────
if mode == '🔍 Vrij zoeken' and not res.empty:
    buffer = io.BytesIO()
    res.drop(columns=['zoek'], errors='ignore').to_excel(buffer, index=False)
    st.download_button('📥 Download Excel', buffer.getvalue(), 'zoekresultaten.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

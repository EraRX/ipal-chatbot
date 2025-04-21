import streamlit as st
import pandas as pd
import base64, io, traceback
import streamlit.components.v1 as components

# ─────────────────────────── Page Config ──────────────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ─────────────────────────── Assets ────────────────────────────
ipal_logo  = base64.b64encode(open("logo.png",              "rb").read()).decode()
doc_logo   = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png",             "rb").read()).decode()

# ─────────────────────────── Header & CSS ─────────────────────────
components.html(f"""
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --accent: #2A44AD;
  --accent2: #5A6AF4;
  --bg: #FFFFFF;
  --page-bg: #F3F6F9;
  --text: #0F274A;
  --border: #E0E0E0;
}
html, body, .stApp {
  margin: 0; padding: 0;
  background: var(--page-bg);
  font-family: 'Inter', sans-serif;
  color: var(--text);
}
.topbar {
  position: relative; left: 50%; transform: translateX(-50%);
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  padding: calc(14px + env(safe-area-inset-top)) 24px 14px;
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: 20px; border-radius: 0 0 20px 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.topbar img { height: 60px; }
.topbar h1 {
  flex-basis: 100%; text-align: center;
  color: #FFFFFF; margin: 6px 0 0;
  font-size: 2rem; font-weight: 600;
}
@media (min-width: 768px) {
  .topbar h1 { flex-basis: auto; margin-left: 18px; }
}
.stSelectbox > div {
  background: var(--bg);
  border: 1px solid var(--border) !important;
  border-radius: 12px;
  padding: 8px 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.stSelectbox label, .stTextInput label, .stRadio label {
  color: var(--accent) !important;
  font-weight: 600;
}
.stButton > button, .stDownloadButton > button {
  background: var(--accent);
  color: #FFFFFF !important;
  font-weight: 600;
  border-radius: 8px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--accent2);
}
.card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
  padding: 22px;
  margin: 20px 0;
}
</style>
<div class="topbar">
  <img src="data:image/png;base64,{ipal_logo}" alt="IPAL">
  <img src="data:image/png;base64,{doc_logo}" alt="DocBase">
  <img src="data:image/png;base64,{exact_logo}" alt="Exact">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
""", height=140, scrolling=False);
  background:linear-gradient(135deg,#2A44AD 0%,#5A6AF4 100%);
  padding:calc(14px + env(safe-area-inset-top)) 24px 14px;
  display:flex; flex-wrap:wrap; align-items:center; justify-content:center;
  gap:20px; border-radius:0 0 20px 20px; box-shadow:0 4px 12px rgba(0,0,0,.15);
}}
.topbar img {{height:60px;}}
.topbar h1 {{flex-basis:100%; text-align:center; color:#fff; margin:6px 0 0; font-size:2rem; font-weight:600;}}
@media(min-width:768px) {{ .topbar h1 {{flex-basis:auto; margin-left:18px;}} }}
.stSelectbox>div {{border:none!important; border-radius:12px; padding:8px 12px; box-shadow:0 2px 8px rgba(0,0,0,.1); background:#fff;}}
.stSelectbox label, .stTextInput label, .stRadio label {{color:#2A44AD!important; font-weight:600;}}
.stButton>button, .stDownloadButton>button {{background:#2A44AD!important; color:#fff!important; font-weight:600!important; border-radius:8px!important;}}
.stButton>button:hover, .stDownloadButton>button:hover {{background:#1B2E8A!important;}}
.card {{background:#FFFFFF; border-radius:16px; box-shadow:0 4px 14px rgba(0,0,0,.08); padding:22px; margin:20px 0;}}
</style>
<div class="topbar">
  <img src="data:image/png;base64,{ipal_logo}" alt="IPAL logo">
  <img src="data:image/png;base64,{doc_logo}" alt="DocBase logo">
  <img src="data:image/png;base64,{exact_logo}" alt="Exact logo">
  <h1>🔍 Helpdesk Zoekfunctie</h1>
</div>
""", height=140, scrolling=False)

# ─────────────────────────── Login ───────────────────────────
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title('🔐 Helpdesk Toegang')
    pw = st.text_input('Voer wachtwoord in:', type='password')
    if pw == 'ipal2024':
        st.session_state.auth = True
        st.success('Toegang verleend. Gebruik de zoekinterface hieronder.')
    else:
        if pw:
            st.error('Onjuist wachtwoord.')
        st.stop()

# ─────────────────────────── Data Loading ─────────────────────────
try:
    df = pd.read_excel('faq.xlsx')
except Exception:
    st.error('Fout bij inlezen van faq.xlsx')
    st.code(traceback.format_exc())
    st.stop()

# Maak zoekkolom
cols = ['Systeem','Subthema','Categorie','Omschrijving melding','Toelichting melding','Soort melding','Antwoord of oplossing']
df['zoek'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)

# ─────────────────────────── Search UI ─────────────────────────
st.markdown('---')
mode = st.radio('🔎 Kies zoekmethode', ['🎯 Gefilterd','🔍 Vrij zoeken'], horizontal=True)

if mode == '🎯 Gefilterd':
    st.subheader('🎯 Gefilterde zoekopdracht')
    temp = df.copy()
    for field,label in [('Systeem','Systeem'),('Subthema','Subthema'),('Categorie','Categorie'),
                        ('Omschrijving melding','Omschrijving melding'),('Toelichting melding','Toelichting melding'),
                        ('Soort melding','Soort melding')]:
        val = st.selectbox(label, sorted(temp[field].dropna().unique()), key=field)
        temp = temp[temp[field]==val]
    res = temp
else:
    st.subheader('🔍 Vrij zoeken in alle velden')
    q = st.text_input('Zoekterm:')
    res = df[df['zoek'].str.contains(q, case=False, na=False)] if q else pd.DataFrame()

# ─────────────────────────── Results ─────────────────────────
if res.empty:
    st.info('Geen resultaten gevonden.')
else:
    st.write(f'### 📄 {len(res)} resultaat/resultaten gevonden')
    import streamlit.components.v1 as components
    def show(r):
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
        height = 280 + (len(str(r['Antwoord of oplossing']))//80)*18
        components.html(html, height=height, scrolling=False)
    res.apply(show, axis=1)

# ─────────────────────────── Download ─────────────────────────
if mode=='🔍 Vrij zoeken' and not res.empty:
    buf=io.BytesIO()
    res.drop(columns=['zoek'], errors='ignore').to_excel(buf, index=False)
    st.download_button('📥 Download Excel', buf.getvalue(), 'resultaten.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

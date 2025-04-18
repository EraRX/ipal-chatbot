import pandas as pd
import streamlit as st
import traceback, io, base64

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# ─────────────────────────────────────────────────────────────
#  Assets
# ─────────────────────────────────────────────────────────────
ipal_logo   = base64.b64encode(open("logo.png",               "rb").read()).decode()
doc_logo    = base64.b64encode(open("logo-docbase-icon.png",   "rb").read()).decode()
exact_logo  = base64.b64encode(open("Exact.png",              "rb").read()).decode()

# ─────────────────────────────────────────────────────────────
#  Global CSS  ➜  2025 soft‑UI look
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{
  --accent:#2A44AD;          /* primair blauw */
  --accent2:#5A6AF4;         /* gradient blauw‑violet */
  --bg:#EAF3FF;              /* zachte pastel */
  --bg-card:#F7FBFF;         /* kaart‑achtergrond */
  --text:#0F274A;
}}
body,.stApp{{background:var(--bg);font-family:"Inter",sans-serif;color:var(--text);} }

/* Header */
.header-container{{
  width: calc(100% - 48px);
  margin:24px auto 32px;
  background:linear-gradient(135deg,var(--accent)0%,var(--accent2)100%);
  border-radius:32px;box-shadow:0 6px 18px rgba(0,0,0,.12);
  padding:20px 32px;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:28px;
}}
.header-logo{{height:56px;object-fit:contain}}
.header-title{{flex-basis:100%;text-align:center;color:#fff;font-size:2.2rem;font-weight:600;margin:6px 0 0}}
@media(min-width:768px){{.header-title{{flex-basis:auto;margin-left:18px}}}}

/* Radio toggle */
section[data-testid="stHorizontalBlock"] .st-b7 {{gap:22px}} /* spacing radios */

/* Soft dropdown */
.stSelectbox>div{{border:none!important;box-shadow:0 2px 8px rgba(0,0,0,.1);background:#fff;border-radius:14px;padding:10px 14px;}}

/* Labels */
.stSelectbox label,.stTextInput label,.stRadio label{{font-weight:600;color:var(--accent)}}

/* Buttons */
.stDownloadButton button,.stButton button{{background:var(--accent);color:#fff;border-radius:10px;font-weight:600}}
.stDownloadButton button:hover,.stButton button:hover{{background:#1b2e8a}}

/* Result card */
.card{{background:var(--bg-card);border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,.06);padding:22px;margin:22px 0}}
.card mark{{background:#ffeb3b;padding:0 3px;border-radius:3px}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
header_html = f"""
<div class='header-container'>
  <img src='data:image/png;base64,{ipal_logo}'  class='header-logo'>
  <img src='data:image/png;base64,{doc_logo}'   class='header-logo'>
  <img src='data:image/png;base64,{exact_logo}' class='header-logo'>
  <h1 class='header-title'>Helpdesk&nbsp;Zoekfunctie</h1>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth=False

if not st.session_state.auth:
    st.title("🔐 Helpdesk Toegang")
    pw = st.text_input("Voer wachtwoord in:", type="password")
    if pw == "ipal2024":
        st.session_state.auth=True
        st.success("Toegang verleend.")
    elif pw:
        st.error("Onjuist wachtwoord.")
    if not st.session_state.auth:
        st.stop()

# ─────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_data():
    return pd.read_excel("faq.xlsx")

df = load_data()

# Extra kolom voor vrije zoek
df["zoektxt"] = df.apply(lambda r: " ".join(str(x) for x in r), axis=1)

# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
st.markdown("---")
mode = st.radio("🔎 Kies zoekmethode:", ["🎯 Gefilterd","🔍 Vrij"], horizontal=True)

if mode == "🎯 Gefilterd":
    st.subheader("Gefilterde zoekopdracht")
    f = df.copy()
    s = st.selectbox("Systeem",     sorted(f["Systeem"].dropna().unique()))
    f = f[f["Systeem"]==s]

    sub = st.selectbox("Subthema",  sorted(f["Subthema"].dropna().unique()))
    f = f[f["Subthema"]==sub]

    cat = st.selectbox("Categorie", sorted(f["Categorie"].dropna().unique()))
    f = f[f["Categorie"]==cat]

    oms = st.selectbox("Omschrijving melding", sorted(f["Omschrijving melding"].dropna().unique()))
    f = f[f["Omschrijving melding"]==oms]

    tol = st.selectbox("Toelichting melding", sorted(f["Toelichting melding"].dropna().unique()))
    f = f[f["Toelichting melding"]==tol]

    typ = st.selectbox("Soort melding",       sorted(f["Soort melding"].dropna().unique()))
    res = f[f["Soort melding"]==typ]

else:
    st.subheader("Vrij zoeken in alle velden")
    term = st.text_input("Zoekterm")
    res = df[df["zoektxt"].str.contains(term,case=False,na=False)] if term else pd.DataFrame()

# ─────────────────────────────────────────────────────────────
#  RESULTAATWEERGAVE
# ─────────────────────────────────────────────────────────────
if res.empty and mode=="🎯 Gefilterd":
    st.info("Geen resultaat.")
for _,row in res.iterrows():
    st.markdown("""
    <div class='card'>
      <strong>💬 Antwoord:</strong><br>
      <div style='color:var(--accent);font-weight:600;'>""" + (row["Antwoord of oplossing"] or "–") + "</div><hr>
      <ul style='list-style:none;padding:0;margin:0'>
        <li>📁 <b>Systeem:</b> """+row["Systeem"]+"""</li>
        <li>🗂️ <b>Subthema:</b> """+row["Subthema"]+"""</li>
        <li>📌 <b>Categorie:</b> """+row["Categorie"]+"""</li>
        <li>📝 <b>Omschrijving:</b> """+row["Omschrijving melding"]+"""</li>
        <li>ℹ️ <b>Toelichting:</b> """+row["Toelichting melding"]+"""</li>
        <li>🏷️ <b>Soort:</b> """+row["Soort melding"]+"""</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Download (alleen bij vrije zoek zodat f niet leeg is)
if mode == "🔍 Vrij" and not res.empty:
    buff = io.BytesIO(); res.to_excel(buff, index=False)
    st.download_button("📥 Download resultaten", data=buff.getvalue(), file_name="zoekresultaten.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
